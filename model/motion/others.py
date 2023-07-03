import copy

import cv2 as cv
import lietorch
import numpy as np
import open3d as o3d
import torch
from einops import rearrange
from lietorch import SE3
from mmseg.models import builder as builder_oss
from mmseg.models.builder import MODELS
from scipy.spatial.transform import Rotation

from utils import denormalize, merge_seg, compute_valid_mask, meshgrid, merge_flow, target_warp
from utils import projective_ops as pops
from .base import MotionBase
from .geom.ba import moba_segm


def compute_flow(Gs_list, disps, seg, intrinsics, ii, jj, meta, return_target=False):
    flow_list = []
    target_list = []
    for Gs in Gs_list:
        coords, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj, meta)
        seg_ii_binary = seg[:, ii] > 0.75  # BobjxNxHxW, 0.75 for confident estimate only
        coords, valid = merge_flow(seg_ii_binary, coords, valid_mask)

        B, _, H, W = disps.shape
        y, x = torch.meshgrid(
            torch.arange(H).to(disps.device).float(),
            torch.arange(W).to(disps.device).float())
        y = y[None, None].expand(B, ii.shape[0], -1, -1)
        x = x[None, None].expand(B, ii.shape[0], -1, -1)
        pt1 = torch.stack([x, y, disps[:, ii]], dim=-1)

        if return_target:
            target_list.append([coords, valid])
        else:
            flow = (coords - pt1) * valid.float()
            flow_list.append(flow)

    if return_target:
        return target_list
    else:
        return flow_list


@MODELS.register_module()
class GTMotion(MotionBase):
    def __init__(self, config=None, loss=None):
        super(GTMotion, self).__init__()
        self.dummy_param = torch.nn.Parameter(torch.tensor([0.0]))
        self.d_scale = config.get('optimize_d_scale', 100.0)
        self.steps = config.get('steps', 3)
        self.loss = builder_oss.build_loss(loss)
        self.lm = config.get('lm', 0.0001)
        self.ep = config.get('ep', 0.1)
        self.sample_target_disp = config.get('sample_target_disp', False)

    def forward(self, left, pred_disp, pred_semantic_seg, disp_weight, meta, state, outputs, **kwargs):
        B, M, _, H, W = left.shape
        N = M - 1

        pred_disp = self.normalized_disparity(pred_disp, state['intrinsics'], meta['baseline'])

        seg = pred_semantic_seg[..., 3::8, 3::8]
        disps = pred_disp[..., 3::8, 3::8]  # this is normalized, so no need for scaling
        intrinsics = state['intrinsics'] / 8.0
        meta_lr = copy.deepcopy(meta)
        meta_lr['disp_range'] = (meta['disp_range'][0] / 8.0, meta['disp_range'][1] / 8.0)

        ii = torch.arange(M - 1).to(disps.device)
        jj = ii + 1

        # initialize random pose
        Gs = SE3.Identity([B * 2, M], device=disps.device)  # BobjxM

        # move seg objects to batch dim
        H, W = disps.shape[-2:]
        seg = merge_seg(seg)

        flow_weight = torch.ones([B, ii.shape[0], H, W, 3], device=left.device)

        # flow
        Gs_gt = lietorch.cat([state['gt_pose_cam_patient'], state['gt_pose_cam_drill']], dim=0)
        # NOTE: this needs to be of shape BxNxHxWx3, and last dimension is not scaled up
        gt_target = compute_flow([Gs_gt], disps, seg, intrinsics, ii, jj, meta_lr, return_target=True)[-1][0]

        if self.sample_target_disp:
            gt_target = gt_target[..., :2]
            # sample target disp
            target_for_warp = gt_target.view(B * N, H, W, 2)  # BNxHxWx2
            target_for_warp = target_for_warp.permute(0, 3, 1, 2)  # BNx2xHxW
            target_disp = disps[:, jj].view(B * N, 1, H, W)  # BNx1xHxW
            target_disp, _ = target_warp(target_disp, target_for_warp, padding_mode="zeros", mode="nearest")  # BNx1xHxW
            target_disp = target_disp.view(B, N, H, W, 1)  # BxNxHxWx1
            gt_target = torch.cat([gt_target, target_disp], dim=-1)

        # BA
        for i in range(self.steps):
            Gs = moba_segm(gt_target, seg, disp_weight, flow_weight, Gs, disps, intrinsics, ii, jj, meta_lr,
                           d_scale=self.d_scale, lm=self.lm, ep=self.ep)

        target_list = compute_flow([Gs], disps, seg, intrinsics, ii, jj, meta_lr, return_target=True)

        # compute flow
        disps = pred_disp
        intrinsics = state['intrinsics']
        seg = pred_semantic_seg
        seg = merge_seg(seg)

        flow_list = compute_flow([Gs], disps, seg, intrinsics, ii, jj, meta)

        pred_pose_cam_patient = Gs[:B]
        pred_pose_cam_drill = Gs[B:]

        if self.training:
            outputs['pred_poses'] = [Gs]
            outputs['pred_flow'] = flow_list
            outputs['pred_target'] = target_list
        else:
            outputs['pred_pose_cam_patient'] = pred_pose_cam_patient
            outputs['pred_pose_cam_drill'] = pred_pose_cam_drill
            outputs['pred_flow'] = flow_list[-1]
            outputs['pred_target'] = target_list

        return outputs

    def losses(self, loss, output, state, meta, *args):
        Gs_list, flow_list = output['pred_poses'], output['pred_flow']
        residual_list = output.get('pred_target', None)
        disps = output['pred_disp']
        self.loss(Gs_list, residual_list, flow_list, disps, state, loss, meta)
        print("loss", loss)

        loss['loss_motion'] = self.dummy_param - 1.0

        return


@MODELS.register_module()
class KeypointBasedMotion(MotionBase):
    def __init__(self, max_corr=100):
        super(KeypointBasedMotion, self).__init__()
        self.dummy_param = torch.nn.Parameter(torch.tensor([0.0]))
        self.loss = "a random loss"
        self.orb = cv.ORB_create()  # Initiate ORB detector
        self.bf = cv.BFMatcher(cv.NORM_HAMMING, crossCheck=True)  # create BFMatcher object
        self.max_corr = max_corr

    def losses(self, loss, *args):
        loss['loss_motion'] = self.dummy_param - 1.0
        return

    @staticmethod
    def arun(A, B):
        """Solve 3D registration using Arun's method: B = RA + t
        """
        N = A.shape[1]
        assert B.shape[1] == N

        # calculate centroids
        A_centroid = np.reshape(1 / N * (np.sum(A, axis=1)), (3, 1))
        B_centroid = np.reshape(1 / N * (np.sum(B, axis=1)), (3, 1))

        # calculate the vectors from centroids
        A_prime = A - A_centroid
        B_prime = B - B_centroid

        # rotation estimation
        H = np.zeros([3, 3])
        for i in range(N):
            ai = A_prime[:, i]
            bi = B_prime[:, i]
            H = H + np.outer(ai, bi)
        U, S, V_transpose = np.linalg.svd(H)
        V = np.transpose(V_transpose)
        U_transpose = np.transpose(U)
        R = V @ np.diag([1, 1, np.linalg.det(V) * np.linalg.det(U_transpose)]) @ U_transpose

        # translation estimation
        t = B_centroid - R @ A_centroid

        return R, t

    @staticmethod
    def to_3d(pts, intrinsics):
        pts[0, :] = (pts[0, :] - intrinsics[2]) / intrinsics[0] * pts[-1, :]
        pts[1, :] = (pts[1, :] - intrinsics[3]) / intrinsics[1] * pts[-1, :]
        return pts

    @staticmethod
    def check_validity(u0, v0, u1, v1, segm, disp_mask, object_idx):
        valid_seg = segm[0, object_idx, np.rint(v0).astype(int), np.rint(u0).astype(int)] and segm[
            1, object_idx, np.rint(v1).astype(int), np.rint(u1).astype(int)]
        valid_disp = disp_mask[0, np.rint(v0).astype(int), np.rint(u0).astype(int)] and disp_mask[
            1, np.rint(v1).astype(int), np.rint(u1).astype(int)]
        return valid_disp & valid_seg

    def feature_matching(self, img0, img1, disps, intrinsics, segm, disp_mask):
        segm = segm > 0.5

        # find the keypoints and descriptors with ORB
        kp0, des0 = self.orb.detectAndCompute(img0, None)  # query
        kp1, des1 = self.orb.detectAndCompute(img1, None)  # train

        # Match descriptors.
        matches = self.bf.match(des0, des1)
        # Sort them in the order of their distance.
        matches = sorted(matches, key=lambda x: x.distance)

        pts_0_p = []
        pts_1_p = []
        pts_0_d = []
        pts_1_d = []
        valid_matches_patient = []
        valid_matches_drill = []
        for idx, m in enumerate(matches):
            u0, v0 = kp0[matches[idx].queryIdx].pt
            d0 = 1.0 / disps[0, np.rint(v0).astype(int), np.rint(u0).astype(int)]
            u1, v1 = kp1[matches[idx].trainIdx].pt
            d1 = 1.0 / disps[1, np.rint(v1).astype(int), np.rint(u1).astype(int)]

            if self.check_validity(u0, v0, u1, v1, segm, disp_mask, 0):
                pts_0_p.append(np.array([u0, v0, d0]))
                pts_1_p.append(np.array([u1, v1, d1]))
                valid_matches_patient.append(m)
            elif self.check_validity(u0, v0, u1, v1, segm, disp_mask, 1):
                pts_0_d.append(np.array([u0, v0, d0]))
                pts_1_d.append(np.array([u1, v1, d1]))
                valid_matches_drill.append(m)
            else:
                continue

        if len(pts_0_p) != 0:
            pts_0_p = np.stack(pts_0_p, axis=-1)[:, :self.max_corr]
            pts_1_p = np.stack(pts_1_p, axis=-1)[:, :self.max_corr]
            # convert to 3D
            pts_0_p = self.to_3d(pts_0_p, intrinsics[0])
            pts_1_p = self.to_3d(pts_1_p, intrinsics[1])
        else:
            pts_0_p = None
            pts_1_p = None
        if len(pts_0_d) != 0:
            pts_0_d = np.stack(pts_0_d, axis=-1)[:, :self.max_corr]
            pts_1_d = np.stack(pts_1_d, axis=-1)[:, :self.max_corr]
            pts_0_d = self.to_3d(pts_0_d, intrinsics[0])
            pts_1_d = self.to_3d(pts_1_d, intrinsics[1])
        else:
            pts_0_d = None
            pts_1_d = None

        return pts_0_p, pts_1_p, pts_0_d, pts_1_d

    def forward(self, left, pred_disp, pred_semantic_seg, disp_weight, meta, state, outputs, **kwargs):
        B, M, _, H, W = left.shape

        pred_disp = self.normalized_disparity(pred_disp, state['intrinsics'], meta['baseline'])

        left_1, left_2 = torch.unbind(left, dim=1)
        left_1 = denormalize(left_1.permute(0, 2, 3, 1)).squeeze().cpu().numpy()
        left_2 = denormalize(left_2.permute(0, 2, 3, 1)).squeeze().cpu().numpy()
        intrinsics = state['intrinsics'][..., 0, None, None]
        disp_mask = compute_valid_mask(meta, gt_disp=pred_disp * intrinsics * meta['baseline'])

        pts_0_p, pts_1_p, pts_0_d, pts_1_d = self.feature_matching(left_1, left_2, pred_disp.squeeze().cpu().numpy(),
                                                                   state['intrinsics'].squeeze().cpu().numpy(),
                                                                   pred_semantic_seg.squeeze().cpu().numpy(),
                                                                   disp_mask.squeeze().cpu().numpy())

        if pts_0_p is not None and pts_0_p.shape[1] > 3:
            R_p, t_p = self.arun(pts_0_p, pts_1_p)
            R_p = torch.from_numpy(Rotation.from_matrix(R_p).as_quat())
            t_p = torch.from_numpy(t_p).squeeze()
            vec_p = torch.cat([t_p, R_p]).to(left.device)
            pred_pose_cam_patient = SE3.InitFromVec(vec_p)[None]
        else:
            pred_pose_cam_patient = None
        outputs['pred_pose_cam_patient'] = pred_pose_cam_patient

        if pts_0_d is not None and pts_0_d.shape[1] > 3:
            R_d, t_d = self.arun(pts_0_d, pts_1_d)
            R_d = torch.from_numpy(Rotation.from_matrix(R_d).as_quat())
            t_d = torch.from_numpy(t_d).squeeze()
            vec_d = torch.cat([t_d, R_d]).to(left.device)
            pred_pose_cam_drill = SE3.InitFromVec(vec_d)[None]
        else:
            pred_pose_cam_drill = None
        outputs['pred_pose_cam_drill'] = pred_pose_cam_drill

        return outputs


@MODELS.register_module()
class ICPMotion(MotionBase):
    def __init__(self, iters=200):
        super(ICPMotion, self).__init__()
        self.trans_init = np.identity(4)
        self.iters = iters

    @staticmethod
    def iproj(disps, left, intrinsics, segm, img_meta):
        fx, fy, cx, cy = torch.split(intrinsics[..., None], 1, dim=1)

        xy = meshgrid(disps.unsqueeze(1))
        xy = rearrange(xy, 'm xy h w -> m h w xy')
        z = 1.0 / disps * 1000.0  # MxHxW, mm
        x = (xy[..., 0] - cx) / fx * z
        y = (xy[..., 1] - cy) / fy * z
        xyz = torch.stack([x, y, z], dim=-1)

        valid_mask = compute_valid_mask(img_meta, gt_disp=disps * fx * img_meta['baseline'])  # MxHxW

        patient_valid = valid_mask * (segm[:, 0] > 0.75)  # MxHxW
        drill_valid = valid_mask * (segm[:, 1] > 0.75)  # MxHxw

        xyz_p0 = xyz[0, patient_valid.bool()[0]].cpu().numpy()
        xyz_t0 = xyz[0, drill_valid.bool()[0]].cpu().numpy()
        xyz_p1 = xyz[1, patient_valid.bool()[1]].cpu().numpy()
        xyz_t1 = xyz[1, drill_valid.bool()[1]].cpu().numpy()

        left_0, left_1 = torch.unbind(left, dim=0)
        left_0 = denormalize(left_0.permute(1, 2, 0)).squeeze().cpu().numpy()
        left_1 = denormalize(left_1.permute(1, 2, 0)).squeeze().cpu().numpy()
        color_p0 = left_0[patient_valid.bool()[0].cpu()] / 255.
        color_t0 = left_0[drill_valid.bool()[0].cpu()] / 255.
        color_p1 = left_1[patient_valid.bool()[1].cpu()] / 255.
        color_t1 = left_1[drill_valid.bool()[1].cpu()] / 255.

        return [xyz_p0, color_p0], [xyz_t0, color_t0], [xyz_p1, color_p1], [xyz_t1, color_t1]

    @staticmethod
    def icp(xyz_p0, xyz_t0, xyz_p1, xyz_t1, trans_init, iters):
        p0 = o3d.geometry.PointCloud()
        p1 = o3d.geometry.PointCloud()
        t0 = o3d.geometry.PointCloud()
        t1 = o3d.geometry.PointCloud()

        p0.points = o3d.utility.Vector3dVector(xyz_p0[0])
        p0.colors = o3d.utility.Vector3dVector(xyz_p0[1])
        p1.points = o3d.utility.Vector3dVector(xyz_p1[0])
        p1.colors = o3d.utility.Vector3dVector(xyz_p1[1])
        t0.points = o3d.utility.Vector3dVector(xyz_t0[0])
        t0.colors = o3d.utility.Vector3dVector(xyz_t0[1])
        t1.points = o3d.utility.Vector3dVector(xyz_t1[0])
        t1.colors = o3d.utility.Vector3dVector(xyz_t1[1])

        threshold = 0.1
        reg_p = o3d.pipelines.registration.registration_icp(
            p0, p1, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
        reg_d = o3d.pipelines.registration.registration_icp(
            t0, t1, threshold, trans_init,
            o3d.pipelines.registration.TransformationEstimationPointToPoint(),
            o3d.pipelines.registration.ICPConvergenceCriteria(max_iteration=iters))
        reg_p = reg_p.transformation
        reg_d = reg_d.transformation
        return reg_p, reg_d

    @staticmethod
    def color_icp(xyz_p0, xyz_t0, xyz_p1, xyz_t1, trans_init, iters):
        p0 = o3d.geometry.PointCloud()
        p1 = o3d.geometry.PointCloud()
        t0 = o3d.geometry.PointCloud()
        t1 = o3d.geometry.PointCloud()

        p0.points = o3d.utility.Vector3dVector(xyz_p0[0])
        p0.colors = o3d.utility.Vector3dVector(xyz_p0[1])
        p1.points = o3d.utility.Vector3dVector(xyz_p1[0])
        p1.colors = o3d.utility.Vector3dVector(xyz_p1[1])
        t0.points = o3d.utility.Vector3dVector(xyz_t0[0])
        t0.colors = o3d.utility.Vector3dVector(xyz_t0[1])
        t1.points = o3d.utility.Vector3dVector(xyz_t1[0])
        t1.colors = o3d.utility.Vector3dVector(xyz_t1[1])

        voxel_radius = [0.5]
        max_iter = [200]
        current_p = np.identity(4)
        current_d = np.identity(4)
        for scale in range(len(max_iter)):
            iter = max_iter[scale]
            radius = voxel_radius[scale]

            p0 = p0.voxel_down_sample(radius)
            p1 = p1.voxel_down_sample(radius)
            t0 = t0.voxel_down_sample(radius)
            t1 = t1.voxel_down_sample(radius)

            p0.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 10, max_nn=30))
            p1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 10, max_nn=30))
            t0.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 10, max_nn=30))
            t1.estimate_normals(o3d.geometry.KDTreeSearchParamHybrid(radius=radius * 10, max_nn=30))

            reg_p = o3d.pipelines.registration.registration_colored_icp(
                p0, p1, radius, current_p,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                  relative_rmse=1e-6,
                                                                  max_iteration=iter))
            current_p = reg_p.transformation
            reg_d = o3d.pipelines.registration.registration_colored_icp(
                t0, t1, radius, current_d,
                o3d.pipelines.registration.TransformationEstimationForColoredICP(),
                o3d.pipelines.registration.ICPConvergenceCriteria(relative_fitness=1e-6,
                                                                  relative_rmse=1e-6,
                                                                  max_iteration=iter))
            current_d = reg_d.transformation

        return current_p, current_d

    def forward(self, left, pred_disp, pred_semantic_seg, disp_weight, meta, state, outputs, **kwargs):
        B, M, _, H, W = left.shape
        pred_disp = self.normalized_disparity(pred_disp, state['intrinsics'], meta['baseline'])

        xyz_p0, xyz_t0, xyz_p1, xyz_t1 = self.iproj(pred_disp.squeeze(0), left.squeeze(0),
                                                    state['intrinsics'].squeeze(0),
                                                    pred_semantic_seg.squeeze(0), meta)

        reg_p, reg_d = self.icp(xyz_p0, xyz_t0, xyz_p1, xyz_t1, self.trans_init, self.iters)

        reg_p = np.array(reg_p)
        reg_d = np.array(reg_d)
        R_p = torch.from_numpy(Rotation.from_matrix(reg_p[:3, :3]).as_quat())
        R_d = torch.from_numpy(Rotation.from_matrix(reg_d[:3, :3]).as_quat())
        t_p = torch.from_numpy(reg_p[:3, 3]) / 1000.0
        t_d = torch.from_numpy(reg_d[:3, 3]) / 1000.0

        vec_p = torch.cat([t_p, R_p]).to(left.device)
        vec_d = torch.cat([t_d, R_d]).to(left.device)
        Gs = SE3.Identity([B * 2, M]).to(left.device)  # hardcoded for 2 objects
        Gs[:B, 1] = SE3.InitFromVec(vec_p)[None]
        Gs[B:, 1] = SE3.InitFromVec(vec_d)[None]

        # compute flow
        disps = pred_disp
        intrinsics = state['intrinsics']
        H, W = disps.shape[-2:]
        seg = pred_semantic_seg
        seg = seg.transpose(1, 2).view(B * 2, M, H, W)

        ii = torch.arange(M - 1).to(left.device)
        jj = ii + 1

        flow_list = compute_flow([Gs], disps, seg, intrinsics, ii, jj, meta)

        pred_pose_cam_patient = Gs[:B]
        pred_pose_cam_drill = Gs[B:]

        outputs['pred_pose_cam_patient'] = pred_pose_cam_patient
        outputs['pred_pose_cam_drill'] = pred_pose_cam_drill
        outputs['pred_flow'] = flow_list[-1]

        return outputs

    def losses(self, loss, *args):
        loss['loss_motion'] = self.dummy_param - 1.0
        return