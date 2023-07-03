import lietorch
import open3d as o3d
import torch
import torch.nn as nn
from mmseg.models.builder import MODELS

import utils.projective_ops as pops
from utils import flow_warp, compute_valid_mask


@MODELS.register_module()
class Renderer(nn.Module):
    def __init__(self, vis_3d=False):
        super(Renderer, self).__init__()
        self.loss = None
        self.vis_3d = vis_3d

    def forward(self, left, disps, segm, img_meta, state, outputs, **kwargs):
        intrinsics = state['intrinsics']

        B, M, _, H, W = left.shape
        left_0, left_1 = torch.unbind(left, dim=1)  # TODO: this is hard-coded for two frames

        flow = outputs['pred_flow'].squeeze(1)  # NxHxWx3
        flow = flow[..., :2].permute(0, 3, 1, 2)  # Nx2xHxW

        left_0_synthesized, valid = flow_warp(left_1, flow, padding_mode="zeros")

        import matplotlib.pyplot as plt
        plt.subplot(131)
        plt.imshow(left_0.squeeze()[0].cpu())
        plt.title('img 0')
        plt.subplot(132)
        plt.imshow(left_1.squeeze()[0].cpu())
        plt.title('img 1')
        plt.subplot(133)
        plt.imshow(left_0_synthesized.squeeze()[0].cpu())
        plt.title('predicted img 0')
        plt.show()

        outputs['pred_img'] = left_0_synthesized

        # 3D
        if self.vis_3d:
            fx = intrinsics[..., 0][..., None, None]

            pose_cam_patient = outputs['pred_pose_cam_patient']
            pose_cam_drill = outputs['pred_pose_cam_drill']
            poses = lietorch.cat([pose_cam_patient, pose_cam_drill], dim=0)

            ii = torch.arange(M - 1).to(disps.device)
            jj = ii + 1

            # inverse project (pinhole)
            X0 = pops.iproj(disps[:, ii], intrinsics[:, ii])
            X0_expand = X0.expand(poses.shape[0], -1, -1, -1, -1)  # expand batch dim to match poses, BobjxNxHxWx3

            # transform
            Gij = poses[:, jj] * poses[:, ii].inv()
            X1_est, _ = pops.actp(Gij, X0_expand)  # BobjxNxHxWx3
            seg_ii = segm[:, ii]  # BxNxobjxHxWx3
            X1_est = X1_est[:B] * seg_ii[:, :, 0].unsqueeze(-1) + X1_est[B:] * seg_ii[:, :, 1].unsqueeze(
                -1)  # BxNxHxWx3

            X1 = pops.iproj(disps[:, jj], intrinsics[:, jj])
            X0 = X0[..., :3] / X0[..., 3].unsqueeze(-1)
            X1_est = X1_est[..., :3] / X1_est[..., 3].unsqueeze(-1)
            X1 = X1[..., :3] / X1[..., 3].unsqueeze(-1)

            # exclude points out of range
            valid_0 = compute_valid_mask(img_meta, gt_disp=disps[:, ii] * fx[:, ii] * img_meta['baseline']).bool()
            valid_1 = compute_valid_mask(img_meta, gt_disp=disps[:, jj] * fx[:, jj] * img_meta['baseline']).bool()
            
            X0 = X0[valid_0, :].cpu().numpy()
            X1_est = X1_est[valid_0, :].cpu().numpy()
            X1 = X1[valid_1, :].cpu().numpy()

            pcd_0 = o3d.geometry.PointCloud()
            pcd_0.points = o3d.utility.Vector3dVector(X0)
            pcd_0.paint_uniform_color([1, 0.0, 0])
            pcd_1_est = o3d.geometry.PointCloud()
            pcd_1_est.points = o3d.utility.Vector3dVector(X1_est)
            pcd_1_est.paint_uniform_color([0, 1.0, 0])
            pcd_1 = o3d.geometry.PointCloud()
            pcd_1.points = o3d.utility.Vector3dVector(X1)
            pcd_1.paint_uniform_color([0, 0, 1])

            o3d.visualization.draw([pcd_0, pcd_1, pcd_1_est])

        return outputs

    def losses(self, loss, *args):
        loss['loss_motion'] = self.dummy_param - 1.0
        return
