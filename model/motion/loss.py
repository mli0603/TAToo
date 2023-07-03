import copy

import lietorch
import torch
import torch.nn as nn
from mmseg.models import LOSSES

from utils import compute_valid_mask, merge_seg
from .others import compute_flow


@LOSSES.register_module()
class MotionLoss(nn.Module):
    def __init__(self, loss_weight=(1.0), geo_weight=10.0, res_weight=0.1, flo_weight=0.05, w_tau=1.0, w_phi=1.0,
                 w_p=1.0, w_d=1.0, gamma=0.8, supervise_match_weight=False):
        super(MotionLoss, self).__init__()
        self.loss_weight = loss_weight
        self.w1 = geo_weight
        self.w2 = res_weight
        self.w3 = flo_weight
        self.w_tau = w_tau
        self.w_phi = w_phi
        self.w_p = w_p
        self.w_d = w_d
        self.gamma = gamma
        self.supervise_match_weight = supervise_match_weight

    def geodesic_loss(self, Gs_list, ii, jj, gt_poses, **kwargs):
        iters = len(Gs_list)
        gt_poses = (gt_poses[:, jj] * gt_poses[:, ii].inv())
        B = gt_poses.shape[0] // 2
        total_loss = 0.0

        for i, Gs in enumerate(Gs_list):
            w = self.gamma ** (iters - 1 - i)
            dG = (Gs[:, jj] * Gs[:, ii].inv())
            e_poses = (gt_poses.inv() * dG).log()
            tau, phi = e_poses.split([3, 3], -1)

            e_p = tau.norm(dim=-1)[:B].mean() * self.w_tau + phi.norm(dim=-1)[:B].mean() * self.w_phi
            e_d = tau.norm(dim=-1)[B:].mean() * self.w_tau + phi.norm(dim=-1)[B:].mean() * self.w_phi

            total_loss += w * e_p * self.w_p
            total_loss += w * e_d * self.w_d
        return total_loss  # , tau_loss, phi_loss

    def target_loss(self, target_list, gt_target, valid_mask, seg, ii, invalid_mask, match_weight_list):
        """
        supervision on target (x,y only) before geometric optimization (use GT seg, GT depth)
        ensures proper learning of correspondence
        """
        iters = len(target_list)
        B = seg.shape[0] // 2

        total_loss = 0.0
        if invalid_mask is not None:
            valid_mask = valid_mask & (1 - invalid_mask).bool().unsqueeze(-1)
        for i, (target, target_valid) in enumerate(target_list):
            w = self.gamma ** (iters - 1 - i)
            valid = (valid_mask & target_valid).squeeze(-1)
            match_weight = match_weight_list[i]

            if self.supervise_match_weight:
                residual = (target - gt_target).abs() * match_weight
            else:
                residual = (target - gt_target).abs()[..., :2]
            residual_p = residual[valid & seg[:B, ii].bool()].mean()
            residual_d = residual[valid & seg[B:, ii].bool()].mean()

            total_loss += w * (residual_p * self.w_p + residual_d * self.w_d)
        return total_loss

    def flow_loss(self, flow_list, gt_flow, meta, seg, ii, scale, invalid_mask):
        """
        supervision on pose induced flow after geometric optimization (use pred seg, pred depth)
        ensures proper learning of confidence weight
        """
        iters = len(flow_list)
        B = seg.shape[0] // 2
        total_loss = 0.0
        mask_flow = compute_valid_mask(gt_flow=gt_flow, meta=meta)
        if invalid_mask is not None:
            mask_flow = mask_flow & (1 - invalid_mask).bool()
        for i, pred_flow in enumerate(flow_list):
            w = self.gamma ** (iters - 1 - i)
            epe = (pred_flow - gt_flow).abs() * scale
            epe_p = epe[mask_flow & seg[:B, ii].bool()].mean()
            epe_d = epe[mask_flow & seg[B:, ii].bool()].mean()

            total_loss += w * (epe_p * self.w_p + epe_d * self.w_d)
        return total_loss

    def forward(self, Gs_list, target_list, flow_list, disps, seg, state, loss, meta, **kwargs):
        """
        disps: in pixel coordinate (loss is called after forward pass), so need conversions
        """
        B, M = state['gt_pose_cam_patient'].shape[:2]
        device = state['gt_pose_cam_patient'].device
        ii = torch.arange(M - 1).to(device)
        jj = ii + 1
        scale = torch.ones([B, 1, 1, 1, 3], device=device)  # BxNxHxWx3
        scale[..., -1] = meta['baseline'] * state['intrinsics'][:, ii, 0, None, None]  # convert to pixel
        img_h, img_w = meta['img_shape']
        match_weight = kwargs.get('match_weight', None)

        Gs_gt = lietorch.cat([state['gt_pose_cam_patient'], state['gt_pose_cam_drill']], dim=0)

        geodesic_loss = self.geodesic_loss(Gs_list, ii, jj, Gs_gt, **state)

        # induced scene flow loss, use pred seg/disp
        if flow_list is not None:
            # compute gt
            disps_norm = disps / meta['baseline'] / state['intrinsics'][..., 0, None, None]
            invalid_mask = state.get('invalid_mask', None)
            if invalid_mask is not None:
                invalid_mask = invalid_mask[:, ii]
            seg = merge_seg(seg)
            # NOTE: this needs to be of shape BxNxHxWx3, and last dimension is not scaled up
            gt_flow = compute_flow([Gs_gt], disps_norm, seg, state['intrinsics'], ii, jj, meta)[-1]

            # compute loss
            flow_loss = self.flow_loss(flow_list, gt_flow, meta, seg, ii, scale, invalid_mask)
            loss["flow"] = flow_loss
        else:
            flow_loss = torch.tensor([0.0], device=device)
            loss["flow"] = flow_loss

        # predicted optical flow loss, use gt seg/disp
        if target_list is not None:
            # compute gt (use gt seg/disp)
            # residual - at 1/8 resolution
            disps_norm = state['gt_disp'] / meta['baseline'] / state['intrinsics'][..., 0, None, None]
            disps_norm_lr = disps_norm[..., 3::8, 3::8]
            intrinsics_lr = state['intrinsics'] / 8.0
            seg = merge_seg(state['gt_semantic_seg'])
            seg_lr = seg[..., 3::8, 3::8]
            invalid_mask_lr = state.get('invalid_mask', None)
            if invalid_mask_lr is not None:
                invalid_mask_lr = invalid_mask_lr[:, ii]
                invalid_mask_lr = invalid_mask_lr[..., 3::8, 3::8]
            meta_lr = copy.deepcopy(meta)
            meta_lr['disp_range'] = (meta['disp_range'][0] / 8.0, meta['disp_range'][1] / 8.0)
            # NOTE: this needs to be of shape BxNxHxWx3, and last dimension is not scaled up
            gt_target, gt_target_valid_mask = \
                compute_flow([Gs_gt], disps_norm_lr, seg_lr, intrinsics_lr, ii, jj, meta_lr, True)[-1]
            gt_target_valid_mask[..., img_h // 8:, :, :] = False
            gt_target_valid_mask[..., img_w // 8:, :] = False

            # compute loss
            target_loss = self.target_loss(target_list, gt_target, gt_target_valid_mask, seg_lr, ii, invalid_mask_lr,
                                           match_weight)
            loss["target"] = target_loss
        else:
            target_loss = torch.tensor([0.0], device=device)
            loss["target"] = target_loss

        loss_total = self.w1 * geodesic_loss + self.w2 * target_loss + self.w3 * flow_loss

        loss["loss_motion"] = loss_total * self.loss_weight
        loss["geodesic"] = geodesic_loss

        return
