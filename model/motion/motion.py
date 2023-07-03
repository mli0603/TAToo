import copy

import torch
import torch.nn as nn
from lietorch import SE3
from mmseg.models import builder as builder_oss
from mmseg.models.builder import MODELS

from utils import merge_seg, target_warp
from utils import projective_ops as pops
from .base import MotionBase
from .flow.clipping import GradientClip
from .flow.corr import CorrBlock
from .flow.extractor import BasicEncoder
from .flow.gru import ConvGRU
from .geom.ba import moba_segm
from .others import compute_flow, merge_flow


class UpdateModule(nn.Module):
    def __init__(self):
        super(UpdateModule, self).__init__()
        cor_planes = 4 * (2 * 3 + 1) ** 2 + 1 + 2  # disparity + segmentation mask
        flow_planes = 2 + 1 + 2  # disparity + segmentation mask

        self.corr_encoder = nn.Sequential(
            nn.Conv2d(cor_planes, 128, 1, padding=0),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True))

        self.flow_encoder = nn.Sequential(
            nn.Conv2d(flow_planes, 128, 7, padding=3),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 64, 3, padding=1),
            nn.ReLU(inplace=True))

        self.weight = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
            GradientClip(),
            nn.Sigmoid())

        self.delta = nn.Sequential(
            nn.Conv2d(128, 128, 3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(128, 3, 3, padding=1),
            GradientClip())

        self.gru = ConvGRU(128, 128 + 128 + 64)

    def forward(self, net, inp, corr, flow):
        """ RaftSLAM update operator """

        batch, num, ch, ht, wd = net.shape
        output_dim = (batch, num, -1, ht, wd)
        net = net.view(batch * num, -1, ht, wd)
        inp = inp.view(batch * num, -1, ht, wd)
        corr = corr.view(batch * num, -1, ht, wd)
        flow = flow.view(batch * num, -1, ht, wd)

        corr = self.corr_encoder(corr)
        flow = self.flow_encoder(flow)
        net = self.gru(net, inp, corr, flow)

        ### update variables ###
        delta = self.delta(net).view(*output_dim)
        weight = self.weight(net).view(*output_dim)

        delta = delta.permute(0, 1, 3, 4, 2).contiguous()
        weight = weight.permute(0, 1, 3, 4, 2).contiguous()

        net = net.view(*output_dim)
        return net, delta, weight


@MODELS.register_module()
class Motion(MotionBase):
    def __init__(self, config=None, loss=None, pretrained=None):
        """motion network

        Args:
            raft3d (dict, optional): config for raft3d. Defaults to None.
            ds_scale (int, optional): low res scale. Defaults to 4.
            iters (int, optional): optimization iterations. Defaults to 16.
            loss (dict, optional): config for losses. Defaults to None.
        """
        super(Motion, self).__init__()

        self.freezed = False

        # feature network, context network, and update block
        self.fnet = BasicEncoder(output_dim=128, norm_fn='instance')
        self.cnet = BasicEncoder(output_dim=256, norm_fn='none')
        self.update = UpdateModule()

        if loss is not None:
            self.loss = builder_oss.build_loss(loss)
        else:
            self.loss = None

        # load pretrained weights
        if pretrained is not None:
            pretrained_weights = torch.load(pretrained)
            from collections import OrderedDict
            new_state_dict = OrderedDict()
            for k, v in pretrained_weights.items():
                k_ = k.replace('module.', '')
                new_state_dict[k_] = v
            self.load_state_dict(new_state_dict, strict=True)
            print("load success")

        self.delta_d_scale = config.get('delta_d_scale', 1.0)
        self.d_scale = config.get('optimize_d_scale', 100.0)
        self.use_disp_weight = config.get('optimize_disp_weight', True)
        self.use_flow_weight = config.get('optimize_flow_weight', True)
        self.steps = config.get('steps', 12)
        self.gn_steps = config.get('gn_steps', 3)
        self.increase_gn_steps = config.get('increase_gn_steps', False)
        self.lm = config.get('lm', 0.0001)
        self.ep = config.get('ep', 0.1)
        self.sample_target_disp = config.get('sample_target_disp', False)

        n_parameters = sum(p.numel() for n, p in self.named_parameters())
        print(
            "PARAM STATUS: total number of parameters %.3fM in motion network"
            % (n_parameters / 1000 ** 2)
        )

    def extract_features(self, images):
        """ run feeature extraction networks """
        fmaps = self.fnet(images)
        net = self.cnet(images)

        net, inp = net.split([128, 128], dim=2)
        net = torch.tanh(net)
        inp = torch.relu(inp)
        return fmaps, net, inp

    def optimize(self, pred_semantic_seg, pred_disp, disp_weight, corr_fn, net, inp, meta, state, outputs, ii, jj,
                 Gs=None):
        '''
        disp_weight: 1/8 resolution
        '''
        
        seg = pred_semantic_seg[..., 3::8, 3::8]  # BxMxobjxHxW
        disps = pred_disp[..., 3::8, 3::8]  # this is normalized, so no need for scaling
        intrinsics = state['intrinsics'] / 8.0
        mask = outputs['pred_disp_mask'].float()[..., 3::8, 3::8]
        meta_lr = copy.deepcopy(meta)
        meta_lr['disp_range'] = (meta['disp_range'][0] / 8.0, meta['disp_range'][1] / 8.0)
        # eliminates invalid depth during optimization by masking out disp_weight (later warped)
        disp_weight = mask * disp_weight

        B, M = disps.shape[:2]
        N = ii.shape[0]  # N: number of edges
        H, W = disps.shape[-2:]

        if Gs is None:
            Gs = SE3.Identity([B * 2, M], device=disps.device)  # BobjxM

        # move seg objects to batch dim
        seg_batched = merge_seg(seg)
        seg_ii_binary = seg_batched[:, ii] > 0.75  # BobjxNxHxW, 0.75 for confident estimates only

        coords, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj, meta_lr)
        coords, valid_mask = merge_flow(seg_ii_binary, coords, valid_mask)
        residual = torch.zeros_like(coords[..., :2])

        Gs_list, target_list, weight_list = [], [], []
        gn_steps = self.gn_steps
        for iter in range(self.steps):
            Gs = Gs.detach()
            coords = coords.detach()
            residual = residual.detach()

            corr = corr_fn(coords[..., :2])
            flow = residual.permute(0, 1, 4, 2, 3).clamp(-32.0, 32.0)

            corr = torch.cat([corr, disps[:, ii, None], seg[:, ii]], dim=2)
            flow = torch.cat([flow, disps[:, ii, None], seg[:, ii]], dim=2)
            net, delta, weight = self.update(net, inp, corr, flow)

            deltaxy, deltaz = delta[..., :2], delta[..., [-1]]
            # delta is in pixel space, convert to normalized space
            deltaz = deltaz / (meta['baseline'] * intrinsics[:, ii, 0, None, None, None]) * self.delta_d_scale
            delta_scaled = torch.cat([deltaxy, deltaz], dim=-1)

            target = coords + delta_scaled

            if self.sample_target_disp:
                target = target[..., :2]

                # add disparity dimension # TODO refactor BA sample so we have warp at one place
                target_for_warp = target.view(B * N, H, W, 2)  # BNxHxWx2
                target_for_warp = target_for_warp.permute(0, 3, 1, 2)  # BNx2xHxW
                target_disp = disps[:, jj].view(B * N, 1, H, W)  # BNx1xHxW
                target_disp, _ = target_warp(target_disp, target_for_warp, padding_mode="zeros",
                                             mode="nearest")  # BNx1xHxW
                target_disp = target_disp.view(B, N, H, W, 1)  # BxNxHxWx1
                target = torch.cat([target, target_disp], dim=-1)

            # optimization configs
            if not self.use_disp_weight:
                disp_weight = torch.ones_like(disp_weight)
            if not self.use_flow_weight:
                weight = torch.ones_like(weight)

            for i in range(gn_steps):
                Gs = moba_segm(target, seg_batched, disp_weight, weight, Gs, disps, intrinsics, ii, jj, meta_lr,
                               d_scale=self.d_scale, lm=self.lm, ep=self.ep)
            if self.increase_gn_steps:
                gn_steps += 1

            coords, valid_mask = pops.projective_transform(Gs, disps, intrinsics, ii, jj, meta_lr)
            coords, valid_mask = merge_flow(seg_ii_binary, coords, valid_mask)
            residual = (target - coords)[..., :2]
            Gs_list.append(Gs)

            valid_mask = valid_mask * mask[:, ii].unsqueeze(-1)
            target_list.append([target, valid_mask.bool()])
            weight_list.append(weight)

        return Gs_list, target_list, weight_list

    def forward(self, left, pred_disp, pred_semantic_seg, disp_weight, meta, state, outputs, Gs=None, **kwargs):
        """ Estimates SE3 or Sim3 between pair of frames """
        B, M, _, H, W = left.shape
        pred_disp = self.normalized_disparity(pred_disp, state['intrinsics'], meta['baseline'])

        ii = torch.arange(M - 1).to(left.device)
        jj = ii + 1

        fmaps, net, inp = self.extract_features(left)
        net, inp = net[:, ii], inp[:, ii]
        corr_fn = CorrBlock(fmaps[:, ii], fmaps[:, jj], num_levels=4, radius=3)

        Gs_list, target_list, match_weight = self.optimize(pred_semantic_seg, pred_disp, disp_weight, corr_fn, net, inp,
                                                           meta, state, outputs, ii, jj)

        # for computing flow at full res
        disps = pred_disp  # this is normalized, so no need for scaling
        seg = pred_semantic_seg
        seg_batched = merge_seg(seg)
        intrinsics = state['intrinsics']
        if self.training and not self.freezed:
            outputs['pred_poses'] = Gs_list
            outputs['pred_flow'] = compute_flow(Gs_list, disps, seg_batched, intrinsics, ii, jj, meta)
            outputs['pred_target'] = target_list
            outputs['pred_match_weight'] = match_weight
        else:
            outputs['pred_pose_cam_patient'] = Gs_list[-1][:B]
            outputs['pred_pose_cam_drill'] = Gs_list[-1][B:]
            outputs['pred_flow'] = compute_flow([Gs_list[-1]], disps, seg_batched, intrinsics, ii, jj, meta)[-1]
            outputs['pred_target'] = target_list
            outputs['pred_match_weight'] = match_weight

        return outputs

    def losses(self, loss, output, state, meta, **kwargs):
        Gs_list, flow_list = output['pred_poses'], output['pred_flow']
        target_list = output.get('pred_target', None)
        disps = output['pred_disp']
        seg = output['pred_semantic_seg']
        match_weight = output['pred_match_weight']
        self.loss(Gs_list, target_list, flow_list, disps, seg, state, loss, meta, match_weight=match_weight)

    def freeze(self):
        self.eval()
        self.loss.eval()
        for param in self.parameters():
            param.requires_grad = False

        self.freezed = True
