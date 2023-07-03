import torch
import torch.nn as nn
import torch.nn.functional as F
from mmseg.models import builder as builder_oss
from mmseg.models.builder import MODELS

from utils import thres_metric, disp_warp, compute_valid_mask
from ...builder import ESTIMATORS


@ESTIMATORS.register_module()
class HITNetMF(nn.Module):
    """Implementation of HITNet
    <https://arxiv.org/abs/1904.07850>.
    """

    def __init__(
            self,
            backbone,
            initialization,
            propagation,
            loss=None,
    ):
        super(HITNetMF, self).__init__()

        self.backbone = builder_oss.build_backbone(backbone)
        self.tile_init = MODELS.build(initialization)
        self.tile_update = MODELS.build(propagation)
        self.freezed = False

        if loss is not None:
            self.loss = builder_oss.build_loss(loss)
        else:
            self.loss = None

        n_parameters = sum(p.numel() for n, p in self.named_parameters())
        print(
            "PARAM STATUS: total number of parameters %.3fM in stereo network"
            % (n_parameters / 1000 ** 2)
        )

    def extract_feat(self, img):
        """Extract features from images."""
        x = self.backbone(img)
        return x

    def losses(self, loss, outputs, state, meta):
        init_cv_pyramid = outputs["init_cv_pyramid"]
        prop_disp_pyramid = outputs["prop_disp_pyramid"]
        dx_pyramid = outputs["dx_pyramid"]
        dy_pyramid = outputs["dy_pyramid"]
        w_pyramid = outputs["w_pyramid"]
        gt_disp = state['gt_disp']
        mask_disp = compute_valid_mask(gt_disp=gt_disp, meta=meta)

        loss["loss_disp"], loss_dict = self.loss(
            init_cv_pyramid,
            prop_disp_pyramid,
            dx_pyramid,
            dy_pyramid,
            w_pyramid,
            gt_disp,
            meta
        )

    def forward(self, left, right, img_meta, state, outputs):
        left_fea_pyramid = self.extract_feat(left)
        right_fea_pyramid = self.extract_feat(right)
        init_cv_pyramid, init_tile_pyramid = self.tile_init(
            left_fea_pyramid, right_fea_pyramid
        )
        outputs = self.tile_update(
            left_fea_pyramid, right_fea_pyramid, init_tile_pyramid
        )

        if self.training and not self.freezed:
            outputs["init_cv_pyramid"] = init_cv_pyramid
            outputs["pred_disp"] = outputs["prop_disp_pyramid"][-1]
        else:
            outputs = dict(pred_disp=outputs)

        left_feat = left_fea_pyramid[1]
        right_feat = right_fea_pyramid[1]
        right_feat_warp, valid = disp_warp(right_feat, outputs['pred_disp'][..., 3::8, 3::8] / 8.0,
                                           padding_mode="zeros")
        confidence = (left_feat - right_feat_warp).abs().sum(1, keepdim=False)
        confidence = torch.exp(-confidence) * valid[:, 0]  # TODO: maybe divide by 12
        outputs['disp_confidence'] = confidence  # BMxHxW

        if len(outputs["pred_disp"].shape) == 3:
            outputs["pred_disp"] = outputs["pred_disp"].unsqueeze(1)

        return outputs

    def freeze(self):
        self.tile_update.eval()
        for param in self.tile_update.parameters():
            param.requires_grad = False

        self.tile_init.eval()
        for param in self.tile_init.parameters():
            param.requires_grad = False

        feature_extractor = (
            self.backbone if self.backbone is not None else self.feature_extractor
        )
        feature_extractor.eval()
        for param in feature_extractor.parameters():
            param.requires_grad = False

        self.loss.eval()
        for param in self.loss.parameters():
            param.requires_grad = False

        self.freezed = True
