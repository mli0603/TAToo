import torch
import torch.nn as nn
import torch.nn.functional as F

from ...builder import ESTIMATORS
from mmseg.models.builder import MODELS
from mmseg.models import builder as builder_oss
from utils import thres_metric, disp_warp, compute_valid_mask


@ESTIMATORS.register_module()
class CREStereoMF(nn.Module):
    def __init__(self, crestereo, loss=None):
        super(CREStereoMF, self).__init__()

        self.net = MODELS.build(crestereo)
        if loss is not None:
            self.loss = builder_oss.build_loss(loss)
        else:
            self.loss = None

        n_parameters = sum(p.numel() for n, p in self.named_parameters())
        print(
            "PARAM STATUS: total number of parameters %.3fM in stereo network"
            % (n_parameters / 1000 ** 2)
        )

    def losses(self, loss, outputs, state, meta):
        flow_preds = outputs['flow_preds']
        flow_gt = state['gt_disp']
        valid = compute_valid_mask(gt_disp=flow_gt, meta=meta)
        loss["loss_disp"] = self.loss(flow_preds, flow_gt, valid)

    def forward(self, left, right, img_meta, state, outputs):
        # To comply with crestereo's data aug
        left = (left - left.min()) / (left.max() - left.min())
        right = (right - right.min()) / (right.max() - right.min())
        flow_predictions, flow_8, corr = self.net(
            left, right, test_mode=not self.training)

        confidence = torch.sigmoid(corr.mean(dim=1, keepdim=True))
        confidence = F.interpolate(confidence, scale_factor=0.5) # keep it at 1/8 resolution
        outputs['disp_confidence'] = confidence  # BMxHxW
        outputs['flow_preds'] = flow_predictions
        outputs['pred_disp'] = flow_predictions[-1][:, [0]]

        return outputs

    def freeze(self):
        self.net.eval()
        for param in self.net.parameters():
            param.requires_grad = False

        self.freezed = True
