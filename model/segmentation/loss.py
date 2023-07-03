import torch
import torch.nn as nn
from mmseg.models import LOSSES


@LOSSES.register_module()
class SegmentationLoss(nn.Module):
    def __init__(self, loss_weight=(1.0)):
        super(SegmentationLoss, self).__init__()
        self.loss_weight = loss_weight
        self.ce = torch.nn.CrossEntropyLoss(weight=torch.tensor([1.0, 1.0, 10.0]), reduction='mean', ignore_index=255)

    def forward(self, gt_segm, pred_segm_raw, loss, meta, invalid_mask, **kwargs):
        """
        gt_segm: BxMxobjxHxW
        """
        # to remove padded region due to reflection
        img_h, img_w = meta["img_shape"]
        gt_segm = gt_segm[..., :img_h, :img_w]
        pred_segm_raw = pred_segm_raw[..., :img_h, :img_w]

        gt_segm = gt_segm > 0.9999
        gt_background = (~gt_segm[:, :, 0]) & (~gt_segm[:, :, 1])
        target = torch.zeros_like(gt_background).long()  # BxMxHxW

        target[gt_background] = 0
        target[gt_segm[:, :, 0]] = 1  # patient
        target[gt_segm[:, :, 1]] = 2  # drill

        if invalid_mask is not None:
            invalid_mask = invalid_mask[..., :img_h, :img_w]
            target[invalid_mask.bool()] = 255  # invalid for real data

        B, M, _, H, W = gt_segm.shape
        target = target.view(B * M, H, W)

        loss["loss_segm"] = self.loss_weight * self.ce(pred_segm_raw, target)
        return
