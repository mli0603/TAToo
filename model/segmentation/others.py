import torch.nn as nn
from mmseg.models.builder import MODELS


@MODELS.register_module()
class GTSegmentation(nn.Module):
    def __init__(self):
        super(GTSegmentation, self).__init__()
        self.loss = None

    def forward(self, left, img_metas, state, outputs, **kwargs):
        BM, _, H, W = left.shape
        gt_semantic_seg = state['gt_semantic_seg']
        gt_semantic_seg = gt_semantic_seg.view(BM, 2, H, W)
        outputs['pred_semantic_seg'] = gt_semantic_seg
        return outputs

    def freeze(self):
        self.freezed = True
