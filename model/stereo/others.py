import torch.nn as nn
from mmseg.models.builder import MODELS


@MODELS.register_module()
class GTStereo(nn.Module):
    def __init__(self):
        super(GTStereo, self).__init__()
        self.loss = None

    def forward(self, left, right, img_metas, state, outputs, **kwargs):
        BM, _, H, W = left.shape
        gt_disp = state['gt_disp']
        gt_disp = gt_disp.view(BM, 1, H, W)
        outputs['pred_disp'] = gt_disp

        outputs['disp_confidence'] = (gt_disp > 5)[..., 3::8, 3::8].float()  # 1/8 resolution
        return outputs

    def freeze(self):
        self.freezed = True
