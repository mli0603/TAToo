import torch.nn as nn
from mmseg.models.builder import MODELS


@MODELS.register_module()
class MotionBase(nn.Module):
    def __init__(self, **kwargs):
        super(MotionBase, self).__init__()

    @staticmethod
    def normalized_disparity(disp, intrinsics, baseline):
        fx = intrinsics[..., 0, None, None]
        disp = disp / baseline / fx
        return disp
