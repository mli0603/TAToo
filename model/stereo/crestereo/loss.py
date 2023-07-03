import torch
import torch.nn as nn
from mmseg.models import LOSSES


def sequence_loss(flow_preds, flow_gt, valid, loss_gamma=0.9, negate=True):
    """ Loss function defined over sequence of flow predictions """

    n_predictions = len(flow_preds)
    assert n_predictions >= 1
    flow_loss = 0.0

    # exclude extremly large displacements
    assert valid.shape == flow_gt.shape, [valid.shape, flow_gt.shape]
    assert not torch.isinf(flow_gt[valid.bool()]).any()

    for i in range(n_predictions):
        assert not torch.isnan(flow_preds[i]).any(
        ) and not torch.isinf(flow_preds[i]).any()
        # We adjust the loss_gamma so it is consistent for any number of RAFT-Stereo iterations
        adjusted_loss_gamma = loss_gamma**(15/(n_predictions - 1))
        i_weight = adjusted_loss_gamma**(n_predictions - i - 1)
        if negate: # flow is negative, disparity is positive
            i_loss = (-1.0 * flow_preds[i] - flow_gt).abs()
        else:
            i_loss = (flow_preds[i] - flow_gt).abs()
        assert i_loss.shape == valid.shape, [
            i_loss.shape, valid.shape, flow_gt.shape, flow_preds[i].shape]
        flow_loss += i_weight * i_loss[valid.bool()].mean()

    return flow_loss


@LOSSES.register_module()
class FlowLoss(nn.Module):

    def __init__(self, gamma=0.9, loss_weight=1.0, negate=True):
        super(FlowLoss, self).__init__()
        self.gamma = gamma
        self.loss_weight = loss_weight
        self.negate = negate

    def forward(self, flow_preds, flow_gt, valid):
        return sequence_loss(flow_preds, flow_gt, valid, loss_gamma=self.gamma, negate=self.negate) * self.loss_weight
