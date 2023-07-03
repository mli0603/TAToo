import numpy as np
import torch
from lietorch import SO3
from einops import rearrange

EPSILON = 1e-8


def epe_metric(d_est, d_gt, mask, use_np=False):
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        epe = np.mean(np.abs(d_est - d_gt))
    else:
        epe = torch.mean(torch.abs(d_est - d_gt))

    return epe


def thres_metric(d_est, d_gt, mask, thres, use_np=False):
    assert isinstance(thres, (int, float))
    d_est, d_gt = d_est[mask], d_gt[mask]
    if use_np:
        e = np.abs(d_gt - d_est)
    else:
        e = torch.abs(d_gt - d_est)
    err_mask = e > thres

    if use_np:
        mean = np.mean(err_mask.astype("float"))
    else:
        mean = torch.mean(err_mask.float())

    return mean


def pose_acc(gt_poses, pred_poses, eval_motion=False, obj=''):
    B, M = gt_poses.shape[:2]
    ii = torch.arange(M - 1).to(gt_poses.device)
    jj = ii + 1

    d_gt = (gt_poses[:, jj] * gt_poses[:, ii].inv())
    gt_trans, gt_orien = d_gt.data.split([3, 4], -1)
    gt_orien = SO3(gt_orien).log() / np.pi * 180

    if pred_poses is not None:  # tracking success
        d_pred = (pred_poses[:, jj] * pred_poses[:, ii].inv())
        if eval_motion:
            e_poses = d_gt.inv() * d_pred
        else:  # eval target pose
            pred_jj = d_pred * gt_poses[:, ii]
            e_poses = gt_poses[:, jj].inv() * pred_jj

        e_trans, e_orien = e_poses.data.split([3, 4], -1)
        e_orien = SO3(e_orien).log() / np.pi * 180

        e_trans_norm = e_trans.norm(dim=-1)
        e_orien_norm = e_orien.norm(dim=-1)

        pred_trans, pred_orien = d_pred.data.split([3, 4], -1)
        pred_orien = SO3(pred_orien).log() / np.pi * 180

        results = {
            'tau_{}'.format(obj): e_trans_norm,
            'tau_{}_x_pred'.format(obj): pred_trans[..., 0],
            'tau_{}_y_pred'.format(obj): pred_trans[..., 1],
            'tau_{}_z_pred'.format(obj): pred_trans[..., 2],
            'tau_{}_x_gt'.format(obj): gt_trans[..., 0],
            'tau_{}_y_gt'.format(obj): gt_trans[..., 1],
            'tau_{}_z_gt'.format(obj): gt_trans[..., 2],
            'phi_{}'.format(obj): e_orien_norm,
            'phi_{}_x_pred'.format(obj): pred_orien[..., 0],
            'phi_{}_y_pred'.format(obj): pred_orien[..., 1],
            'phi_{}_z_pred'.format(obj): pred_orien[..., 2],
            'phi_{}_x_gt'.format(obj): gt_orien[..., 0],
            'phi_{}_y_gt'.format(obj): gt_orien[..., 1],
            'phi_{}_z_gt'.format(obj): gt_orien[..., 2],
        }
    else:
        nan = torch.tensor([np.NAN]).to(gt_poses.device)
        results = {
            'tau_{}'.format(obj): nan,
            'tau_{}_x_pred'.format(obj): nan,
            'tau_{}_y_pred'.format(obj): nan,
            'tau_{}_z_pred'.format(obj): nan,
            'tau_{}_x_gt'.format(obj): gt_trans[..., 0],
            'tau_{}_y_gt'.format(obj): gt_trans[..., 1],
            'tau_{}_z_gt'.format(obj): gt_trans[..., 2],
            'phi_{}'.format(obj): nan,
            'phi_{}_x_pred'.format(obj): nan,
            'phi_{}_y_pred'.format(obj): nan,
            'phi_{}_z_pred'.format(obj): nan,
            'phi_{}_x_gt'.format(obj): gt_orien[..., 0],
            'phi_{}_y_gt'.format(obj): gt_orien[..., 1],
            'phi_{}_z_gt'.format(obj): gt_orien[..., 2],
        }
    return results


def pose_acc_rel(gt_poses_d, gt_poses_p, pred_poses_d, pred_poses_p, eval_motion=False):
    B, M = gt_poses_d.shape[:2]
    ii = torch.arange(M - 1).to(gt_poses_d.device)
    jj = ii + 1

    pose_p_d_gt = gt_poses_p[:, jj].inv() * gt_poses_d[:, jj] # drill in patient coord
    gt_trans, gt_orien = pose_p_d_gt.data.split([3, 4], -1)
    gt_orien = SO3(gt_orien).log() / np.pi * 180

    if pred_poses_d is not None:  # tracking success
        d_pred_p = (pred_poses_p[:, jj] * pred_poses_p[:, ii].inv())
        pred_jj_p = d_pred_p * gt_poses_p[:, ii]
        d_pred_d = (pred_poses_d[:, jj] * pred_poses_d[:, ii].inv())
        pred_jj_d = d_pred_d * gt_poses_d[:, ii]
        pose_p_d_pred = pred_jj_p.inv() * pred_jj_d
        e_poses = pose_p_d_gt.inv() * pose_p_d_pred

        e_trans, e_orien = e_poses.data.split([3, 4], -1)
        e_orien = SO3(e_orien).log() / np.pi * 180

        e_trans_norm = e_trans.norm(dim=-1)
        e_orien_norm = e_orien.norm(dim=-1)

        pred_trans, pred_orien = pose_p_d_pred.data.split([3, 4], -1)
        pred_orien = SO3(pred_orien).log() / np.pi * 180
        obj = 'rel'
        results = {
            'tau_{}'.format(obj): e_trans_norm,
            'tau_{}_x_pred'.format(obj): pred_trans[..., 0],
            'tau_{}_y_pred'.format(obj): pred_trans[..., 1],
            'tau_{}_z_pred'.format(obj): pred_trans[..., 2],
            'tau_{}_x_gt'.format(obj): gt_trans[..., 0],
            'tau_{}_y_gt'.format(obj): gt_trans[..., 1],
            'tau_{}_z_gt'.format(obj): gt_trans[..., 2],
            'phi_{}'.format(obj): e_orien_norm,
            'phi_{}_x_pred'.format(obj): pred_orien[..., 0],
            'phi_{}_y_pred'.format(obj): pred_orien[..., 1],
            'phi_{}_z_pred'.format(obj): pred_orien[..., 2],
            'phi_{}_x_gt'.format(obj): gt_orien[..., 0],
            'phi_{}_y_gt'.format(obj): gt_orien[..., 1],
            'phi_{}_z_gt'.format(obj): gt_orien[..., 2],
        }
    else:
        obj = 'rel'
        nan = torch.tensor([np.NAN]).to(pose_p_d_gt.device)
        results = {
            'tau_{}'.format(obj): nan,
            'tau_{}_x_pred'.format(obj): nan,
            'tau_{}_y_pred'.format(obj): nan,
            'tau_{}_z_pred'.format(obj): nan,
            'tau_{}_x_gt'.format(obj): gt_trans[..., 0],
            'tau_{}_y_gt'.format(obj): gt_trans[..., 1],
            'tau_{}_z_gt'.format(obj): gt_trans[..., 2],
            'phi_{}'.format(obj): nan,
            'phi_{}_x_pred'.format(obj): nan,
            'phi_{}_y_pred'.format(obj): nan,
            'phi_{}_z_pred'.format(obj): nan,
            'phi_{}_x_gt'.format(obj): gt_orien[..., 0],
            'phi_{}_y_gt'.format(obj): gt_orien[..., 1],
            'phi_{}_z_gt'.format(obj): gt_orien[..., 2],
        }
    return results
        

def segmentation_metrics(pred_label, label, num_objects=2):
    """
    pred_label, label: BxMxobjxHxW
    """
    pred_label = pred_label > (1.0 / (num_objects + 1))  # binarize label
    pred_label = rearrange(pred_label, 'b m obj h w -> obj (b m h w)')
    label = rearrange(label, 'b m obj h w -> obj (b m h w)')

    intersect = pred_label == label
    intersect = torch.logical_and(intersect, pred_label)
    area_intersect = torch.sum(intersect.float(), dim=-1)  # [0, 1, 3, 4])
    area_pred_label = torch.sum(pred_label.float(), dim=-1)  # [0, 1, 3, 4])
    area_label = torch.sum(label.float(), dim=-1)  # [0, 1, 3, 4])
    area_union = area_pred_label + area_label - area_intersect

    iou = area_intersect / area_union
    acc = area_intersect / area_label
    dice = 2 * area_intersect / (area_pred_label + area_label)
    return iou, acc, dice
