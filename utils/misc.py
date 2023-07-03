import numpy as np
import torch

from .warp import flow_warp

__imagenet_stats = {'mean': [0.485, 0.456, 0.406],
                    'std': [0.229, 0.224, 0.225]}


def compute_valid_mask(meta, gt_disp=None, gt_semantic_seg=None, gt_flow=None, gt_disp_change=None):
    """
    Args:
        meta (List): dataset meta information
        gt_disp (Tensor): NxHxW
        gt_semantic_seg ([type], optional): NxHxW. Defaults to None.
        gt_flow_prev ([type], optional): BxNxHxWx3. Defaults to None.
        gt_disp_change ([type], optional): NxHxW. Defaults to None.

    Returns:
        Tensor: True for valid
    """
    if gt_disp is not None:
        mask = (gt_disp >= meta["disp_range"][0]) & (gt_disp <= meta["disp_range"][1])
    if gt_semantic_seg is not None:
        mask = gt_semantic_seg > 0
    if gt_flow is not None:
        mag = gt_flow.norm(dim=-1)
        mask = (mag >= meta["flow_range"][0]) & (mag <= meta["flow_range"][1])

    mask.detach_()
    if 'img_shape' in meta: # crop borders for robustness
        img_h, img_w = meta['img_shape']
        mask[..., img_h:, :] = False
        mask[..., img_w:] = False

    return mask


def collect_metric(state):
    """store results

    Args:
        state (dict): states storing information

    Returns:
        Tensor: aggregated results
    """
    metric_list = dict()
    for k, v in state.items():
        if "meter" in k:
            metric_list[k.replace('_meter', '')] = torch.tensor([v.avg])
        if "all" in k:
            metric_list[k.replace('_all', '')] = torch.tensor([v])
    return metric_list


def reset_meter(state):
    """reset results in states when new sequence starts

    Args:
        state (dict)): states storing information
    """
    for k, v in state.items():
        if "meter" in k:
            v.reset()
        if "all" in k:
            state[k] = 0.0


def denormalize(inp):
    """
    inp: NxHxWxC
    """

    assert inp.shape[-1] == 3

    if len(inp.shape) == 3:
        mean = torch.tensor(__imagenet_stats['mean'], device=inp.device)[None, None]
        std = torch.tensor(__imagenet_stats['std'], device=inp.device)[None, None]
    elif len(inp.shape) == 4:
        mean = torch.tensor(__imagenet_stats['mean'], device=inp.device)[None, None, None]
        std = torch.tensor(__imagenet_stats['std'], device=inp.device)[None, None, None]

    output = inp * std
    output = output + mean
    output = output * 255
    output = output.byte()

    return output


def quaternion_multiply(q1, q2):
    """
    rotation multiplication as quaternion
    """
    x1, y1, z1, w1 = np.split(q1, 4, axis=-1)
    x2, y2, z2, w2 = np.split(q2, 4, axis=-1)
    w = w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2
    x = w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2
    y = w1 * y2 + y1 * w2 + z1 * x2 - x1 * z2
    z = w1 * z2 + z1 * w2 + x1 * y2 - y1 * x2
    return np.concatenate([x, y, z, w], axis=-1).astype(float)


def binarize_seg_mask(list_colors, segm):
    B, H, W = segm.shape[:3]
    seg_output = np.zeros([B, H, W])
    for color in list_colors:
        seg_output = np.logical_or(seg_output, np.all(segm == np.array(color), axis=-1))

    return seg_output


def repeat_batch(list_of_tensors, B):
    for idx in range(len(list_of_tensors)):
        inp = list_of_tensors[idx]
        if inp.shape[0] != B:
            num_repeat = B // inp.shape[0]
            inp = torch.cat([inp] * num_repeat, dim=0)
            list_of_tensors[idx] = inp

    return list_of_tensors


def merge_seg(seg):
    """
    seg: BxMxobjxHxW
    return: BobjxMxHxW
    """
    seg_patient, seg_drill = torch.chunk(seg, 2, dim=2)
    seg = torch.cat([seg_patient, seg_drill], dim=0).squeeze(2)
    return seg


def merge_flow(seg_ii_binary, coords, valid_mask):
    coords = seg_ii_binary.unsqueeze(-1) * coords  # BobjxNxHxWx3
    coords_patient, coords_drill = torch.chunk(coords, 2, dim=0)
    coords = coords_patient + coords_drill

    valid_mask = seg_ii_binary.unsqueeze(-1) * valid_mask
    valid_mask_patient, valid_mask_drill = torch.chunk(valid_mask.bool(), 2, dim=0)
    valid_mask = valid_mask_patient | valid_mask_drill
    return coords, valid_mask


def quat2trans(quat):
    """
    Quaternion to 4x4 transformation.
    Args: - quat (7, numpy array): x, y, z, rx, ry, rz, rw
    Returns: - rotm: (4x4 numpy array): transformation matrix
    """
    x = quat[3]
    y = quat[4]
    z = quat[5]
    w = quat[6]
    t_x = quat[0]
    t_y = quat[1]
    t_z = quat[2]
    s = w * w + x * x + y * y + z * z
    homo = np.array([[1 - 2 * (y * y + z * z) / s, 2 * (x * y - z * w) / s, 2 * (x * z + y * w) / s, t_x],
                     [2 * (x * y + z * w) / s, 1 - 2 * (x * x + z * z) / s, 2 * (y * z - x * w) / s, t_y],
                     [2 * (x * z - y * w) / s, 2 * (y * z + x * w) / s, 1 - 2 * (x * x + y * y) / s, t_z],
                     [0, 0, 0, 1]])
    return homo


def trans2quat(trans):
    t_x = trans[0, -1]
    t_y = trans[1, -1]
    t_z = trans[2, -1]

    rot = trans[:3, :3]
    trace = rot[0, 0] + rot[1, 1] + rot[2, 2]
    if trace > 0:
        S = np.sqrt(trace + 1.0) * 2
        qw = 0.25 * S
        qx = (rot[2, 1] - rot[1, 2]) / S
        qy = (rot[0, 2] - rot[2, 0]) / S
        qz = (rot[1, 0] - rot[0, 1]) / S
    elif (rot[0, 0] > rot[1, 1]) and (rot[0, 0] > rot[2, 2]):
        S = np.sqrt(1.0 + rot[0, 0] - rot[1, 1] - rot[2, 2]) * 2
        qw = (rot[2, 1] - rot[1, 2]) / S
        qx = 0.25 * S
        qy = (rot[0, 1] + rot[1, 0]) / S
        qz = (rot[0, 2] + rot[2, 0]) / S
    elif rot[1, 1] > rot[2, 2]:
        S = np.sqrt(1.0 + rot[1, 1] - rot[0, 0] - rot[2, 2]) * 2
        qw = (rot[0, 2] - rot[2, 0]) / S
        qx = (rot[0, 1] + rot[1, 0]) / S
        qy = 0.25 * S
        qz = (rot[1, 2] + rot[2, 1]) / S
    else:
        S = np.sqrt(1.0 + rot[2, 2] - rot[0, 0] - rot[1, 1]) * 2
        qw = (rot[1, 0] - rot[0, 1]) / S
        qx = (rot[0, 2] + rot[2, 0]) / S
        qy = (rot[1, 2] + rot[2, 1]) / S
        qz = 0.25 * S
    return np.array([t_x, t_y, t_z, qx, qy, qz, qw])
