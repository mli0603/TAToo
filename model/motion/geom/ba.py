import lietorch
import torch

from utils import projective_ops as pops
from utils import target_warp, repeat_batch
from .chol import block_solve


# utility functions for scattering ops
def safe_scatter_add_mat(H, data, ii, jj, B, M, D):
    v = (ii >= 0) & (jj >= 0)
    H.scatter_add_(1, (ii[v] * M + jj[v]).view(1, -1, 1, 1).repeat(B, 1, D, D), data[:, v])


def safe_scatter_add_vec(b, data, ii, B, M, D):
    v = ii >= 0
    b.scatter_add_(1, ii[v].view(1, -1, 1).repeat(B, 1, D), data[:, v])


def moba_segm(target, segm_weight, disp_weight, flow_weight, poses, disps, intrinsics, ii, jj, meta, fixedp=1,
              lm=0.0001, ep=0.1, d_scale=100.0):
    """ MoBA: Motion Only Bundle Adjustment """

    B, M = poses.shape[:2]  # B: batch size * obj, M: number of poses
    D = poses.manifold_dim
    N = ii.shape[0]  # N: number of edges
    H, W = disps.shape[-2:]

    target, disp_weight, flow_weight = repeat_batch([target, disp_weight, flow_weight], B)

    # find current residual
    coords, valid, (Ji, Jj) = pops.projective_transform(poses, disps, intrinsics, ii, jj, meta, jacobian=True)
    r = target - coords

    # import matplotlib.pyplot as plt
    # plt.subplot(131)
    # plt.imshow(r[0, 0, ..., 0].cpu().abs(), vmax=5)
    # plt.subplot(132)
    # plt.imshow(r[0, 0, ..., 1].cpu().abs(), vmax=5)
    # plt.subplot(133)
    # plt.imshow(r[0, 0, ..., 2].cpu().abs(), vmax=5)
    # plt.show()

    r = r.view(B, N, -1, 1)

    # build weight for least squares
    ## disp
    target_for_warp = target.view(B * N, H, W, 3)[..., :2]  # BNxHxWx2
    target_for_warp = target_for_warp.permute(0, 3, 1, 2)  # BNx2xHxW
    disp_weight_ii = disp_weight[:, ii].unsqueeze(-1)  # BxNxHxWx1
    disp_weight_jj = disp_weight[:, jj].view(B * N, 1, H, W)  # BNx1xHxW
    disp_weight_jj, _ = target_warp(disp_weight_jj, target_for_warp, padding_mode="zeros", mode='nearest')  # BNx1xHxW
    disp_weight_jj = disp_weight_jj.view(B, N, H, W, 1)  # BxNxHxWx1
    weight = disp_weight_ii * disp_weight_jj  # BxNxHxWx1

    # import matplotlib.pyplot as plt
    # plt.imshow(weight[0, 0, ..., 0].cpu())
    # plt.show()

    ## Seg
    seg_ii = segm_weight[:, ii].unsqueeze(-1)  # BxNxHxWx1
    seg_jj = segm_weight[:, jj, None].view(B * N, 1, H, W)  # BNx1xHxW
    seg_jj, _ = target_warp(seg_jj, target_for_warp, padding_mode="zeros", mode='nearest')
    seg_jj = seg_jj.view(B, N, 1, H, W).permute(0, 1, 3, 4, 2)  # BxNxHxWx1
    weight = weight * seg_ii * seg_jj  # BxNxHxWx1

    # import matplotlib.pyplot as plt
    # plt.imshow(seg_jj[0, 0, ..., 0].cpu())
    # plt.show()

    ## flow
    weight = weight * flow_weight  # BxNxHxWx3

    ## scale disp up if used
    weight[..., -1] = weight[..., -1] * d_scale  # balance disparity to the same order as pixels
    w = (valid * weight).view(B, N, -1, 1)
    # # print("residual", ((w * r) ** 2).sum())

    # 2: construct linear system ###
    Ji = Ji.view(B, N, -1, D)
    Jj = Jj.view(B, N, -1, D)
    wJiT = (.001 * w * Ji).transpose(2, 3)
    wJjT = (.001 * w * Jj).transpose(2, 3)

    Hii = torch.matmul(wJiT, Ji)
    Hij = torch.matmul(wJiT, Jj)
    Hji = torch.matmul(wJjT, Ji)
    Hjj = torch.matmul(wJjT, Jj)

    vi = torch.matmul(wJiT, r).squeeze(-1)
    vj = torch.matmul(wJjT, r).squeeze(-1)

    # only optimize keyframe poses
    M = M - fixedp
    ii = ii - fixedp
    jj = jj - fixedp

    H = torch.zeros(B, M * M, D, D, device=target.device)  # batch, pose ** 2, var size, var size
    safe_scatter_add_mat(H, Hii, ii, ii, B, M, D)
    safe_scatter_add_mat(H, Hij, ii, jj, B, M, D)
    safe_scatter_add_mat(H, Hji, jj, ii, B, M, D)
    safe_scatter_add_mat(H, Hjj, jj, jj, B, M, D)
    H = H.reshape(B, M, M, D, D)

    v = torch.zeros(B, M, D, device=target.device)
    safe_scatter_add_vec(v, vi, ii, B, M, D)
    safe_scatter_add_vec(v, vj, jj, B, M, D)

    # 3: solve the system + apply retraction ###
    dx = block_solve(H, v, ep=ep, lm=lm)

    poses1, poses2 = poses[:, :fixedp], poses[:, fixedp:]
    poses2 = poses2.retr(dx)

    poses = lietorch.cat([poses1, poses2], dim=1)
    return poses
