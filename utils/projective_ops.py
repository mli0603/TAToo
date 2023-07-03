import torch
from lietorch import SE3, Sim3

from .misc import compute_valid_mask, repeat_batch


# MIN_DEPTH = 0.1


def extract_intrinsics(intrinsics):
    return intrinsics[..., None, None, :].unbind(dim=-1)


def iproj(disps, intrinsics):
    """ pinhole camera inverse projection """
    ht, wd = disps.shape[2:]
    fx, fy, cx, cy = extract_intrinsics(intrinsics)

    y, x = torch.meshgrid(
        torch.arange(ht).to(disps.device).float(),
        torch.arange(wd).to(disps.device).float())

    i = torch.ones_like(disps)
    X = (x - cx) / fx
    Y = (y - cy) / fy
    return torch.stack([X, Y, i, disps], dim=-1)


def proj(Xs, intrinsics, jacobian=False):
    """ pinhole camera projection """
    fx, fy, cx, cy = extract_intrinsics(intrinsics)
    X, Y, Z, D = Xs.unbind(dim=-1)
    d = torch.where(Z.abs() < 0.001, torch.zeros_like(Z), 1.0 / Z)  # for numerical stability

    x = fx * (X * d) + cx
    y = fy * (Y * d) + cy
    coords = torch.stack([x, y, D * d], dim=-1)

    if jacobian:
        B, N, H, W = d.shape
        o = torch.zeros_like(d)
        proj_jac = torch.stack([
            fx * d, o, -fx * X * d * d, o,
            o, fy * d, -fy * Y * d * d, o,
            o, o, -D * d * d, d,
        ], dim=-1).view(B, N, H, W, 3, 4)

        return coords, proj_jac

    return coords, None


def actp(Gij, X0, jacobian=False):
    """ action on point cloud """
    X1 = Gij[:, :, None, None] * X0

    if jacobian:
        X, Y, Z, d = X1.unbind(dim=-1)
        o = torch.zeros_like(d)
        B, N, H, W = d.shape

        if isinstance(Gij, SE3):
            Ja = torch.stack([
                d, o, o, o, Z, -Y,
                o, d, o, -Z, o, X,
                o, o, d, Y, -X, o,
                o, o, o, o, o, o,
            ], dim=-1).view(B, N, H, W, 4, 6)

        elif isinstance(Gij, Sim3):
            Ja = torch.stack([
                d, o, o, o, Z, -Y, X,
                o, d, o, -Z, o, X, Y,
                o, o, d, Y, -X, o, Z,
                o, o, o, o, o, o, o
            ], dim=-1).view(B, N, H, W, 4, 7)

        return X1, Ja

    return X1, None


def projective_transform(poses, disps, intrinsics, ii, jj, meta, jacobian=False):
    """ map points from ii->jj """
    # poses: BobjxMxHxW
    # disps: BxMxHxW
    # intrinsics: BxMx4
    # ii, jj: N

    disps, intrinsics = repeat_batch([disps, intrinsics], poses.shape[0])

    # inverse project (pinhole)
    X0 = iproj(disps[:, ii], intrinsics[:, ii])
    # X0 = X0.expand(poses.shape[0], -1, -1, -1, -1)  # expand batch dim to match poses, BobjxNxHxWx4

    # transform
    Gij = poses[:, jj] * poses[:, ii].inv()
    X1, Ja = actp(Gij, X0, jacobian=jacobian)

    # project (pinhole)
    x1, Jp = proj(X1, intrinsics[:, jj], jacobian=jacobian)

    # exclude points out of range
    fx = intrinsics[..., 0, None, None]
    valid_0 = compute_valid_mask(meta, gt_disp=disps[:, ii] * fx[:, ii] * meta['baseline'])
    valid_1 = compute_valid_mask(meta, gt_disp=x1[..., -1] * fx[:, jj] * meta['baseline'])
    valid = (valid_0 & valid_1).float().unsqueeze(-1)
    # valid = ((X1[..., 2] > MIN_DEPTH) & (X0[..., 2] > MIN_DEPTH)).float()
    # valid = valid.unsqueeze(-1)

    if jacobian:
        Jj = torch.matmul(Jp, Ja)
        Ji = -Gij[:, :, None, None, None].adjT(Jj)
        return x1, valid, (Ji, Jj)

    return x1, valid
