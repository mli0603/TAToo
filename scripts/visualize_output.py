import os
import os.path as osp
from argparse import ArgumentParser

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from PIL import Image
from natsort import natsorted
from pytransform3d import transformations as pt
from tqdm import tqdm


def write_left_video(args):
    assert args.split_file is not None
    assert args.base_dir is not None

    f = open(osp.join(args.base_dir, args.split_file), 'r')
    lines = f.readlines()

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(
        osp.join(args.video_write_dir, 'left.avi'), fourcc, args.hz, (w, h))
    f = dict()
    for line in tqdm(lines):
        filename, idx, large_motion_read = line.split(' ')
        h5py_file = osp.join(args.base_dir, filename)
        if h5py_file not in f:
            f[h5py_file] = h5py.File(h5py_file, 'r')
        if large_motion_read.strip() == 'False':
            left = f[h5py_file]['data']['l_img'][int(idx)]
            video_writer.write(left)
    return


def write_seg_video(args):
    assert args.data_dir is not None

    files = os.listdir(args.data_dir)

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(
        osp.join(args.video_write_dir, 'seg.avi'), fourcc, args.hz, (w, h))
    for file in tqdm(natsorted(files)):
        if '_segmentation' in file:
            seg_for_vis = np.zeros([h, w, 3])
            seg = np.load(osp.join(args.data_dir, file))['data']
            seg_for_vis += seg[0, 0, 0, :h, :w][..., None] * \
                np.array([255, 125, 125])[None, None]
            seg_for_vis += seg[0, 0, 1, :h, :w][..., None] * \
                np.array([125, 255, 125])[None, None]
            seg_for_vis = seg_for_vis.astype(np.uint8)[..., ::-1]
            video_writer.write(seg_for_vis)
    return


def write_disp_video(args):
    assert args.data_dir is not None

    files = os.listdir(args.data_dir)
    sns.set_theme()
    cmap = plt.get_cmap()

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(
        osp.join(args.video_write_dir, 'disp.avi'), fourcc, args.hz, (w, h))
    for file in tqdm(natsorted(files)):
        if '_disp' in file:
            disp = np.load(osp.join(args.data_dir, file))['data'].astype(float)
            disp_for_vis = (disp[0, 0, :h, :w] - 64.0) / (256.0 - 64)
            disp_for_vis = cmap(disp_for_vis)
            disp_for_vis = (disp_for_vis[..., :3]
                            [..., ::-1] * 255).astype(np.uint8)
            video_writer.write(disp_for_vis)
    return


def write_flow_video(args):
    assert args.data_dir is not None

    files = os.listdir(args.data_dir)
    sns.set_theme()
    cmap = plt.get_cmap()

    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    video_writer = cv2.VideoWriter(
        osp.join(args.video_write_dir, 'flow.avi'), fourcc, args.hz, (w, h))
    for file in tqdm(natsorted(files)):
        if '_flow' in file:
            flow = np.load(osp.join(args.data_dir, file))['data'].astype(float)
            flow_for_vis = flow[0, 0, :h, :w]
            flow_for_vis = (flow_for_vis + 7.5) / 15.0
            flow_for_vis = cmap(flow_for_vis)
            flow_for_vis = (
                flow_for_vis[..., 1, [1, 0, 2]] * 255).astype(np.uint8)[..., ::-1]
            video_writer.write(flow_for_vis)
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--video_write_dir', type=str)
    parser.add_argument('--hz', type=int, default=10)
    parser.add_argument('--video_size', nargs="+",
                        type=int, default=[360, 640])
    parser.add_argument('--data_dir', type=str, default=None)

    parser.add_argument('--left', action='store_true', help='Visualize left images')
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--base_dir', type=str, default=None)

    parser.add_argument('--seg', action='store_true', help='Visualize segmentation output')
    parser.add_argument('--disp', action='store_true', help='Visualize disparity output')
    parser.add_argument('--flow', action='store_true', help='Visualize flow output')

    args = parser.parse_args()
    h = int(args.video_size[0])
    w = int(args.video_size[1])
    if args.left:
        write_left_video(args)
    if args.seg:
        write_seg_video(args)
    if args.disp:
        write_disp_video(args)
    if args.flow:
        write_flow_video(args)
