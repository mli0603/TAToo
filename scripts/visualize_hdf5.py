import argparse
import os.path as osp

import cv2
import h5py
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from tqdm import tqdm


def write_to_video(args):
    fourcc = cv2.VideoWriter_fourcc('M', 'J', 'P', 'G')
    if args.base_folder is None:
        video_file = 'video.avi'
    else:
        video_file = osp.join(args.base_folder, 'video.avi')
    video_writer = cv2.VideoWriter(video_file, fourcc, 10, (640 * 4, 360))
    sns.set_theme()
    cmap = plt.get_cmap()

    if args.split_file is not None:
        split_file = open(args.split_file, 'r')
        lines = split_file.readlines()
        f = dict()
        for ii, l in enumerate(tqdm(lines)):
            filename, idx, large_motion = l.split(' ')
            if large_motion.strip() == 'True':
                continue
            filename = osp.join(args.base_folder, filename)
            if filename not in f:
                f[filename] = h5py.File(filename, 'r')
            intrinsics = f[filename]['metadata']['camera_intrinsic']
            bl = f[filename]['metadata']['baseline']
            l_img = f[filename]['data']['l_img'][int(idx)]
            r_img = f[filename]['data']['r_img'][int(idx)]
            seg_img = f[filename]['data']['segm'][int(idx)]
            depth = f[filename]['data']['depth'][int(idx)]
            depth[depth > 1.0] = np.inf
            disp = intrinsics[0, 0] * bl / depth
            disp_for_vis = (disp - 8.0) / (320.0 - 8.0)
            disp_for_vis = cmap(disp_for_vis)
            disp_for_vis = (disp_for_vis[..., :3]
                            [..., ::-1] * 255).astype(np.uint8)
            img = np.concatenate([l_img, r_img, seg_img, disp_for_vis], axis=1)

            video_writer.write(img)

    elif args.hdf5_file is not None:
        sns.set_theme()
        cmap = plt.get_cmap('flare_r')
        for h5pyfile in args.hdf5_file:
            filename = osp.join(args.base_folder, h5pyfile)
            f = h5py.File(filename, 'r')
            print("keys", f['data'].keys())
            for idx in tqdm(range(len(f['data']['time']))):
                intrinsics = f['metadata']['camera_intrinsic']
                bl = f['metadata']['baseline']
                l_img = f['data']['l_img'][int(idx)]
                r_img = f['data']['r_img'][int(idx)]
                seg_img = f['data']['segm'][int(idx)]
                depth = f['data']['depth'][int(idx)]
                depth[depth > 1.0] = np.inf
                disp = intrinsics[0, 0] * bl / depth
                disp_for_vis = (disp - 8.0) / (320.0 - 8.0)
                disp_for_vis = cmap(disp_for_vis)
                disp_for_vis = (disp_for_vis[..., :3]
                                [..., ::-1] * 255).astype(np.uint8)
                img = np.concatenate([l_img, r_img, seg_img, disp_for_vis], axis=1)

                video_writer.write(img)

    video_writer.release()
    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str)
    parser.add_argument('--split_file', type=str, default=None)
    parser.add_argument('--hdf5_file', nargs='+', default=None)
    args = parser.parse_args()

    write_to_video(args)
