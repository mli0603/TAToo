import os
from pathlib import Path
import sys

code_path = Path(os.path.dirname(os.path.realpath(__file__))).parents[0]  # NOQA
sys.path.append(code_path.__str__())  # NOQA

from argparse import ArgumentParser
import os.path as osp

import mmcv
import torch
from mmcv.runner import load_checkpoint
from mmcv.utils import DictAction

from apis import multi_gpu_inference, single_gpu_inference
from model import build_estimator
from datasets.stereo_augmentor import StereoNormalizor

from mmcv.utils import print_log, mkdir_or_exist
import glob
import cv2
import numpy as np
from tqdm import tqdm
from natsort import natsorted


def inference_on_images(args):
    if args.config is not None:
        cfg = mmcv.Config.fromfile(args.config)
    else:
        print(code_path)
        cfg = mmcv.Config.fromfile(
            osp.join(code_path, 'configs/models/tatoo.py'))

    # build the model and load checkpoint
    cfg.model.train_cfg = None
    model = build_estimator(cfg.model, test_cfg=cfg.get("test_cfg"))
    load_checkpoint(model, args.checkpoint, map_location="cpu")
    model.eval()
    model = model.cuda()

    mkdir_or_exist(args.output_dir)

    left_imgs = natsorted(glob.glob(args.left))
    right_imgs = natsorted(glob.glob(args.right))
    img_meta = dict(disp_range=(32.0, 256.0), flow_range=(0.0, 160.0))
    preprocess_op = StereoNormalizor()

    for i, (left, right) in enumerate(tqdm(zip(left_imgs, right_imgs))):
        if i == 0:
            left_prev = cv2.imread(left)
            right_prev = cv2.imread(right)
            h, w, _ = left_prev.shape
            if args.intrinsics is not None:
                intrinsics = torch.tensor(args.intrinsics)
            else:
                intrinsics = torch.tensor([1.5*h, 1.5*h, 0.25 * w, 0.5 * h])
            img_meta.update({
                'img_shape': (h, w),
                'baseline': args.baseline,
            })

            continue

        left_curr = cv2.imread(left)
        right_curr = cv2.imread(right)

        data = dict(left=np.stack([left_prev, left_curr]),
                    right=np.stack([right_prev, right_curr]))
        data = preprocess_op(data)
        data['left'] = data['left'][None].cuda()  # expand batch dim
        data['right'] = data['right'][None].cuda()  # expand batch dim
        data['intrinsics'] = intrinsics[None, None].expand(1, 2, -1).cuda()
        data['img_metas'] = [img_meta]

        with torch.no_grad():
            result = model(return_loss=False, rescale=True,
                           evaluate=False, **data)

        out_file = osp.join(args.output_dir, osp.splitext(left)[0])
        model.show_result(
            None,
            result,
            show=True,
            out_file=out_file,
            inp=data,
        )

        left_prev = left_curr
        right_prev = right_curr
    return


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--left', type=str, required=True,
                        help='Filename pattern of left images')
    parser.add_argument('--right', type=str, required=True,
                        help='Filename pattern of right images')
    parser.add_argument('--checkpoint', type=str, required=True,
                        help='Path to ckpt file')

    parser.add_argument('--config', type=str, default=None,
                        help='Path to config file')
    parser.add_argument('--output_dir', type=str, default='work_dirs/output',
                        help='Output directory')

    parser.add_argument('--baseline', type=float, default=0.025,
                        help='Baseline in meters; default 0.025')
    parser.add_argument('--intrinsics', nargs=4, default=None,
                        help='Intrinsics fx, fy, cx, cy in pixels; default 1.5*h, 1.5*h, 0.5w, 0.5h')

    args = parser.parse_args()

    inference_on_images(args)
