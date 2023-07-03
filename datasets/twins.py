import os.path as osp

import h5py
import numpy as np
from mmcv.parallel import DataContainer as DC
from mmseg.datasets import DATASETS

from utils import binarize_seg_mask, quat2trans, trans2quat
from .base import BaseDataset

SCALE = 1.0


@DATASETS.register_module()
class TwinS(BaseDataset):
    def __init__(
            self,
            pipeline,
            disp_range=(1, 210),
            split=None,
            data_root=None,
            test_mode=False,
            num_samples=None,
            prefix_pattern=r'data_\d*.hdf5$',
            gt_keys=None,
            **kwargs,
    ):
        super(TwinS, self).__init__(pipeline=pipeline, disp_range=disp_range, split=split,
                                           data_root=data_root, test_mode=test_mode, num_samples=num_samples,
                                           prefix_pattern=prefix_pattern, gt_keys=gt_keys, **kwargs)
        self.patient_color_map = [
            [210, 210, 210]
        ]

        self.drill_color_map = [
            [0, 0, 0]
        ]

    def __getitem__(self, index):
        img_info = self.img_infos[index]
        filenames, ixs = img_info['filename'], img_info['index']

        pose_cam_list = []
        pose_drill_list = []
        pose_patient_list = []

        l_img_list = []
        r_img_list = []
        intrinsics_list = []
        disps_list = []
        segm_list = []

        for seq_idx, (filename, ix) in enumerate(zip(filenames, ixs)):
            ix = np.array([ix])
            if filename not in self.h5files:
                f = h5py.File(osp.join(filename), 'r')
                self.h5files[filename] = f
            else:
                f = self.h5files[filename]

            K = f['metadata']['camera_intrinsic']
            intrinsics = np.array([K[0, 0], K[1, 1], K[0, -1], K[1, -1]])[None]
            intrinsics_list.append(intrinsics)

            baseline = f['metadata']['baseline'][()] * SCALE

            t_cb_c = f['metadata']['T_cb_c']
            t_db_d = f['metadata']['T_db_d']
            t_pb_p = f['metadata']['T_pb_p']

            l_img = f['data']['l_img'][ix]
            r_img = f['data']['r_img'][ix]
            l_img_list.append(l_img)
            r_img_list.append(r_img)
            img_h, img_w = l_img.shape[1:3]

            segm = f['data']['segm'][ix]
            segm_patient = binarize_seg_mask(self.patient_color_map, segm)
            segm_drill = binarize_seg_mask(self.drill_color_map, segm)
            segm = np.concatenate([segm_patient[:, None], segm_drill[:, None]], axis=1)  # MxobjxHxW
            segm_list.append(segm)

            depth = f['data']['depth'][ix] * SCALE
            disps = baseline * K[0, 0] / depth  # MxHxW
            disps[depth > 1.0] = 0.0
            disps[~(segm_patient | segm_drill)] = 0.0  # remove background depth
            disps_list.append(disps)

            pose_cam = trans2quat(quat2trans(f['data']['pose_camhand'][ix].squeeze()) @ t_cb_c)[None]
            pose_cam[:, :3] = pose_cam[:, :3] * SCALE
            pose_cam_list.append(pose_cam)

            pose_drill = trans2quat(quat2trans(f['data']['pose_drill'][ix].squeeze()) @ t_db_d)[None]  # T_w_t
            pose_drill[:, :3] = pose_drill[:, :3] * SCALE
            pose_drill_list.append(pose_drill)

            pose_patient = trans2quat(quat2trans(f['data']['pose_pan'][ix].squeeze()) @ t_pb_p)[None]  # T_w_p
            pose_patient[:, :3] = pose_patient[:, :3] * SCALE
            pose_patient_list.append(pose_patient)

        results = self.list_to_result(l_img_list, r_img_list, pose_cam_list, pose_drill_list, pose_patient_list,
                                      intrinsics_list, disps_list, segm_list)

        results = self.aug(results)

        img_meta = {'disp_range': self.disp_range, 'flow_range': self.flow_range,
                    'img_shape': (img_h - self.border, img_w - self.border),
                    'baseline': baseline,
                    'filename': filename + ':' + '-'.join([str(ii) for ii in img_info['index']]),
                    'ori_filename': osp.basename(filename).replace('.hdf5', '') + '_' + '-'.join(
                        [str(ii) for ii in img_info['index']]), 'new_sequence': img_info['new_sequence']}
        results['img_metas'] = DC(img_meta, cpu_only=True)

        return results
