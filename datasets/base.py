import copy
import os.path as osp
import re
import sys

import numpy as np
from mmcv.utils import build_from_cfg, print_log
from mmseg.datasets import DATASETS, CustomDataset, PIPELINES
from mmseg.utils import get_root_logger
from terminaltables import AsciiTable
from tqdm import tqdm

from utils import AverageMeter

sys.setrecursionlimit(
    100000
)  # NOTE: to avoid "RuntimeError: maximum recursion depth exceeded while calling a Python object"

SCALE = 1.0

@DATASETS.register_module()
class BaseDataset(CustomDataset):
    def __init__(
            self,
            pipeline,
            disp_range=(1, 210),
            flow_range=(1, 210),
            split=None,
            data_root=None,
            test_mode=False,
            num_samples=None,
            prefix_pattern=r'\d*_\d*.hdf5$',
            gt_keys=None,
            border=50,  # used to remove pixels along edges, which are usually not reliable
            **kwargs,
    ):

        self.test_mode = test_mode
        self.disp_range = disp_range
        self.flow_range = flow_range
        self.prefix_pattern = prefix_pattern

        self.num_frames = kwargs.get("num_frames", 2)
        self.gt_keys = gt_keys

        self.h5files = dict()
        self.img_infos = self.load_annotations(data_root, split, num_samples)

        self.aug = build_from_cfg(pipeline, PIPELINES)
        self.border = border
        self.patient_color_map = None
        self.drill_color_map = None

        self.compute_invalid = False  # default to False because simulation needs some pixels for background

    def update_mf_history(self, history, new_entry, num_frames, large_motion, pattern=r'\d*_\d*.hdf5$'):
        if num_frames > 0:
            if large_motion == 'True':  # if motion is large from current to previous, reset to a new sequence
                history = [new_entry]
                new_entry['new_sequence'] = 2
            elif len(history) == 0:
                history.append(new_entry)
                new_entry['new_sequence'] = 1
            else:
                first_entry_name = history[0]["filename"]
                first_entry_prefix = re.sub(pattern, "", first_entry_name)
                new_entry_name = new_entry["filename"]
                new_entry_prefix = re.sub(pattern, "", new_entry_name)
                if first_entry_prefix == new_entry_prefix:
                    history.append(new_entry)
                    new_entry['new_sequence'] = 0
                else:
                    # print("Changing from old sequence:", first_entry_prefix, "to new sequence", new_entry_prefix)
                    history = [new_entry]
                    new_entry['new_sequence'] = 1
            assert len(history) <= num_frames, "History cannot be longer than MF"
            if len(history) == num_frames:
                curr_history = copy.copy(history)
                first_entry = curr_history[0]
                first_entry['filename'] = [info["filename"] for info in curr_history]
                first_entry['index'] = [info["index"] for info in curr_history]
                history.pop(0)
                return first_entry, history
            else:
                return None, history

    def load_annotations(self, data_root, split, num_samples):
        img_infos = []
        history = []

        if split is not None:
            with open(split) as f:
                for line in f:
                    filename, index, large_motion = line.strip().split(' ')

                    img_info = dict(filename=osp.join(data_root, filename), index=int(index))
                    first_img_info, history = self.update_mf_history(history, img_info, self.num_frames, large_motion,
                                                                     pattern=self.prefix_pattern)
                    if first_img_info is not None:
                        img_infos.append(first_img_info)

        if num_samples is not None and 0 < num_samples <= len(img_infos):
            img_infos = img_infos[:num_samples]

        print_log(f"Loaded {len(img_infos)} images", logger=get_root_logger())
        return img_infos

    def __getitem__(self, index):
        raise NotImplemented('implement in child class')

    def list_to_result(self, l_img_list, r_img_list, pose_cam_list, pose_drill_list, pose_patient_list, intrinsics_list,
                       disps_list, segm_list):
        l_img = np.concatenate(l_img_list, axis=0)
        r_img = np.concatenate(r_img_list, axis=0)
        pose_cam = np.concatenate(pose_cam_list, axis=0)
        pose_drill = np.concatenate(pose_drill_list, axis=0)
        pose_patient = np.concatenate(pose_patient_list, axis=0)
        intrinsics = np.concatenate(intrinsics_list, axis=0)

        results = dict(left=l_img, right=r_img, gt_pose_cam=pose_cam, gt_pose_drill=pose_drill,
                       gt_pose_patient=pose_patient, intrinsics=intrinsics)
        results['gt_disp'] = np.concatenate(disps_list, axis=0)
        results['gt_semantic_seg'] = np.concatenate(segm_list, axis=0)

        return results

    def evaluate_disp(self, results, logger):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        # disp metric
        epe_meter = AverageMeter()
        depth_meter = AverageMeter()
        th3_meter = AverageMeter()
        for _, result in tqdm(enumerate(results)):
            epe_meter.update(result['epe'].item())
            depth_meter.update(result['depth'].item())
            th3_meter.update(result['th3'].item())

        # depth summary table
        summary_table_content = [
            ("epe", epe_meter, 1),
            ("depth (mm)", depth_meter, 1000.0 / SCALE),
            ("th3", th3_meter, 1)
        ]

        header = [k[0] for k in summary_table_content]
        summary_row = [np.round(k[1].avg * k[2], 3) for k in summary_table_content]

        summary_table_data = [header, summary_row]
        print_log("Summary:", logger)
        table = AsciiTable(summary_table_data)
        print_log("\n" + table.table, logger=logger)

        eval_results = {}
        for i in range(len(summary_table_data[0])):
            eval_results[summary_table_data[0][i].split(" ")[0]] = summary_table_data[1][i]

        return eval_results

    def evaluate_motion(self, results, logger):
        tau_p_meter = AverageMeter()
        phi_p_meter = AverageMeter()
        tau_d_meter = AverageMeter()
        phi_d_meter = AverageMeter()

        scene_flow_epe_x_meter_p = AverageMeter()
        scene_flow_epe_y_meter_p = AverageMeter()
        scene_flow_epe_z_meter_p = AverageMeter()
        scene_flow_epe_x_meter_d = AverageMeter()
        scene_flow_epe_y_meter_d = AverageMeter()
        scene_flow_epe_z_meter_d = AverageMeter()

        optical_flow_epe_x_meter_p = AverageMeter()
        optical_flow_epe_y_meter_p = AverageMeter()
        optical_flow_epe_x_meter_d = AverageMeter()
        optical_flow_epe_y_meter_d = AverageMeter()

        for _, result in tqdm(enumerate(results)):
            tau_p_meter.update(result['tau_p'].item())
            phi_p_meter.update(result['phi_p'].item())
            tau_d_meter.update(result['tau_d'].item())
            phi_d_meter.update(result['phi_d'].item())

            scene_flow_epe_x_meter_p.update(result['scene_flow_epe_x_p'].item())
            scene_flow_epe_y_meter_p.update(result['scene_flow_epe_y_p'].item())
            scene_flow_epe_z_meter_p.update(result['scene_flow_epe_z_p'].item())
            scene_flow_epe_x_meter_d.update(result['scene_flow_epe_x_d'].item())
            scene_flow_epe_y_meter_d.update(result['scene_flow_epe_y_d'].item())
            scene_flow_epe_z_meter_d.update(result['scene_flow_epe_z_d'].item())

            optical_flow_epe_x_meter_p.update(result['optical_flow_epe_x_p'].item())
            optical_flow_epe_y_meter_p.update(result['optical_flow_epe_y_p'].item())
            optical_flow_epe_x_meter_d.update(result['optical_flow_epe_x_d'].item())
            optical_flow_epe_y_meter_d.update(result['optical_flow_epe_x_d'].item())

        summary_table_content = [
            ("tau_p (mm)", tau_p_meter, 1000.0 / SCALE),
            ("phi_p", phi_p_meter, 1.0),
            ("tau_d (mm)", tau_d_meter, 1000.0 / SCALE),
            ("phi_d", phi_d_meter, 1.0),
            ("scene_flow_epe_x_p", scene_flow_epe_x_meter_p, 1.0),
            ("scene_flow_epe_y_p", scene_flow_epe_y_meter_p, 1.0),
            ("scene_flow_epe_z_p", scene_flow_epe_z_meter_p, 1.0),
            ("scene_flow_epe_x_d", scene_flow_epe_x_meter_d, 1.0),
            ("scene_flow_epe_y_d", scene_flow_epe_y_meter_d, 1.0),
            ("scene_flow_epe_z_d", scene_flow_epe_z_meter_d, 1.0),
            ("optical_flow_epe_x_p", optical_flow_epe_x_meter_p, 1.0),
            ("optical_flow_epe_y_p", optical_flow_epe_y_meter_p, 1.0),
            ("optical_flow_epe_x_d", optical_flow_epe_x_meter_d, 1.0),
            ("optical_flow_epe_y_d", optical_flow_epe_y_meter_d, 1.0)
        ]

        header = [k[0] for k in summary_table_content]
        summary_row = [np.round(k[1].avg * k[2], 3) for k in summary_table_content]

        summary_table_data = [header, summary_row]
        print_log("Summary:", logger)
        table = AsciiTable(summary_table_data)
        print_log("\n" + table.table, logger=logger)

        eval_results = {}
        for i in range(len(summary_table_data[0])):
            eval_results[summary_table_data[0][i].split(" ")[0]] = summary_table_data[1][i]
        return eval_results

    def evaluate_segmentation(self, results, logger):
        iou_meter = AverageMeter()
        acc_meter = AverageMeter()
        dice_meter = AverageMeter()

        iou_p_meter = AverageMeter()
        acc_p_meter = AverageMeter()
        dice_p_meter = AverageMeter()

        iou_d_meter = AverageMeter()
        acc_d_meter = AverageMeter()
        dice_d_meter = AverageMeter()

        for _, result in tqdm(enumerate(results)):
            iou_meter.update(result['iou'].item())
            acc_meter.update(result['acc'].item())
            dice_meter.update(result['dice'].item())

            iou_p_meter.update(result['iou_p'].item())
            acc_p_meter.update(result['acc_p'].item())
            dice_p_meter.update(result['dice_p'].item())

            iou_d_meter.update(result['iou_d'].item())
            acc_d_meter.update(result['acc_d'].item())
            dice_d_meter.update(result['dice_d'].item())

        summary_table_content = [
            ("iou", iou_meter, 1),
            ("acc", acc_meter, 1),
            ("dice", dice_meter, 1),

            ("iou_p", iou_meter, 1),
            ("acc_p", acc_meter, 1),
            ("dice_p", dice_meter, 1),

            ("iou_d", iou_meter, 1),
            ("acc_d", acc_meter, 1),
            ("dice_d", dice_meter, 1)
        ]

        header = [k[0] for k in summary_table_content]
        summary_row = [np.round(k[1].avg * k[2], 3) for k in summary_table_content]

        summary_table_data = [header, summary_row]
        print_log("Summary:", logger)
        table = AsciiTable(summary_table_data)
        print_log("\n" + table.table, logger=logger)

        eval_results = {}
        for i in range(len(summary_table_data[0])):
            eval_results[summary_table_data[0][i].split(" ")[0]] = summary_table_data[1][i]
        return eval_results

    def evaluate(self, results, metric="default", logger=None, **kwargs):
        """Evaluate the dataset.

        Args:
            results (list): Testing results of the dataset.
            metric (str | list[str]): Metrics to be evaluated.
            logger (logging.Logger | None | str): Logger used for printing
                related information during evaluation. Default: None.

        Returns:
            dict[str, float]: Default metrics.
        """
        if not isinstance(metric, str):
            assert len(metric) == 1
            metric = metric[0]
        allowed_metrics = ["default", "stereo_only", "motion_only", "segmentation_only"]
        if metric not in allowed_metrics:
            raise KeyError("metric {} is not supported".format(metric))

        if metric == "stereo_only":
            return self.evaluate_disp(results, logger)
        elif metric == "motion_only":
            return self.evaluate_motion(results, logger)
        elif metric == "segmentation_only":
            return self.evaluate_segmentation(results, logger)
        elif metric == "default":
            eval_results = self.evaluate_disp(results, logger)
            eval_results.update(self.evaluate_motion(results, logger))
            eval_results.update(self.evaluate_segmentation(results, logger))
            return eval_results
