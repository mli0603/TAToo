import csv
from argparse import ArgumentParser
from collections import defaultdict
import numpy as np


def parse_result(file):
    data_dict = defaultdict(list)
    with open(file) as csvfile:
        reader = csv.DictReader(csvfile, delimiter=',')
        for row in reader:
            for k, v in row.items():
                data_dict[k].append(v)

    if 'tau_p' in data_dict.keys():
        tau_p = np.asarray(data_dict['tau_p']).astype(float) * 1000
        phi_p = np.asarray(data_dict['phi_p']).astype(float)
        tau_d = np.asarray(data_dict['tau_d']).astype(float) * 1000
        phi_d = np.asarray(data_dict['phi_d']).astype(float)
        tau_rel = np.asarray(data_dict['tau_rel']).astype(float) * 1000
        phi_rel = np.asarray(data_dict['phi_rel']).astype(float)

        print('Patient mean +/- std %.2f %.2f %.2f %.2f' % (np.nanmean(tau_p), np.nanstd(tau_p),
                                                            np.nanmean(phi_p), np.nanstd(phi_p)))
        print('Drill mean +/- std %.2f %.2f %.2f %.2f' % (np.nanmean(tau_d), np.nanstd(tau_d),
                                                          np.nanmean(phi_d), np.nanstd(phi_d)))
        print('Relative mean +/- std %.2f %.2f %.2f %.2f' % (np.nanmean(tau_rel), np.nanstd(tau_rel),
                                                             np.nanmean(phi_rel), np.nanstd(phi_rel)))
        print('Failure rate patient: %.2f, drill: %.2f' % (np.isnan(tau_p).sum() / len(tau_p),
                                                           np.isnan(tau_d).sum() / len(tau_d)))

        phi_d[phi_d > 90] = phi_d[phi_d > 90] - 90  # hack for icp
        phi_d[phi_d > 45] = phi_d[phi_d > 45] - 45
        phi_d[phi_d > 22.5] = phi_d[phi_d > 22.5] - 22.5

    if 'epe' in data_dict.keys():
        epe = np.asarray(data_dict['epe']).astype(float)
        print('epe %.2f' % epe.mean())

        depth = np.asarray(data_dict['depth']).astype(float)
        print('depth (mm) %.2f' % (depth.mean() * 1000.0))

    if 'iou' in data_dict.keys():
        iou = np.asarray(data_dict['iou']).astype(float)
        iou_p = np.asarray(data_dict['iou_p']).astype(float)
        iou_d = np.asarray(data_dict['iou_d']).astype(float)
        print('iou mean %.2f, patient %.2f, drill %.2f' %
              (iou.mean(), iou_p.mean(), iou_d.mean()))

        dice = np.asarray(data_dict['dice']).astype(float)
        dice_p = np.asarray(data_dict['dice_p']).astype(float)
        dice_d = np.asarray(data_dict['dice_d']).astype(float)
        print('dice mean %.2f, patient %.2f, drill %.2f' %
              (dice.mean(), dice_p.mean(), dice_d.mean()))

    if 'optical_flow_epe_x' in data_dict.keys():
        optical_flow_epe_x = np.asarray(
            data_dict['optical_flow_epe_x']).astype(float)
        print('optical_flow_epe_x', optical_flow_epe_x.mean())
        optical_flow_epe_y = np.asarray(
            data_dict['optical_flow_epe_y']).astype(float)
        print('optical_flow_epe_y', optical_flow_epe_y.mean())

        scene_flow_epe_x = np.asarray(
            data_dict['scene_flow_epe_x']).astype(float)
        print('scene_flow_epe_x', scene_flow_epe_x.mean())
        scene_flow_epe_y = np.asarray(
            data_dict['scene_flow_epe_y']).astype(float)
        print('scene_flow_epe_y', scene_flow_epe_y.mean())
        scene_flow_epe_z = np.asarray(
            data_dict['scene_flow_epe_z']).astype(float)
        print('scene_flow_epe_z', scene_flow_epe_z.mean())

    if 'tau_p' in data_dict.keys():
        return tau_p, phi_p, tau_d, phi_d
    else:
        return


if __name__ == '__main__':
    parser = ArgumentParser()

    parser.add_argument('--files', type=str, nargs="+")

    args = parser.parse_args()

    for file in args.files:
        parse_result(file)
