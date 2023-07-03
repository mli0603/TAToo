import argparse
import os
import os.path as osp
import re

import h5py
import numpy as np
from natsort import natsorted
from scipy.spatial.transform import Rotation


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
                     [2 * (x * y + z * w) / s, 1 - 2 * (x * x + z * z) /
                      s, 2 * (y * z - x * w) / s, t_y],
                     [2 * (x * z - y * w) / s, 2 * (y * z + x * w) /
                      s, 1 - 2 * (x * x + y * y) / s, t_z],
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


def check_motion_validity(curr_patient_pose, curr_drill_pose, prev_patient_pose, prev_drill_pose):
    """
    return small_motion, large_motion, prev poses

    small_motion: if the motion is not too small (simply skip this index)
    large_motion: if the motion is too large (need to specify this in split file as we cannot simply skip)
    """
    if prev_patient_pose is None:  # no prev, directly return
        return False, False
    else:
        d_patient_vec = curr_patient_pose[:3] - prev_patient_pose[:3]
        d_patient = np.linalg.norm(d_patient_vec)
        d_drill_vec = curr_drill_pose[:3] - prev_drill_pose[:3]
        d_drill = np.linalg.norm(d_drill_vec)
        rot_patient_curr = Rotation.from_quat(curr_patient_pose[3:])
        rot_patient_prev = Rotation.from_quat(prev_patient_pose[3:])
        rot_drill_curr = Rotation.from_quat(curr_drill_pose[3:])
        rot_drill_prev = Rotation.from_quat(prev_drill_pose[3:])
        ang_patient_vec = (rot_patient_curr.inv() *
                           rot_patient_prev).as_rotvec()
        ang_patient = np.linalg.norm(ang_patient_vec)
        ang_drill_vec = (rot_drill_curr.inv() * rot_drill_prev).as_rotvec()
        ang_drill = np.linalg.norm(ang_drill_vec)

        # motion too small
        if d_patient < 0.001 and d_drill < 0.001 and ang_patient < 0.01 and ang_drill < 0.01:
            return True, False
        # large motion
        if d_patient > 0.05 or d_drill > 0.05 or ang_patient > 0.3 or ang_drill > 0.4:
            return False, True
        return False, False


def write_all_data(args):
    if not osp.isdir(args.base_folder):
        print("target folder doesn't exist")
    else:
        all_lines = []
        for file in natsorted(os.listdir(args.base_folder)):
            prev_patient_pose = None
            prev_drill_pose = None

            if '_processed.hdf5' in file:
                file = osp.join(args.base_folder, file)
                f = h5py.File(file, 'r')

                t_cb_c = f['metadata']['T_cb_c']
                t_db_b = f['metadata']['T_db_d']
                t_pb_p = f['metadata']['T_pb_p']

                for idx in range(len(f['data']['time'])):
                    cam_pos = quat2trans(
                        f['data']['pose_camhand'][idx]) @ t_cb_c
                    drill_pose = quat2trans(
                        f['data']['pose_drill'][idx]) @ t_db_b
                    patient_pose = quat2trans(
                        f['data']['pose_pan'][idx]) @ t_pb_p

                    curr_patient_pose = trans2quat(
                        np.linalg.inv(cam_pos) @ patient_pose)
                    curr_drill_pose = trans2quat(
                        np.linalg.inv(cam_pos) @ drill_pose)

                    small_motion, large_motion = check_motion_validity(curr_patient_pose, curr_drill_pose,
                                                                       prev_patient_pose, prev_drill_pose)
                    if not small_motion:  # skip if motion too small
                        string_to_write = file.replace(args.base_folder, "")[
                            1:]  # skip first slash
                        string_to_write += " " + str(idx) + " " + str(
                            large_motion) + '\n'  # write if motion is large
                        all_lines.append(string_to_write)

                        prev_patient_pose = curr_patient_pose
                        prev_drill_pose = curr_drill_pose

        if args.train_val_ratio is None:
            split_file_train = open(
                args.base_folder + '/' + args.split_file, 'w')
            for line in all_lines:
                split_file_train.write(line)
        else:
            split_file_train = open(
                args.base_folder + '/' + args.split_file.replace('.txt', '_train.txt'), 'w')
            split_file_val = open(
                args.base_folder + '/' + args.split_file.replace('.txt', '_val.txt'), 'w')
            train_lines = all_lines[:int(
                len(all_lines) * args.train_val_ratio)]
            val_lines = all_lines[int(len(all_lines) * args.train_val_ratio):]
            for line in train_lines:
                split_file_train.write(line)
            for line in val_lines:
                split_file_val.write(line)


def resample_lines(args, f, lines, split_file_target):
    total_times = args.resampling_times
    for iter in range(total_times):
        if args.with_offset:
            start = np.arange(2 ** iter)
        else:
            start = [0]

        for ss in start:
            curr_lines = lines[ss::2 ** iter]
            print("number of data", len(curr_lines))
            prev_patient_pose = None
            prev_drill_pose = None
            prev_file = None
            prev_idx = None
            for line in curr_lines:
                filename, idx, _ = line.split(' ')
                h5py_file = osp.join(args.base_folder, filename)
                if h5py_file not in f:
                    f[h5py_file] = h5py.File(h5py_file, 'r')

                t_cb_c = f[h5py_file]['metadata']['T_cb_c']
                t_db_b = f[h5py_file]['metadata']['T_db_d']
                t_pb_p = f[h5py_file]['metadata']['T_pb_p']

                cam_pos = quat2trans(
                    f[h5py_file]['data']['pose_camhand'][int(idx)]) @ t_cb_c
                drill_pose = quat2trans(
                    f[h5py_file]['data']['pose_drill'][int(idx)]) @ t_db_b
                patient_pose = quat2trans(
                    f[h5py_file]['data']['pose_pan'][int(idx)]) @ t_pb_p

                curr_patient_pose = trans2quat(
                    np.linalg.inv(cam_pos) @ patient_pose)
                curr_drill_pose = trans2quat(
                    np.linalg.inv(cam_pos) @ drill_pose)

                if prev_file is None:
                    large_motion = True
                    small_motion = False
                else:
                    pattern = r'data_\d*.hdf5$'
                    first_entry_prefix = re.sub(pattern, "", prev_file)
                    new_entry_prefix = re.sub(pattern, "", h5py_file)

                    if first_entry_prefix == new_entry_prefix:
                        # check if motion is reasonable
                        small_motion, large_motion = check_motion_validity(curr_patient_pose, curr_drill_pose,
                                                                           prev_patient_pose, prev_drill_pose)

                if not small_motion:
                    string_to_write = filename + " " + idx + " " + \
                        str(large_motion) + '\n'  # write if motion is large
                    split_file_target.write(string_to_write)
                    prev_patient_pose = curr_patient_pose
                    prev_drill_pose = curr_drill_pose

                prev_file = h5py_file
                prev_idx = idx


def resampling(args):
    file = osp.join(args.base_folder, args.split_file)
    if not osp.isfile(file):
        print("target split file doens't exist")
    else:
        split_file = open(file, 'r')
        split_file_target = open(file.replace('.txt', '_reverse.txt'), 'w')
        lines = split_file.readlines()

        f = dict()
        resample_lines(args, f, lines, split_file_target)

        if args.reverse_order:
            lines = lines[::-1]
            resample_lines(args, f, lines, split_file_target)

    return


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--base_folder', type=str)
    parser.add_argument('--split_file', type=str)
    parser.add_argument('--train_val_ratio', type=float, default=None)

    parser.add_argument('--resampling', action='store_true')
    parser.add_argument('--resampling_times', type=int, default=3)
    parser.add_argument('--with_offset', action='store_true')
    parser.add_argument('--reverse_order', action='store_true')
    args = parser.parse_args()

    if args.resampling:
        resampling(args)
    else:
        write_all_data(args)
