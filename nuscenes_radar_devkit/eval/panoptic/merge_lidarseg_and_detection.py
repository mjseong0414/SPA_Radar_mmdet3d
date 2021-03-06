"""
Script to generate nuScenes-panoptic predictions from nuScenes-detection and nuScene-lidarseg predictions.
Code written by Motional and the Robot Learning Lab, University of Freiburg.
"""
import argparse
import os
from typing import List, Tuple

import numpy as np
from tqdm import tqdm

from nuscenes_radar_devkit.nuscenes import NuScenes
from nuscenes_radar_devkit.nuscenes.eval.common.loaders import load_prediction, add_center_dist
from nuscenes_radar_devkit.nuscenes.eval.common.utils import boxes_to_sensor
from nuscenes_radar_devkit.nuscenes.eval.detection.data_classes import DetectionBox
from nuscenes_radar_devkit.nuscenes.eval.lidarseg.utils import get_samples_in_eval_set
from nuscenes_radar_devkit.nuscenes.utils.data_classes import LidarSegPointCloud
from nuscenes_radar_devkit.nuscenes.utils.geometry_utils import points_in_box

from nuscenes_radar_devkit.nuscenes.eval.panoptic.utils import PanopticClassMapper

OVERLAP_THRESHOLD = 0.5
CONFIDENCE_THRESHOLD = 0.5


def generate_panoptic_labels(nusc: NuScenes,
                             seg_folder: str,
                             det_json: str,
                             eval_set: str,
                             out_dir: str = None,
                             verbose: bool = False) -> None:
    """
    Generate NuScenes lidar panoptic predictions.
    :param nusc: NuScenes instance.
    :param seg_folder: Path to the directory where the lidarseg predictions are stored.
    :param det_json: Path of the json where the detection predictions are stored.
    :param eval_set: Which dataset split to evaluate on, train, val or test.
    :param out_dir: Path to save any output to.
    :param verbose: Whether to print any output.
    """
    sample_tokens = get_samples_in_eval_set(nusc, eval_set)
    num_samples = len(sample_tokens)
    mapper = PanopticClassMapper(nusc)
    coarse2idx = mapper.get_coarse2idx()
    if verbose:
        print(f'There are {num_samples} samples.')

    panoptic_subdir = os.path.join('panoptic', eval_set)
    panoptic_dir = os.path.join(out_dir, panoptic_subdir)
    os.makedirs(panoptic_dir, exist_ok=True)

    # Load the predictions.
    pred_boxes_all, meta = load_prediction(det_json, 1000, DetectionBox, verbose=verbose)
    pred_boxes_all = add_center_dist(nusc, pred_boxes_all)

    for sample_token in tqdm(sample_tokens, disable=not verbose):
        sample = nusc.get('sample', sample_token)

        # TODO detection does not do scene_token stuff

        # Get the sample data token of the point cloud.
        sd_record = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_record['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_record['ego_pose_token'])

        lidar_path = os.path.join(nusc.dataroot, sd_record['filename'])
        # Load the predictions for the point cloud.
        lidarseg_pred_filename = os.path.join(seg_folder, 'lidarseg', nusc.version.split('-')[-1],
                                              sd_record['token'] + '_lidarseg.bin')
        lidar_seg = LidarSegPointCloud(lidar_path, lidarseg_pred_filename)

        panop_labels = np.zeros(lidar_seg.labels.shape, dtype=np.uint16)
        overlaps = np.zeros(lidar_seg.labels.shape, dtype=np.uint8)

        pred_boxes = pred_boxes_all[sample_token]
        pred_boxes = confidence_threshold(pred_boxes)
        sorted_pred_boxes, pred_cls = sort_confidence(pred_boxes)
        sorted_pred_boxes = boxes_to_sensor(sorted_pred_boxes, pose_record, cs_record)

        for instance_id, (pred_box, cl) in enumerate(zip(sorted_pred_boxes, pred_cls)):
            cl_id = coarse2idx[cl]
            # TODO detection does not do scene_token stuff
            msk = np.zeros(lidar_seg.labels.shape, dtype=np.uint8)
            indices = np.where(points_in_box(pred_box, lidar_seg.points[:, :3].T))[0]

            msk[indices] = 1
            msk[np.logical_and(lidar_seg.labels != cl_id, msk == 1)] = 0
            intersection = np.logical_and(overlaps, msk)
            if np.sum(intersection) / np.float32(np.sum(msk)) > OVERLAP_THRESHOLD:
                continue
            # Add non-overlapping part to output.
            msk = msk - intersection
            panop_labels[msk != 0] = (cl_id * 1000 + instance_id + 1)
            overlaps += msk

        stuff_msk = np.logical_and(panop_labels == 0, lidar_seg.labels >= 11)
        panop_labels[stuff_msk] = lidar_seg.labels[stuff_msk] * 1000
        panoptic_file = sd_record['token'] + '_panoptic.npz'
        np.savez_compressed(os.path.join(panoptic_dir, panoptic_file), data=panop_labels.astype(np.uint16))


def confidence_threshold(boxes: List[DetectionBox]) -> List[DetectionBox]:
    """
    Filter a list of boxes with a given confidence threshold.
    :param boxes: A list of boxes.
    :return: A list of boxes with confidences higher than the threshold.
    """
    return [box for box in boxes if box.detection_score > CONFIDENCE_THRESHOLD]


def sort_confidence(boxes: List[DetectionBox]) -> Tuple[List[DetectionBox], List[str]]:
    """
    Sort a list of boxes by confidence.
    :param boxes: A list of boxes.
    :return: A list of boxes sorted by confidence and a list of classes corresponding to each box.
    """
    scores = [box.detection_score for box in boxes]
    inds = np.argsort(scores)[::-1]
    sorted_bboxes = []
    for ind in inds:
        sorted_bboxes.append(boxes[ind])
    sorted_cls = [box.detection_name for box in sorted_bboxes]
    return sorted_bboxes, sorted_cls


def main():
    """
    The main method for this script.
    """
    parser = argparse.ArgumentParser(
        description='Generate panoptic from nuScenes-lidarseg and detection results.')
    parser.add_argument('--seg_path', type=str, help='The path to the segmentation results folder.')
    parser.add_argument('--det_path', type=str, help='The path to the detection json file.')
    parser.add_argument('--eval_set', type=str, default='val',
                        help='Which dataset split to evaluate on, train, val or test.')
    parser.add_argument('--dataroot', type=str, default='/data/sets/nuscenes', help='Default nuScenes data directory.')
    parser.add_argument('--version', type=str, default='v1.0-trainval',
                        help='Which version of the nuScenes dataset to evaluate on, e.g. v1.0-trainval.')
    parser.add_argument('--verbose', type=bool, default=False, help='Whether to print to stdout.')
    parser.add_argument('--out_dir', type=str, default=None, help='Folder to write the panoptic labels to.')
    args = parser.parse_args()

    out_dir = args.out_dir if args.out_dir is not None else f'nuScenes-panoptic-merged-prediction-{args.version}'

    print(f'Start Generation... \nArguments: {args}')
    nusc = NuScenes(version=args.version, dataroot=args.dataroot, verbose=args.verbose)

    generate_panoptic_labels(nusc=nusc,
                             seg_folder=args.seg_path,
                             det_json=args.det_path,
                             eval_set=args.eval_set,
                             out_dir=out_dir,
                             verbose=args.verbose)
    print(f'Generated results saved at {out_dir}. \nFinished.')


if __name__ == '__main__':
    main()
