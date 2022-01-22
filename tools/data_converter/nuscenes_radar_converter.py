# Copyright (c) OpenMMLab. All rights reserved.
import mmcv
import numpy as np
import os
import copy
import sys
from collections import OrderedDict
from nuscenes_radar_devkit.nuscenes import NuScenes
from nuscenes.utils.geometry_utils import view_points
from os import path as osp
from pyquaternion import Quaternion
from shapely.geometry import MultiPoint, box
from typing import List, Tuple, Union
from pathlib import Path

from mmdet3d.core.bbox.box_np_ops import points_cam2img
from mmdet3d.core.bbox.box_np_ops import points_count_rbbox
from mmdet3d.core.bbox.box_np_ops import points_count_rbbox_second
from mmdet3d.core.bbox.box_np_ops import points_in_rbbox
from mmdet3d.datasets import NuScenesRadarDataset

nus_categories = ('car', 'truck', 'trailer', 'bus', 'construction_vehicle',
                  'bicycle', 'motorcycle', 'pedestrian', 'traffic_cone',
                  'barrier')

nus_attributes = ('cycle.with_rider', 'cycle.without_rider',
                  'pedestrian.moving', 'pedestrian.standing',
                  'pedestrian.sitting_lying_down', 'vehicle.moving',
                  'vehicle.parked', 'vehicle.stopped', 'None')


def create_nuscenes_infos(root_path,
                          info_prefix,
                          home_path,
                          radar_version,
                          filter_version,
                          version='v1.0-trainval',
                          max_sweeps=10):
    """Create info file of nuscene dataset.

    Given the raw data, generate its related info file in pkl format.

    Args:
        root_path (str): Path of the data root.
        info_prefix (str): Prefix of the info file to be generated.
        version (str): Version of the data.
            Default: 'v1.0-trainval'
        max_sweeps (int): Max number of sweeps.
            Default: 10
    """
    from nuscenes_radar_devkit.nuscenes import NuScenes
    from nuscenes_radar_devkit.can_bus.can_bus_api import NuScenesCanBus
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    nusc_can = NuScenesCanBus(dataroot=root_path)
    from nuscenes_radar_devkit.utils import splits
    available_vers = ['v1.0-trainval', 'v1.0-test', 'v1.0-mini']
    assert version in available_vers
    if version == 'v1.0-trainval':
        train_scenes = splits.train
        val_scenes = splits.val
    elif version == 'v1.0-test':
        train_scenes = splits.test
        val_scenes = []
    elif version == 'v1.0-mini':
        train_scenes = splits.mini_train
        val_scenes = splits.mini_val
    else:
        raise ValueError('unknown')
    
    # filter existing scenes.
    available_scenes = _get_available_scenes(nusc)
    available_scene_names = [s['name'] for s in available_scenes]
    train_scenes = list(
        filter(lambda x: x in available_scene_names, train_scenes))
    val_scenes = list(filter(lambda x: x in available_scene_names, val_scenes))
    train_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in train_scenes
    ])
    val_scenes = set([
        available_scenes[available_scene_names.index(s)]['token']
        for s in val_scenes
    ])
    
    test = 'test' in version
    if test:
        print('test scene: {}'.format(len(train_scenes)))
    else:
        print('train scene: {}, val scene: {}'.format(
            len(train_scenes), len(val_scenes)))
    
    train_nusc_infos, val_nusc_infos = _fill_trainval_infos_official(
        nusc, nusc_can, train_scenes, val_scenes, radar_version, root_path, filter_version,test=test, max_sweeps=max_sweeps)

    metadata = dict(version=version)
    if test:
        print('test sample: {}'.format(len(train_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(home_path,'{}_infos_test.pkl'.format(info_prefix))
        mmcv.dump(data, info_path)
    else:
        print('train sample: {}, val sample: {}'.format(
            len(train_nusc_infos), len(val_nusc_infos)))
        data = dict(infos=train_nusc_infos, metadata=metadata)
        info_path = osp.join(home_path,f"{info_prefix}_infos_train_{filter_version}_{radar_version}_sweeps{max_sweeps}.pkl")
        mmcv.dump(data, info_path)
        data['infos'] = val_nusc_infos
        info_val_path = osp.join(home_path,f"{info_prefix}_infos_val_{filter_version}_{radar_version}_sweeps{max_sweeps}.pkl")
        mmcv.dump(data, info_val_path)


def _get_available_scenes(nusc):
    """Get available scenes from the input nuscenes class.

    Given the raw data, get the information of available scenes for
    further info generation.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.

    Returns:
        available_scenes (list[dict]): List of basic information for the
            available scenes.
    """
    available_scenes = []
    print('total scene num: {}'.format(len(nusc.scene)))
    for scene in nusc.scene:
        scene_token = scene['token']
        scene_rec = nusc.get('scene', scene_token)
        sample_rec = nusc.get('sample', scene_rec['first_sample_token'])
        sd_rec = nusc.get('sample_data', sample_rec['data']['LIDAR_TOP'])
        has_more_frames = True
        scene_not_exist = False
        while has_more_frames:
            lidar_path, boxes, _ = nusc.get_sample_data(sd_rec['token'])
            lidar_path = str(lidar_path)
            if os.getcwd() in lidar_path:
                # path from lyftdataset is absolute path
                lidar_path = lidar_path.split(f'{os.getcwd()}/')[-1]
                # relative path
            if not mmcv.is_filepath(lidar_path):
                scene_not_exist = True
                break
            else:
                break
        if scene_not_exist:
            continue
        available_scenes.append(scene)
    print('exist scene num: {}'.format(len(available_scenes)))
    return available_scenes

def _get_available_samples(nusc):
    available_samples = []
    can_black_list = [161, 162, 163, 164, 165, 166, 167, 168, 170, 171, 172, 173, 174, 175, 176, 309, 310, 311, 312, 313, 314]
    for i in range(len(nusc.sample)):
        scene_tk = nusc.sample[i]['scene_token']
        scene_mask = int(nusc.get('scene',scene_tk)['name'][-4:])
        if scene_mask in can_black_list:
            continue
        else:
            available_samples.append(nusc.sample[i])
    return available_samples

def _fill_trainval_infos_official(nusc,nusc_can,
                         train_scenes,
                         val_scenes,
                         radar_version,
                         root_path,
                         filter_version,
                         test=False,
                         max_sweeps=10):
    """Generate the train/val infos from the raw data.

    Args:
        nusc (:obj:`NuScenes`): Dataset class in the nuScenes dataset.
        train_scenes (list[str]): Basic information of training scenes.
        val_scenes (list[str]): Basic information of validation scenes.
        test (bool): Whether use the test mode. In the test mode, no
            annotations can be accessed. Default: False.
        max_sweeps (int): Max number of sweeps. Default: 10.

    Returns:
        tuple[list[dict]]: Information of training set and validation set
            that will be saved to the info file.
    """
    train_nusc_infos = []
    val_nusc_infos = []
    # total_samples = _get_available_samples(nusc)
    for sample in mmcv.track_iter_progress(nusc.sample):
        lidar_token = sample['data']['LIDAR_TOP']
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        cs_record = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
        pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
        lidar_path, boxes, _ = nusc.get_sample_data(lidar_token)

        # radar path
        radar_f_token = sample['data']['RADAR_FRONT']
        radar_fr_token = sample['data']['RADAR_FRONT_RIGHT']
        radar_fl_token = sample['data']['RADAR_FRONT_LEFT']
        radar_bl_token = sample['data']['RADAR_BACK_LEFT']
        radar_br_token = sample['data']['RADAR_BACK_RIGHT']
        radar_f_path,_,_ = nusc.get_sample_data(radar_f_token)
        radar_fr_path,_,_ = nusc.get_sample_data(radar_fr_token)
        radar_fl_path,_,_ = nusc.get_sample_data(radar_fl_token)
        radar_bl_path,_,_ = nusc.get_sample_data(radar_bl_token)
        radar_br_path,_,_ = nusc.get_sample_data(radar_br_token)
        # radar_path = [radar_f_path, radar_fr_path, radar_fl_path, radar_bl_path, radar_br_path]
        radar_npy = make_multisweep_radar_data_official(nusc,nusc_can,sample,radar_version,max_sweeps,root_path)
        mmcv.check_file_exist(lidar_path)

        info = {
            'lidar_path': lidar_path,
            'radar_path': radar_f_path,
            'token': sample['token'],
            'sweeps': [],
            'radar_sweeps':[],
            'cams': dict(),
            'lidar2ego_translation': cs_record['translation'],
            'lidar2ego_rotation': cs_record['rotation'],
            'ego2global_translation': pose_record['translation'],
            'ego2global_rotation': pose_record['rotation'],
            'timestamp': sample['timestamp'],
        }

        l2e_r = info['lidar2ego_rotation']
        l2e_t = info['lidar2ego_translation']
        e2g_r = info['ego2global_rotation']
        e2g_t = info['ego2global_translation']
        l2e_r_mat = Quaternion(l2e_r).rotation_matrix
        e2g_r_mat = Quaternion(e2g_r).rotation_matrix

        # obtain 6 image's information per frame
        camera_types = [
            'CAM_FRONT',
            'CAM_FRONT_RIGHT',
            'CAM_FRONT_LEFT',
            'CAM_BACK',
            'CAM_BACK_LEFT',
            'CAM_BACK_RIGHT',
        ]
        
        for cam in camera_types:
            cam_token = sample['data'][cam]
            cam_path, _, cam_intrinsic = nusc.get_sample_data(cam_token)
            cam_info = obtain_sensor2top(nusc, cam_token, l2e_t, l2e_r_mat, e2g_t, e2g_r_mat, cam)
            cam_info.update(cam_intrinsic=cam_intrinsic)
            info['cams'].update({cam: cam_info})
        
        # obtain 5 radar's information per frame
        radar_types = [
            'RADAR_FRONT',
            'RADAR_FRONT_RIGHT',
            'RADAR_FRONT_LEFT',
            'RADAR_BACK_LEFT',
            'RADAR_BACK_RIGHT',
        ]
        sweeps = []
        RF_sweeps = []
        RFR_sweeps = []
        RFL_sweeps = []
        RBL_sweeps = []
        RBR_sweeps = []
        
        for radar in radar_types:
            sd_rec = nusc.get('sample_data', sample['data'][radar])
            if radar == 'RADAR_FRONT':
                sweeps_bucket = RF_sweeps
            elif radar == 'RADAR_FRONT_RIGHT':
                sweeps_bucket = RFR_sweeps
            elif radar == 'RADAR_FRONT_LEFT':
                sweeps_bucket = RFL_sweeps
            elif radar == 'RADAR_BACK_LEFT':
                sweeps_bucket = RBL_sweeps
            elif radar == 'RADAR_BACK_RIGHT':
                sweeps_bucket = RBR_sweeps
            while len(sweeps_bucket) < max_sweeps:
                if not sd_rec['prev'] == '':
                    sd_rec =nusc.get('sample_data', sd_rec['prev'])
                    sweeps.append(sd_rec['token'])
                    sweeps_bucket.append(sd_rec['token'])
                else:
                    break
        info['radar_sweeps'] = sweeps
        
        # obtain sweeps for a single key-frame
        sd_rec = nusc.get('sample_data', sample['data']['LIDAR_TOP'])
        sweeps = []
        while len(sweeps) < max_sweeps:
            if not sd_rec['prev'] == '':
                sweep = obtain_sensor2top(nusc, sd_rec['prev'], l2e_t,
                                          l2e_r_mat, e2g_t, e2g_r_mat, 'lidar')
                sweeps.append(sweep)
                sd_rec = nusc.get('sample_data', sd_rec['prev'])
            else:
                break
        info['sweeps'] = sweeps
        
        # obtain annotation
        if not test:
            annotations = [
                nusc.get('sample_annotation', token)
                for token in sample['anns']
            ]
            locs = np.array([b.center for b in boxes]).reshape(-1, 3)
            dims = np.array([b.wlh for b in boxes]).reshape(-1, 3)
            rots = np.array([b.orientation.yaw_pitch_roll[0]
                             for b in boxes]).reshape(-1, 1)
            velocity = np.array(
                [nusc.box_velocity(token)[:2] for token in sample['anns']])
            valid_flag = np.array(
                [(anno['num_lidar_pts'] + anno['num_radar_pts']) > 0
                 for anno in annotations],
                dtype=bool).reshape(-1)
            # convert velo from global to lidar
            for i in range(len(boxes)):
                velo = np.array([*velocity[i], 0.0])
                velo = velo @ np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                    l2e_r_mat).T
                velocity[i] = velo[:2]

            names = [b.name for b in boxes]
            
            for i in range(len(names)):
                if names[i] in NuScenesRadarDataset.NameMapping:
                    names[i] = NuScenesRadarDataset.NameMapping[names[i]]
            names = np.array(names)
            # we need to convert rot to SECOND format.
            gt_boxes = np.concatenate([locs, dims, -rots - np.pi / 2], axis=1)
            
            # for make new multisweep BEV GT boxes
            ref_boxes = copy.deepcopy(gt_boxes)
            ref_boxes[:,2]=0
            ref_boxes[:,5]=10
            points = radar_npy
            # To find the gt box that include multisweep point
            # indices = points_count_rbbox(points,ref_boxes)
            indices = points_count_rbbox_second(points,ref_boxes)
            mask = np.array([k for k,v in enumerate(indices) if v >0])
            radar_points = np.array([v for k,v in enumerate(indices) if v >0])
            if mask.dtype != 'int' and mask.dtype != 'bool':
                mask =  np.array([a["num_lidar_pts"]==-1 for a in annotations])

            assert len(gt_boxes) == len(annotations), f'{len(gt_boxes)}, {len(annotations)}'
            try:
                info["gt_boxes"] = gt_boxes
                info["filtered_gt_boxes"] = gt_boxes[mask]
                info["gt_names"] = names
                info["filtered_gt_names"] = names[mask]
                info["gt_velocity"] = velocity.reshape(-1, 2)
                info["filtered_gt_velocity"] = velocity.reshape(-1, 2)[mask]
                info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
                info["filtered_num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])[mask]
                info["num_radar_pts"] = np.array([a['num_radar_pts'] for a in annotations])
                info["filtered_num_radar_pts"] = radar_points
                info['valid_flag'] = valid_flag
                
            except:
                new_mask = radar_points>0
                info["gt_boxes"] = gt_boxes
                info["filtered_gt_boxes"] = gt_boxes[new_mask]
                info["gt_names"] = names
                info["filtered_gt_names"] = names[new_mask]
                info["gt_velocity"] = velocity.reshape(-1, 2)
                info["filtered_gt_velocity"] = velocity.reshape(-1, 2)[new_mask]
                info["num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])
                info["filtered_num_lidar_pts"] = np.array([a["num_lidar_pts"] for a in annotations])[new_mask]
                info["num_radar_pts"] = np.array([a['num_radar_pts'] for a in annotations])
                info["filtered_num_radar_pts"] = radar_points
                info['valid_flag'] = valid_flag

        if sample['scene_token'] in train_scenes:
            train_nusc_infos.append(info)
        else:
            val_nusc_infos.append(info)

    return train_nusc_infos, val_nusc_infos

def make_multisweep_radar_data_official(nusc, nusc_can, sample, radar_version, max_sweeps, root_path):
    from nuscenes_radar_devkit.utils.data_classes import RadarPointCloud
    from pyquaternion import Quaternion
    from nuscenes_radar_devkit.utils.geometry_utils import transform_matrix
    all_pc = np.zeros((0, 18))

    # lidar information at sample time
    lidar_token = sample["data"]["LIDAR_TOP"]
    ref_sd_rec = nusc.get('sample_data', sample['data']["LIDAR_TOP"])
    ref_cs_record = nusc.get('calibrated_sensor',ref_sd_rec['calibrated_sensor_token'])
    ref_pose_record = nusc.get('ego_pose', ref_sd_rec['ego_pose_token'])
    
    # # For using can_bus data
    # scene_rec = nusc.get('scene',sample['scene_token'])
    # scene_name = scene_rec['name']
    # scene_pose = nusc_can.get_messages(scene_name,'pose')
    # # utimes, vel_data and yaw_rate are all data types in NumPy array format. 
    # utimes = np.array([m['utime'] for m in scene_pose])
    # vel_data = np.array([m['vel'][0] for m in scene_pose])
    # yaw_rate = np.array([m['rotation_rate'][2] for m in scene_pose])
    # orientations = np.array([m['orientation'] for m in scene_pose])
    # # import pdb; pdb.set_trace()

    # lidar transformation matrix information
    l2e_r = ref_cs_record['rotation']
    l2e_t = ref_cs_record['translation']
    e2g_r = ref_pose_record['rotation']
    e2g_t = ref_pose_record['translation']
    l2e_r_mat = Quaternion(l2e_r).rotation_matrix
    e2g_r_mat = Quaternion(e2g_r).rotation_matrix
    
    # Directions of radar data
    radar_channels = ['RADAR_FRONT','RADAR_FRONT_RIGHT','RADAR_FRONT_LEFT','RADAR_BACK_LEFT','RADAR_BACK_RIGHT']

    for radar_channel in radar_channels:
        # each radar data information
        radar_data_token = sample['data'][radar_channel]
        radar_sample_data = nusc.get('sample_data',radar_data_token)
        ref_radar_time = radar_sample_data['timestamp']

        # At each sweep, We need to merge data from 5 directions
        for _ in range(max_sweeps):
            radar_path = osp.join(root_path,radar_sample_data['filename'])
            radar_cs_record = nusc.get('calibrated_sensor',radar_sample_data['calibrated_sensor_token'])
            radar_pose_record = nusc.get('ego_pose', radar_sample_data['ego_pose_token'])
            time_gap = (ref_radar_time-radar_sample_data['timestamp'])*1e-6

            # #Use can_bus data to get ego_vehicle_speed, We get the Ego vehicle speed that fits the radar sensor data
            # time_idx = np.argmin(np.abs(utimes-ref_radar_time))
            # current_from_car = transform_matrix([0,0,0], Quaternion(radar_cs_record['rotation']),inverse=True)[0:2,0:2]
            # # vel_data = scalar data, vehicle_v = vector that coordinate is ego, final_v = vector that coordinate is radar sensor
            # vel_data[time_idx] = np.around(vel_data[time_idx],10)
            # orientation_ = orientations[time_idx]
            # # vehicle_v = np.array((vel_data[time_idx]*np.cos(yaw_rate[time_idx]),vel_data[time_idx]*np.sin(yaw_rate[time_idx])))
            # vehicle_v = np.array([vel_data[time_idx],0])       
            # final_v = vehicle_v@current_from_car.T 

            # get radar data and do some filtering
            current_pc = RadarPointCloud.from_file(radar_path).points.T
            current_pc[:,12] = time_gap
            current_pc = point_filtering(current_pc, filter_version='Valid_filter')
            if current_pc == 'No_point':
                continue
            # version2 == Rel / absolute / absolute + Ego vehicle
            # if radar_version == 'vel_abs':
            #     current_pc[:,:2] += (current_pc[:,8:10] - final_v[0:2])*time_gap
            # elif radar_version == 'vel_abs_V2':
            #     current_pc[:,:2] += (current_pc[:,8:10] + final_v[0:2])*time_gap
            # else:
            #     current_pc[:,:2] += current_pc[:,8:10]*time_gap
            current_pc[:,:2] += current_pc[:,8:10]*time_gap

            # ## make accurate Absolute velocity
            # if radar_version == 'vel_relego':
            #     current_pc[:,6:8] = current_pc[:,8:10] + final_v[0:2]
            # elif radar_version in ['vel_rel','vel_relCego','vel_abs', 'vel_abs_V2']:
            #     current_pc[:,6:8] = current_pc[:,8:10]
            # elif radar_version in ['vel_relCabsCego']:
            #     current_pc[:,16:18] = current_pc[:,8:10] + final_v[0:2]
            # else:
            #     raise NotImplementedError 
            current_pc[:,6:8] = current_pc[:,8:10]
            #radar_point to lidar top coordinate
            r2e_r_s = radar_cs_record['rotation']
            r2e_t_s = radar_cs_record['translation']
            e2g_r_s = radar_pose_record['rotation']
            e2g_t_s = radar_pose_record['translation']
            r2e_r_s_mat = Quaternion(r2e_r_s).rotation_matrix
            e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix

            R = (r2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            T = (r2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
                np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
            T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(
                l2e_r_mat).T) + l2e_t @ np.linalg.inv(l2e_r_mat).T
            
            # sweep['sensor2lidar_rotation'] = R.T
            # sweep['sensor2lidar_translation'] = T
            
            current_pc[:,:3] = current_pc[:,:3] @ R
            current_pc[:,[6,7]] = current_pc[:,[6,7]] @ R[:2,:2]
            if radar_version == 'vel_relCabsCego':
                current_pc[:,[16,17]] = current_pc[:,[16,17]] @ R[:2,:2]
            current_pc[:,:3] += T            
            # current_pc[:,[8,9]] = final_v[0:2] @ R[:2,:2]
            all_pc = np.vstack((all_pc, current_pc))

            if radar_sample_data['prev'] == '':
                break
            else:
                radar_sample_data = nusc.get('sample_data', radar_sample_data['prev'])
    
    filter_version = 'Valid_filter'
    if radar_version in ['vel_rel','vel_relego','vel_abs', 'vel_abs_V2']:
        if filter_version == 'No_filter_V2':
            use_idx = [0,1,2,5,6,7,3,12]
        else:
            use_idx = [0,1,2,5,6,7,12]
    elif radar_version == 'vel_relCego':
        if filter_version == 'No_filter_V2':
            use_idx = [0,1,2,5,6,7,8,9,3,12]
        else:
            use_idx = [0,1,2,5,6,7,8,9,12]
    ## New radar!! 0805
    elif radar_version == 'vel_relCabsCego':
        use_idx = [0,1,2,5,6,7,16,17,8,9,12]
    else:
        raise NotImplementedError
    all_pc = all_pc[:,use_idx]
    return all_pc

def point_filtering(pc, filter_version):
    if filter_version == 'No_filter_V1':
        return pc
    elif filter_version in ['Valid_filter','Full_filter','No_filter_V2']:
        ambig = pc[:,11]
        invalid = pc[:,14]
        dyn = pc[:,3]
        ambig_list = np.where(ambig==3)[0]
        if filter_version in ['Valid_filter','No_filter_V2']:
            valid_criteria = [0,4,8,9,10,11,12,15,16]
            invalid_list = np.array([idx for idx,point in enumerate(invalid) if point in valid_criteria])
            intersect = np.intersect1d(ambig_list,invalid_list)
            if filter_version == 'No_filter_V2':
                relative = np.setdiff1d(np.arange(len(pc)),intersect)
        else:
            valid_list = np.where(valid==0)[0]
            dyn_criteria = list(range(7))
            dyn_list = np.array([idx for idx,point in enumerate(dyn) if point in dyn_criteria])
            intersect = np.intersect1d(ambig_list,dyn_list,valid_list)
        if len(intersect) == 0:
            return 'No_point'
        else:
            if filter_version == 'No_filter_V2':
                pc[intersect,3] == 1 # valid point
                pc[relative,3] == 0 # invalid point
                return pc
            else:
                return pc[intersect,:]
    else:
        raise ValueError

def obtain_sensor2top(nusc,
                      sensor_token,
                      l2e_t,
                      l2e_r_mat,
                      e2g_t,
                      e2g_r_mat,
                      sensor_type='lidar'):
    """Obtain the info with RT matric from general sensor to Top LiDAR.

    Args:
        nusc (class): Dataset class in the nuScenes dataset.
        sensor_token (str): Sample data token corresponding to the
            specific sensor type.
        l2e_t (np.ndarray): Translation from lidar to ego in shape (1, 3).
        l2e_r_mat (np.ndarray): Rotation matrix from lidar to ego
            in shape (3, 3).
        e2g_t (np.ndarray): Translation from ego to global in shape (1, 3).
        e2g_r_mat (np.ndarray): Rotation matrix from ego to global
            in shape (3, 3).
        sensor_type (str): Sensor to calibrate. Default: 'lidar'.

    Returns:
        sweep (dict): Sweep information after transformation.
    """
    sd_rec = nusc.get('sample_data', sensor_token)
    cs_record = nusc.get('calibrated_sensor',
                         sd_rec['calibrated_sensor_token'])
    pose_record = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    data_path = str(nusc.get_sample_data_path(sd_rec['token']))
    if os.getcwd() in data_path:  # path from lyftdataset is absolute path
        data_path = data_path.split(f'{os.getcwd()}/')[-1]  # relative path
    sweep = {
        'data_path': data_path,
        'type': sensor_type,
        'sample_data_token': sd_rec['token'],
        'sensor2ego_translation': cs_record['translation'],
        'sensor2ego_rotation': cs_record['rotation'],
        'ego2global_translation': pose_record['translation'],
        'ego2global_rotation': pose_record['rotation'],
        'timestamp': sd_rec['timestamp']
    }
    l2e_r_s = sweep['sensor2ego_rotation']
    l2e_t_s = sweep['sensor2ego_translation']
    e2g_r_s = sweep['ego2global_rotation']
    e2g_t_s = sweep['ego2global_translation']

    # obtain the RT from sensor to Top LiDAR
    # sweep->ego->global->ego'->lidar
    l2e_r_s_mat = Quaternion(l2e_r_s).rotation_matrix
    e2g_r_s_mat = Quaternion(e2g_r_s).rotation_matrix
    R = (l2e_r_s_mat.T @ e2g_r_s_mat.T) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T = (l2e_t_s @ e2g_r_s_mat.T + e2g_t_s) @ (
        np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T)
    T -= e2g_t @ (np.linalg.inv(e2g_r_mat).T @ np.linalg.inv(l2e_r_mat).T
                  ) + l2e_t @ np.linalg.inv(l2e_r_mat).T
    sweep['sensor2lidar_rotation'] = R.T  # points @ R.T + T
    sweep['sensor2lidar_translation'] = T
    return sweep


def export_2d_annotation(root_path, info_path, version, mono3d=True):
    """Export 2d annotation from the info file and raw data.

    Args:
        root_path (str): Root path of the raw data.
        info_path (str): Path of the info file.
        version (str): Dataset version.
        mono3d (bool): Whether to export mono3d annotation. Default: True.
    """
    # get bbox annotations for camera
    camera_types = [
        'CAM_FRONT',
        'CAM_FRONT_RIGHT',
        'CAM_FRONT_LEFT',
        'CAM_BACK',
        'CAM_BACK_LEFT',
        'CAM_BACK_RIGHT',
    ]
    nusc_infos = mmcv.load(info_path)['infos']
    nusc = NuScenes(version=version, dataroot=root_path, verbose=True)
    # info_2d_list = []
    cat2Ids = [
        dict(id=nus_categories.index(cat_name), name=cat_name)
        for cat_name in nus_categories
    ]
    coco_ann_id = 0
    coco_2d_dict = dict(annotations=[], images=[], categories=cat2Ids)
    for info in mmcv.track_iter_progress(nusc_infos):
        for cam in camera_types:
            cam_info = info['cams'][cam]
            coco_infos = get_2d_boxes(
                nusc,
                cam_info['sample_data_token'],
                visibilities=['', '1', '2', '3', '4'],
                mono3d=mono3d)
            (height, width, _) = mmcv.imread(cam_info['data_path']).shape
            coco_2d_dict['images'].append(
                dict(
                    file_name=cam_info['data_path'].split('data/nuscenes/')
                    [-1],
                    id=cam_info['sample_data_token'],
                    token=info['token'],
                    cam2ego_rotation=cam_info['sensor2ego_rotation'],
                    cam2ego_translation=cam_info['sensor2ego_translation'],
                    ego2global_rotation=info['ego2global_rotation'],
                    ego2global_translation=info['ego2global_translation'],
                    cam_intrinsic=cam_info['cam_intrinsic'],
                    width=width,
                    height=height))
            for coco_info in coco_infos:
                if coco_info is None:
                    continue
                # add an empty key for coco format
                coco_info['segmentation'] = []
                coco_info['id'] = coco_ann_id
                coco_2d_dict['annotations'].append(coco_info)
                coco_ann_id += 1
    if mono3d:
        json_prefix = f'{info_path[:-4]}_mono3d'
    else:
        json_prefix = f'{info_path[:-4]}'
    mmcv.dump(coco_2d_dict, f'{json_prefix}.coco.json')


def get_2d_boxes(nusc,
                 sample_data_token: str,
                 visibilities: List[str],
                 mono3d=True):
    """Get the 2D annotation records for a given `sample_data_token`.

    Args:
        sample_data_token (str): Sample data token belonging to a camera \
            keyframe.
        visibilities (list[str]): Visibility filter.
        mono3d (bool): Whether to get boxes with mono3d annotation.

    Return:
        list[dict]: List of 2D annotation record that belongs to the input
            `sample_data_token`.
    """

    # Get the sample data and the sample corresponding to that sample data.
    sd_rec = nusc.get('sample_data', sample_data_token)

    assert sd_rec[
        'sensor_modality'] == 'camera', 'Error: get_2d_boxes only works' \
        ' for camera sample_data!'
    if not sd_rec['is_key_frame']:
        raise ValueError(
            'The 2D re-projections are available only for keyframes.')

    s_rec = nusc.get('sample', sd_rec['sample_token'])

    # Get the calibrated sensor and ego pose
    # record to get the transformation matrices.
    cs_rec = nusc.get('calibrated_sensor', sd_rec['calibrated_sensor_token'])
    pose_rec = nusc.get('ego_pose', sd_rec['ego_pose_token'])
    camera_intrinsic = np.array(cs_rec['camera_intrinsic'])

    # Get all the annotation with the specified visibilties.
    ann_recs = [
        nusc.get('sample_annotation', token) for token in s_rec['anns']
    ]
    ann_recs = [
        ann_rec for ann_rec in ann_recs
        if (ann_rec['visibility_token'] in visibilities)
    ]

    repro_recs = []

    for ann_rec in ann_recs:
        # Augment sample_annotation with token information.
        ann_rec['sample_annotation_token'] = ann_rec['token']
        ann_rec['sample_data_token'] = sample_data_token

        # Get the box in global coordinates.
        box = nusc.get_box(ann_rec['token'])

        # Move them to the ego-pose frame.
        box.translate(-np.array(pose_rec['translation']))
        box.rotate(Quaternion(pose_rec['rotation']).inverse)

        # Move them to the calibrated sensor frame.
        box.translate(-np.array(cs_rec['translation']))
        box.rotate(Quaternion(cs_rec['rotation']).inverse)

        # Filter out the corners that are not in front of the calibrated
        # sensor.
        corners_3d = box.corners()
        in_front = np.argwhere(corners_3d[2, :] > 0).flatten()
        corners_3d = corners_3d[:, in_front]

        # Project 3d box to 2d.
        corner_coords = view_points(corners_3d, camera_intrinsic,
                                    True).T[:, :2].tolist()

        # Keep only corners that fall within the image.
        final_coords = post_process_coords(corner_coords)

        # Skip if the convex hull of the re-projected corners
        # does not intersect the image canvas.
        if final_coords is None:
            continue
        else:
            min_x, min_y, max_x, max_y = final_coords

        # Generate dictionary record to be included in the .json file.
        repro_rec = generate_record(ann_rec, min_x, min_y, max_x, max_y,
                                    sample_data_token, sd_rec['filename'])

        # If mono3d=True, add 3D annotations in camera coordinates
        if mono3d and (repro_rec is not None):
            loc = box.center.tolist()

            dim = box.wlh
            dim[[0, 1, 2]] = dim[[1, 2, 0]]  # convert wlh to our lhw
            dim = dim.tolist()

            rot = box.orientation.yaw_pitch_roll[0]
            rot = [-rot]  # convert the rot to our cam coordinate

            global_velo2d = nusc.box_velocity(box.token)[:2]
            global_velo3d = np.array([*global_velo2d, 0.0])
            e2g_r_mat = Quaternion(pose_rec['rotation']).rotation_matrix
            c2e_r_mat = Quaternion(cs_rec['rotation']).rotation_matrix
            cam_velo3d = global_velo3d @ np.linalg.inv(
                e2g_r_mat).T @ np.linalg.inv(c2e_r_mat).T
            velo = cam_velo3d[0::2].tolist()

            repro_rec['bbox_cam3d'] = loc + dim + rot
            repro_rec['velo_cam3d'] = velo

            center3d = np.array(loc).reshape([1, 3])
            center2d = points_cam2img(
                center3d, camera_intrinsic, with_depth=True)
            repro_rec['center2d'] = center2d.squeeze().tolist()
            # normalized center2D + depth
            # if samples with depth < 0 will be removed
            if repro_rec['center2d'][2] <= 0:
                continue

            ann_token = nusc.get('sample_annotation',
                                 box.token)['attribute_tokens']
            if len(ann_token) == 0:
                attr_name = 'None'
            else:
                attr_name = nusc.get('attribute', ann_token[0])['name']
            attr_id = nus_attributes.index(attr_name)
            repro_rec['attribute_name'] = attr_name
            repro_rec['attribute_id'] = attr_id

        repro_recs.append(repro_rec)

    return repro_recs


def post_process_coords(
    corner_coords: List, imsize: Tuple[int, int] = (1600, 900)
) -> Union[Tuple[float, float, float, float], None]:
    """Get the intersection of the convex hull of the reprojected bbox corners
    and the image canvas, return None if no intersection.

    Args:
        corner_coords (list[int]): Corner coordinates of reprojected
            bounding box.
        imsize (tuple[int]): Size of the image canvas.

    Return:
        tuple [float]: Intersection of the convex hull of the 2D box
            corners and the image canvas.
    """
    polygon_from_2d_box = MultiPoint(corner_coords).convex_hull
    img_canvas = box(0, 0, imsize[0], imsize[1])

    if polygon_from_2d_box.intersects(img_canvas):
        img_intersection = polygon_from_2d_box.intersection(img_canvas)
        intersection_coords = np.array(
            [coord for coord in img_intersection.exterior.coords])

        min_x = min(intersection_coords[:, 0])
        min_y = min(intersection_coords[:, 1])
        max_x = max(intersection_coords[:, 0])
        max_y = max(intersection_coords[:, 1])

        return min_x, min_y, max_x, max_y
    else:
        return None


def generate_record(ann_rec: dict, x1: float, y1: float, x2: float, y2: float,
                    sample_data_token: str, filename: str) -> OrderedDict:
    """Generate one 2D annotation record given various informations on top of
    the 2D bounding box coordinates.

    Args:
        ann_rec (dict): Original 3d annotation record.
        x1 (float): Minimum value of the x coordinate.
        y1 (float): Minimum value of the y coordinate.
        x2 (float): Maximum value of the x coordinate.
        y2 (float): Maximum value of the y coordinate.
        sample_data_token (str): Sample data token.
        filename (str):The corresponding image file where the annotation
            is present.

    Returns:
        dict: A sample 2D annotation record.
            - file_name (str): flie name
            - image_id (str): sample data token
            - area (float): 2d box area
            - category_name (str): category name
            - category_id (int): category id
            - bbox (list[float]): left x, top y, dx, dy of 2d box
            - iscrowd (int): whether the area is crowd
    """
    repro_rec = OrderedDict()
    repro_rec['sample_data_token'] = sample_data_token
    coco_rec = dict()

    relevant_keys = [
        'attribute_tokens',
        'category_name',
        'instance_token',
        'next',
        'num_lidar_pts',
        'num_radar_pts',
        'prev',
        'sample_annotation_token',
        'sample_data_token',
        'visibility_token',
    ]

    for key, value in ann_rec.items():
        if key in relevant_keys:
            repro_rec[key] = value

    repro_rec['bbox_corners'] = [x1, y1, x2, y2]
    repro_rec['filename'] = filename

    coco_rec['file_name'] = filename
    coco_rec['image_id'] = sample_data_token
    coco_rec['area'] = (y2 - y1) * (x2 - x1)

    if repro_rec['category_name'] not in NuScenesDataset.NameMapping:
        return None
    cat_name = NuScenesDataset.NameMapping[repro_rec['category_name']]
    coco_rec['category_name'] = cat_name
    coco_rec['category_id'] = nus_categories.index(cat_name)
    coco_rec['bbox'] = [x1, y1, x2 - x1, y2 - y1]
    coco_rec['iscrowd'] = 0

    return coco_rec
