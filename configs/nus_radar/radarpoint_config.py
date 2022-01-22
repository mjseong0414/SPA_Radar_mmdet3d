from ensurepip import version


point_cloud_range = [-50, -50, -5, 50, 50, 3]
class_names = ['car','pedestrian' , 'bus' , 'truck', 'motorcycle', 'trailer', 'construction_vehicle', 'bicycle']
unified_class_names = ['car','pedestrian' , 'bus' , 'truck', 'motorcycle']
dataset_type = 'NuScenesRadarDataset'
data_root = '/mnt/sda/minjae/nuscenes/'
home_root = 'data/nuscenes/'
input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=True,
    use_map=False,
    use_external=False)

img_norm_cfg = dict(
    mean=[103.530, 116.280, 123.675], std=[1.0, 1.0, 1.0], to_rgb=False)

file_client_args = dict(backend='disk')
train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadRadarPointsFromFile',
        coord_type='LIDAR',
        load_dim=18,
        # use_dim=6,
        file_client_args=file_client_args,
        load_radar_points=True),
    dict(
        type='LoadRadarPointsFromMultiSweeps',
        sweeps_num=6, # radar max sweep is 6
        file_client_args=file_client_args,
        data_root=data_root,
        version='v1.0-trainval'), # or v1.0-mini
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='PhotoMetricDistortionMultiViewImage'), # multi view image augmentation
    dict( # point cloud augmentation
        type='GlobalRotScaleTrans',
        rot_range=[-0.3925, 0.3925],
        scale_ratio_range=[0.95, 1.05],
        translation_std=[0, 0, 0]),
    dict(type='RandomFlip3D', flip_ratio_bev_horizontal=0.5), # point cloud augmentation
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=unified_class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='PointSample', num_points=2048),
    dict(type='DefaultFormatBundle3D', class_names=unified_class_names),
    dict(type='Collect3D', keys=['points', 'gt_bboxes_3d', 'gt_labels_3d'])
]
test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(
        type='LoadRadarPointsFromFile',
        coord_type='LIDAR',
        load_dim=18,
        # use_dim=6,
        file_client_args=file_client_args,
        load_radar_points=True),
    dict(
        type='LoadRadarPointsFromMultiSweeps',
        sweeps_num=6,
        file_client_args=file_client_args),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1333, 800),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(
                type='GlobalRotScaleTrans',
                rot_range=[0, 0],
                scale_ratio_range=[1., 1.],
                translation_std=[0, 0, 0]),
            dict(type='RandomFlip3D'),
            dict(
                type='PointsRangeFilter', point_cloud_range=point_cloud_range),
            dict(type='PointSample', num_points=2048),
            dict(
                type='DefaultFormatBundle3D',
                class_names=unified_class_names,
                with_label=False),
            dict(type='Collect3D', keys=['points'])
        ])
]
eval_pipeline = [
    dict(
        type='LoadRadarPointsFromFile',
        coord_type='LIDAR',
        load_dim=18,
        # use_dim=6,
        file_client_args=file_client_args,
        load_radar_points=True),
    dict(
        type='DefaultFormatBundle3D',
        class_names=unified_class_names,
        with_label=False),
    dict(type='Collect3D', keys=['points'])
]