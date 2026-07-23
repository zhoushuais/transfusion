"""Stage-2 original query-level camera-LiDAR TransFusion on KITTI."""

_base_ = ['./transfusion_l_kitti.py']

input_modality = dict(use_lidar=True, use_camera=True)
image_scale = (1280, 384)
point_cloud_range = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]
image_meta_keys = (
    'sample_idx',
    'img_path',
    'lidar_path',
    'ori_shape',
    'img_shape',
    'pad_shape',
    'scale_factor',
    'box_type_3d',
    'cam2img',
    'lidar2cam',
    'lidar2img',
    'homography_matrix',
)

model = dict(
    fuse_img=True,
    freeze_img=True,
    freeze_lidar=True,
    img_feature_level=0,
    data_preprocessor=dict(
        mean=[123.675, 116.28, 103.53],
        std=[58.395, 57.12, 57.375],
        bgr_to_rgb=True,
        pad_size_divisor=32,
    ),
    img_backbone=dict(
        type='mmdet.ResNet',
        depth=50,
        num_stages=4,
        out_indices=(0, 1, 2, 3),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
    ),
    img_neck=dict(
        type='mmdet.FPN',
        in_channels=[256, 512, 1024, 2048],
        out_channels=256,
        num_outs=5,
    ),
    bbox_head=dict(
        fuse_img=True,
        num_views=1,
        in_channels_img=256,
        out_size_factor_img=4,
    ),
)

train_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
    ),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True),
    dict(type='ValidateKittiCalibration'),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='PointShuffle'),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img', 'gt_bboxes_3d', 'gt_labels_3d'],
        meta_keys=image_meta_keys,
    ),
]

test_pipeline = [
    dict(type='LoadImageFromFileMono3D'),
    dict(
        type='LoadPointsFromFile',
        coord_type='LIDAR',
        load_dim=4,
        use_dim=4,
    ),
    dict(type='ValidateKittiCalibration'),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='PointsRangeFilter', point_cloud_range=point_cloud_range),
    dict(
        type='Pack3DDetInputs',
        keys=['points', 'img'],
        meta_keys=image_meta_keys,
    ),
]

train_dataloader = dict(
    batch_size=2,
    dataset=dict(
        dataset=dict(
            pipeline=train_pipeline,
            modality=input_modality,
            data_prefix=dict(
                pts='training/velodyne_reduced', img='training/image_2'),
        )),
)
val_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        modality=input_modality,
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
    ))
test_dataloader = dict(
    dataset=dict(
        pipeline=test_pipeline,
        modality=input_modality,
        data_prefix=dict(
            pts='training/velodyne_reduced', img='training/image_2'),
    ))

lr = 1e-4
optim_wrapper = dict(
    _delete_=True,
    type='OptimWrapper',
    optimizer=dict(type='AdamW', lr=lr, weight_decay=0.01),
    clip_grad=dict(max_norm=0.1, norm_type=2),
    accumulative_counts=8,
)
param_scheduler = [
    dict(
        type='OneCycleLR',
        total_steps=6,
        by_epoch=True,
        eta_max=lr * 10,
        pct_start=0.4,
        div_factor=10.0,
        final_div_factor=10000.0,
        convert_to_iter_based=True,
    )
]
train_cfg = dict(by_epoch=True, max_epochs=6, val_interval=1)
auto_scale_lr = dict(enable=False, base_batch_size=16)
