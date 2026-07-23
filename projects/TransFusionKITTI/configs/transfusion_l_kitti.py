"""Stage-1 LiDAR-only TransFusion configuration for KITTI."""

_base_ = [
    '../../../configs/_base_/datasets/kitti-3d-3class.py',
    '../../../configs/_base_/schedules/cyclic-40e.py',
    '../../../configs/_base_/default_runtime.py',
]

custom_imports = dict(
    imports=['projects.TransFusionKITTI.transfusion_kitti'],
    allow_failed_imports=False,
)

class_names = ('Pedestrian', 'Cyclist', 'Car')
metainfo = dict(classes=class_names)
point_cloud_range = [0.0, -40.0, -3.0, 70.4, 40.0, 1.0]
voxel_size = [0.05, 0.05, 0.1]
grid_size = [1408, 1600, 40]
out_size_factor = 8

model = dict(
    type='TransFusionKITTIDetector',
    fuse_img=False,
    data_preprocessor=dict(
        type='Det3DDataPreprocessor',
        voxel=True,
        voxel_layer=dict(
            max_num_points=5,
            point_cloud_range=point_cloud_range,
            voxel_size=voxel_size,
            max_voxels=(16000, 40000),
        ),
    ),
    voxel_encoder=dict(type='HardSimpleVFE'),
    middle_encoder=dict(
        type='SparseEncoder',
        in_channels=4,
        sparse_shape=[41, 1600, 1408],
        order=('conv', 'norm', 'act'),
    ),
    backbone=dict(
        type='SECOND',
        in_channels=256,
        layer_nums=[5, 5],
        layer_strides=[1, 2],
        out_channels=[128, 256],
    ),
    neck=dict(
        type='SECONDFPN',
        in_channels=[128, 256],
        upsample_strides=[1, 2],
        out_channels=[256, 256],
    ),
    bbox_head=dict(
        type='TransFusionKITTIHead',
        fuse_img=False,
        num_proposals=200,
        auxiliary=True,
        in_channels=512,
        hidden_channel=128,
        num_classes=3,
        nms_kernel_size=3,
        bn_momentum=0.1,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransFusionKITTITransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            cross_attn_cfg=dict(embed_dims=128, num_heads=8, dropout=0.1),
            ffn_cfg=dict(
                embed_dims=128,
                feedforward_channels=256,
                num_fcs=2,
                ffn_drop=0.1,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=128),
        ),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]),
        bbox_coder=dict(
            type='TransFusionKITTIBBoxCoder',
            pc_range=point_cloud_range[:2],
            post_center_range=point_cloud_range,
            score_threshold=0.0,
            out_size_factor=out_size_factor,
            voxel_size=voxel_size[:2],
            code_size=8,
        ),
        loss_cls=dict(
            type='mmdet.FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            reduction='mean',
            loss_weight=1.0,
        ),
        loss_heatmap=dict(
            type='mmdet.GaussianFocalLoss',
            reduction='mean',
            loss_weight=1.0,
        ),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
    ),
    train_cfg=dict(
        dataset='KITTI',
        point_cloud_range=point_cloud_range,
        grid_size=grid_size,
        voxel_size=voxel_size,
        out_size_factor=out_size_factor,
        gaussian_overlap=0.1,
        min_radius=2,
        pos_weight=-1,
        code_weights=[1.0] * 8,
        assigner=dict(
            type='TransFusionKITTIHungarianAssigner3D',
            iou_calculator=dict(type='BboxOverlaps3D', coordinate='lidar'),
            cls_cost=dict(
                type='mmdet.FocalLossCost',
                gamma=2.0,
                alpha=0.25,
                weight=0.15,
            ),
            reg_cost=dict(type='TransFusionKITTIBBoxBEVL1Cost', weight=0.25),
            iou_cost=dict(type='TransFusionKITTIIoU3DCost', weight=0.25),
        ),
    ),
    test_cfg=dict(
        dataset='KITTI',
        grid_size=grid_size,
        out_size_factor=out_size_factor,
        voxel_size=voxel_size[:2],
        pc_range=point_cloud_range[:2],
        nms_type=None,
    ),
)

train_dataloader = dict(dataset=dict(dataset=dict(metainfo=metainfo)))
val_dataloader = dict(dataset=dict(metainfo=metainfo))
test_dataloader = dict(dataset=dict(metainfo=metainfo))

optim_wrapper = dict(accumulative_counts=8)

default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best=(
            'Kitti metric/pred_instances_3d/KITTI/'
            'Overall_3D_AP40_moderate'),
        rule='greater',
    ))
