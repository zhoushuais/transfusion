"""Stage-0 KITTI 2D pretraining for the TransFusion image backbone/FPN."""

_base_ = 'mmdet::faster_rcnn/faster-rcnn_r50_fpn_1x_coco.py'

default_scope = 'mmdet'
class_names = ('Pedestrian', 'Cyclist', 'Car')
metainfo = dict(classes=class_names)
data_root = 'data/kitti/'
image_scale = (1280, 384)

model = dict(roi_head=dict(bbox_head=dict(num_classes=3)))

train_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='RandomFlip', prob=0.5),
    dict(type='PackDetInputs'),
]
test_pipeline = [
    dict(type='LoadImageFromFile'),
    dict(type='Resize', scale=image_scale, keep_ratio=True),
    dict(type='LoadAnnotations', with_bbox=True),
    dict(
        type='PackDetInputs',
        meta_keys=('img_id', 'img_path', 'ori_shape', 'img_shape',
                   'scale_factor'),
    ),
]

train_dataloader = dict(
    batch_size=4,
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/kitti_2d_train.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        filter_cfg=dict(filter_empty_gt=True, min_size=1),
        pipeline=train_pipeline,
    ),
)
val_dataloader = dict(
    dataset=dict(
        type='CocoDataset',
        data_root=data_root,
        ann_file='annotations/kitti_2d_val.json',
        data_prefix=dict(img=''),
        metainfo=metainfo,
        test_mode=True,
        pipeline=test_pipeline,
    ))
test_dataloader = val_dataloader

val_evaluator = dict(
    type='CocoMetric',
    ann_file=data_root + 'annotations/kitti_2d_val.json',
    metric='bbox',
)
test_evaluator = val_evaluator

load_from = ('https://download.openmmlab.com/mmdetection/v2.0/faster_rcnn/'
             'faster_rcnn_r50_fpn_1x_coco/'
             'faster_rcnn_r50_fpn_1x_coco_20200130-047c8118.pth')
default_hooks = dict(
    checkpoint=dict(
        type='CheckpointHook',
        interval=1,
        save_best='coco/bbox_mAP',
        rule='greater',
    ))
