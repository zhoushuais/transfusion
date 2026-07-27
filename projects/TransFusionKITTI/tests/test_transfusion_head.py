# flake8: noqa: E402
import pytest

torch = pytest.importorskip('torch')
pytest.importorskip('mmcv')
pytest.importorskip('mmdet')
from mmengine.config import ConfigDict
from mmengine.registry import init_default_scope
from mmengine.structures import InstanceData

from mmdet3d.structures import LiDARInstance3DBoxes

from projects.TransFusionKITTI.transfusion_kitti.models import \
    TransFusionKITTIHead


def make_head(fuse_img=False, train_cfg=None):
    init_default_scope('mmdet3d')
    return TransFusionKITTIHead(
        fuse_img=fuse_img,
        num_proposals=20,
        auxiliary=True,
        in_channels=512,
        hidden_channel=64,
        num_classes=3,
        nms_kernel_size=3,
        num_decoder_layers=1,
        decoder_layer=dict(
            type='TransFusionKITTITransformerDecoderLayer',
            self_attn_cfg=dict(embed_dims=64, num_heads=8, dropout=0.0),
            cross_attn_cfg=dict(embed_dims=64, num_heads=8, dropout=0.0),
            ffn_cfg=dict(
                embed_dims=64,
                feedforward_channels=128,
                num_fcs=2,
                ffn_drop=0.0,
                act_cfg=dict(type='ReLU', inplace=True),
            ),
            norm_cfg=dict(type='LN'),
            pos_encoding_cfg=dict(input_channel=2, num_pos_feats=64),
        ),
        train_cfg=train_cfg,
        test_cfg=ConfigDict(
            dataset='KITTI',
            grid_size=[160, 160, 40],
            out_size_factor=8,
            voxel_size=[0.05, 0.05],
            pc_range=[0.0, -40.0],
            nms_type=None,
        ),
        common_heads=dict(
            center=[2, 2], height=[1, 2], dim=[3, 2], rot=[2, 2]),
        bbox_coder=dict(
            type='TransFusionKITTIBBoxCoder',
            pc_range=[0.0, -40.0],
            post_center_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
            score_threshold=0.0,
            out_size_factor=8,
            voxel_size=[0.05, 0.05],
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
            type='mmdet.GaussianFocalLoss', reduction='mean', loss_weight=1.0),
        loss_bbox=dict(
            type='mmdet.L1Loss', reduction='mean', loss_weight=0.25),
    )


class ZeroIoU:

    def __call__(self, pred_boxes, gt_boxes):
        return pred_boxes.new_zeros((pred_boxes.shape[0], gt_boxes.shape[0]))


def make_train_cfg():
    return ConfigDict(
        dataset='KITTI',
        point_cloud_range=[0.0, -40.0, -3.0, 70.4, 40.0, 1.0],
        grid_size=[1408, 1600, 40],
        voxel_size=[0.05, 0.05, 0.1],
        out_size_factor=8,
        gaussian_overlap=0.1,
        min_radius=2,
        pos_weight=-1,
        code_weights=[1.0] * 8,
        assigner=ConfigDict(
            type='TransFusionKITTIHungarianAssigner3D',
            iou_calculator=ConfigDict(
                type='BboxOverlaps3D', coordinate='lidar'),
            cls_cost=ConfigDict(
                type='mmdet.FocalLossCost',
                gamma=2.0,
                alpha=0.25,
                weight=0.15),
            reg_cost=ConfigDict(
                type='TransFusionKITTIBBoxBEVL1Cost', weight=0.25),
            iou_cost=ConfigDict(
                type='TransFusionKITTIIoU3DCost', weight=0.25),
        ),
    )


def test_lidar_head_forward_has_no_velocity_and_returns_batch_query_state():
    head = make_head(fuse_img=False).eval()
    outputs = head([torch.randn(2, 512, 20, 20)], [{}, {}])
    pred = outputs[0][0]
    assert 'vel' not in pred
    assert pred['center'].shape == (2, 2, 20)
    assert pred['query_labels'].shape == (2, 20)
    assert not hasattr(head, 'query_labels')


def test_non_square_position_grid_matches_feature_flatten_order():
    head = make_head(fuse_img=False)
    grid = head.create_2D_grid(x_size=2, y_size=3)
    expected = torch.tensor([
        [0.5, 0.5],
        [1.5, 0.5],
        [0.5, 1.5],
        [1.5, 1.5],
        [0.5, 2.5],
        [1.5, 2.5],
    ])[None]
    torch.testing.assert_close(grid, expected)


def test_dense_heatmap_target_uses_xy_order_on_non_square_bev():
    head = make_head(fuse_img=False, train_cfg=make_train_cfg())
    head.bbox_assigner.iou_calculator = ZeroIoU()
    gt_instances = InstanceData(
        bboxes_3d=LiDARInstance3DBoxes(
            torch.tensor([[20.0, -10.0, -1.5, 4.0, 2.0, 1.5, 0.0]])),
        labels_3d=torch.tensor([2], dtype=torch.long),
    )
    predictions = dict(
        heatmap=torch.zeros(1, 3, 20),
        center=torch.zeros(1, 2, 20),
        height=torch.zeros(1, 1, 20),
        dim=torch.zeros(1, 3, 20),
        rot=torch.zeros(1, 2, 20),
    )

    heatmap = head.get_targets_single(gt_instances, predictions, 0)[-1][0]

    # x=20 m maps to column 50; y=-10 m maps to row 75.
    assert heatmap.shape == (3, 200, 176)
    assert heatmap[2, 75, 50] == 1
    assert heatmap[2, 50, 75] < 1


def test_invalid_image_query_falls_back_to_lidar_prediction():
    lidar_prediction = {'center': torch.tensor([[[1.0, 2.0], [3.0, 4.0]]])}
    fusion_prediction = {'center': torch.tensor([[[9.0, 9.0], [9.0, 9.0]]])}
    valid = torch.tensor([[True, False]])
    result = TransFusionKITTIHead._fallback_invalid_queries(
        fusion_prediction, lidar_prediction, valid)
    torch.testing.assert_close(result['center'],
                               torch.tensor([[[9.0, 2.0], [9.0, 4.0]]]))


def test_single_visible_query_runs_image_attention():
    assert TransFusionKITTIHead._should_run_image_attention(
        torch.tensor([True, False]))


def test_all_invalid_query_loss_mask_is_zero_not_nan():
    mask = torch.zeros(2, 20, dtype=torch.bool)
    weights = TransFusionKITTIHead._apply_fusion_mask(torch.ones(2, 20), mask)
    assert weights.sum() == 0
    assert torch.isfinite(weights).all()
