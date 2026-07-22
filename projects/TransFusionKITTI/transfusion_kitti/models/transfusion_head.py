# Adapted from MMDetection3D's maintained TransFusion LiDAR head.
import copy
from typing import List, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer
from mmdet.models.task_modules import (AssignResult, PseudoSampler,
                                       build_assigner, build_bbox_coder,
                                       build_sampler)
from mmdet.models.utils import multi_apply
from mmengine.structures import InstanceData
from torch import nn

from mmdet3d.models import circle_nms, draw_heatmap_gaussian, gaussian_radius
from mmdet3d.models.dense_heads.centerpoint_head import SeparateHead
from mmdet3d.models.layers import nms_bev
from mmdet3d.registry import MODELS
from mmdet3d.structures import xywhr2xyxyr
from .projection import project_lidar_points


def clip_sigmoid(x, eps=1e-4):
    y = torch.clamp(x.sigmoid_(), min=eps, max=1 - eps)
    return y


@MODELS.register_module()
class TransFusionKITTIHead(nn.Module):

    def __init__(
        self,
        fuse_img=False,
        num_views=1,
        in_channels_img=256,
        out_size_factor_img=4,
        num_proposals=128,
        auxiliary=True,
        in_channels=128 * 3,
        hidden_channel=128,
        num_classes=4,
        # config for Transformer
        num_decoder_layers=3,
        decoder_layer=dict(),
        num_heads=8,
        nms_kernel_size=1,
        bn_momentum=0.1,
        # config for FFN
        common_heads=dict(),
        num_heatmap_convs=2,
        conv_cfg=dict(type='Conv1d'),
        norm_cfg=dict(type='BN1d'),
        bias='auto',
        # loss
        loss_cls=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        loss_bbox=dict(type='mmdet.L1Loss', reduction='mean'),
        loss_heatmap=dict(type='mmdet.GaussianFocalLoss', reduction='mean'),
        # others
        train_cfg=None,
        test_cfg=None,
        bbox_coder=None,
    ):
        super().__init__()

        self.num_classes = num_classes
        self.fuse_img = fuse_img
        self.num_views = num_views
        self.out_size_factor_img = out_size_factor_img
        self.num_proposals = num_proposals
        self.auxiliary = auxiliary
        self.in_channels = in_channels
        self.num_heads = num_heads
        self.num_decoder_layers = num_decoder_layers
        self.bn_momentum = bn_momentum
        self.nms_kernel_size = nms_kernel_size
        self.train_cfg = train_cfg
        self.test_cfg = test_cfg
        self.common_heads = copy.deepcopy(common_heads)
        if 'vel' in self.common_heads:
            raise ValueError('KITTI TransFusion does not predict velocity')
        if self.fuse_img and self.num_views != 1:
            raise ValueError('KITTI TransFusion supports exactly one camera')
        if self.fuse_img and self.num_decoder_layers != 1:
            raise ValueError(
                'fusion baseline requires one LiDAR decoder layer')

        self.use_sigmoid_cls = loss_cls.get('use_sigmoid', False)
        if not self.use_sigmoid_cls:
            self.num_classes += 1
        self.loss_cls = MODELS.build(loss_cls)
        self.loss_bbox = MODELS.build(loss_bbox)
        self.loss_heatmap = MODELS.build(loss_heatmap)

        self.bbox_coder = build_bbox_coder(bbox_coder)
        self.sampling = False

        # a shared convolution
        self.shared_conv = build_conv_layer(
            dict(type='Conv2d'),
            in_channels,
            hidden_channel,
            kernel_size=3,
            padding=1,
            bias=bias,
        )

        layers = []
        layers.append(
            ConvModule(
                hidden_channel,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
                conv_cfg=dict(type='Conv2d'),
                norm_cfg=dict(type='BN2d'),
            ))
        layers.append(
            build_conv_layer(
                dict(type='Conv2d'),
                hidden_channel,
                num_classes,
                kernel_size=3,
                padding=1,
                bias=bias,
            ))
        self.heatmap_head = nn.Sequential(*layers)
        self.class_encoding = nn.Conv1d(num_classes, hidden_channel, 1)

        # transformer decoder layers for object query with LiDAR feature
        self.decoder = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            self.decoder.append(MODELS.build(decoder_layer))

        # Prediction Head
        self.prediction_heads = nn.ModuleList()
        for i in range(self.num_decoder_layers):
            heads = copy.deepcopy(common_heads)
            heads.update(dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.prediction_heads.append(
                SeparateHead(
                    hidden_channel,
                    heads,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg,
                    bias=bias,
                ))

        if self.fuse_img:
            self.shared_conv_img = build_conv_layer(
                dict(type='Conv2d'),
                in_channels_img,
                hidden_channel,
                kernel_size=3,
                padding=1,
                bias=bias,
            )
            self.image_heatmap_head = copy.deepcopy(self.heatmap_head)
            self.image_collapse_proj = nn.Conv1d(
                hidden_channel, hidden_channel, kernel_size=1)
            image_bev_decoder_cfg = copy.deepcopy(decoder_layer)
            image_bev_decoder_cfg['cross_only'] = True
            self.image_bev_decoder = MODELS.build(image_bev_decoder_cfg)
            self.image_query_decoder = MODELS.build(
                copy.deepcopy(decoder_layer))
            fusion_heads = copy.deepcopy(common_heads)
            fusion_heads.update(
                dict(heatmap=(self.num_classes, num_heatmap_convs)))
            self.fusion_prediction_head = SeparateHead(
                hidden_channel * 2,
                fusion_heads,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                bias=bias,
            )

        self.init_weights()
        self._init_assigner_sampler()

        # Position Embedding for Cross-Attention, which is re-used during training # noqa: E501
        x_size = self.test_cfg['grid_size'][0] // self.test_cfg[
            'out_size_factor']
        y_size = self.test_cfg['grid_size'][1] // self.test_cfg[
            'out_size_factor']
        self.bev_pos = self.create_2D_grid(x_size, y_size)

    def create_2D_grid(self, x_size, y_size):
        batch_y, batch_x = torch.meshgrid(
            torch.arange(y_size, dtype=torch.float32),
            torch.arange(x_size, dtype=torch.float32),
            indexing='ij')
        return torch.stack([batch_x + 0.5, batch_y + 0.5],
                           dim=-1).reshape(1, -1, 2)

    def init_weights(self):
        # initialize transformer
        transformer_modules = [self.decoder]
        if self.fuse_img:
            transformer_modules.extend(
                [self.image_bev_decoder, self.image_query_decoder])
        for module in transformer_modules:
            for parameter in module.parameters():
                if parameter.dim() > 1:
                    nn.init.xavier_uniform_(parameter)
        if hasattr(self, 'query'):
            nn.init.xavier_normal_(self.query)
        self.init_bn_momentum()

    def init_bn_momentum(self):
        for m in self.modules():
            if isinstance(m, (nn.BatchNorm2d, nn.BatchNorm1d)):
                m.momentum = self.bn_momentum

    def lidar_modules(self):
        """Return modules frozen during stage-2 fusion training."""
        return [
            self.shared_conv,
            self.heatmap_head,
            self.class_encoding,
            self.decoder,
            self.prediction_heads,
        ]

    def _init_assigner_sampler(self):
        """Initialize the target assigner and sampler of the head."""
        if self.train_cfg is None:
            return

        if self.sampling:
            self.bbox_sampler = build_sampler(self.train_cfg.sampler)
        else:
            self.bbox_sampler = PseudoSampler()
        if isinstance(self.train_cfg.assigner, dict):
            self.bbox_assigner = build_assigner(self.train_cfg.assigner)
        elif isinstance(self.train_cfg.assigner, list):
            self.bbox_assigner = [
                build_assigner(res) for res in self.train_cfg.assigner
            ]

    def _local_max_heatmap(self, heatmap):
        padding = self.nms_kernel_size // 2
        pooled = F.max_pool2d(
            heatmap, kernel_size=self.nms_kernel_size, stride=1, padding=0)
        if padding == 0:
            local_max = pooled
        else:
            local_max = torch.zeros_like(heatmap)
            local_max[:, :, padding:-padding, padding:-padding] = pooled
        if self.test_cfg['dataset'] == 'KITTI':
            # Pedestrian and Cyclist are too small for the class-agnostic
            # local maximum radius used by Car.
            local_max[:, 0] = heatmap[:, 0]
            local_max[:, 1] = heatmap[:, 1]
        return heatmap * (heatmap == local_max)

    def _image_guided_bev_heatmap(self, lidar_flat, img_inputs, lidar_shape,
                                  bev_pos):
        """Reproduce TransFusion's vertically collapsed image guidance."""
        image_feat = self.shared_conv_img(img_inputs)
        collapsed = image_feat.max(dim=2).values
        collapsed = self.image_collapse_proj(collapsed)
        image_pos = self.create_2D_grid(collapsed.shape[-1], 1).to(
            device=collapsed.device, dtype=collapsed.dtype)
        image_guided_bev = self.image_bev_decoder(
            lidar_flat, key=collapsed, query_pos=bev_pos, key_pos=image_pos)
        dense_image_heatmap = self.image_heatmap_head(
            image_guided_bev.reshape(lidar_shape).float())
        return image_feat, dense_image_heatmap

    def _project_queries(self, lidar_prediction, metas, image_feat):
        """Project decoded query centers and corners into the KITTI image."""
        decoded = self.bbox_coder.decode(
            lidar_prediction['heatmap'].detach(),
            lidar_prediction['rot'].detach(),
            lidar_prediction['dim'].detach(),
            lidar_prediction['center'].detach(),
            lidar_prediction['height'].detach(),
            filter=False)
        batch_size = lidar_prediction['center'].shape[0]
        feature_height, feature_width = image_feat.shape[-2:]
        centers = image_feat.new_zeros((batch_size, self.num_proposals, 2))
        radii = image_feat.new_ones((batch_size, self.num_proposals))
        valid_mask = torch.zeros((batch_size, self.num_proposals),
                                 dtype=torch.bool,
                                 device=image_feat.device)

        for batch_index in range(batch_size):
            meta = metas[batch_index]
            box_type = meta['box_type_3d']
            boxes = box_type(decoded[batch_index]['bboxes'], box_dim=7)
            box_centers = boxes.gravity_center
            box_corners = boxes.corners.reshape(-1, 3)
            points = torch.cat([box_centers, box_corners], dim=0)
            lidar2img = points.new_tensor(meta['lidar2img'])
            if lidar2img.ndim == 3 and lidar2img.shape[0] == 1:
                lidar2img = lidar2img[0]
            homography = points.new_tensor(
                meta.get('homography_matrix', np.eye(3)))
            image_shape = meta.get('pad_shape', meta['img_shape'])
            pixels, point_valid = project_lidar_points(points, lidar2img,
                                                       homography, image_shape)

            center_pixels = pixels[:self.num_proposals]
            center_valid = point_valid[:self.num_proposals]
            corner_pixels = pixels[self.num_proposals:].reshape(
                self.num_proposals, 8, 2)
            corner_valid = point_valid[self.num_proposals:].reshape(
                self.num_proposals, 8)
            feature_centers = center_pixels / self.out_size_factor_img
            center_valid &= ((feature_centers[:, 0] >= 0)
                             & (feature_centers[:, 0] < feature_width)
                             & (feature_centers[:, 1] >= 0)
                             & (feature_centers[:, 1] < feature_height))
            centers[batch_index] = feature_centers
            valid_mask[batch_index] = center_valid

            for query_index in torch.where(center_valid)[0].tolist():
                visible_corners = corner_valid[query_index]
                if visible_corners.any():
                    query_corners = corner_pixels[query_index][
                        visible_corners] / self.out_size_factor_img
                    extent = query_corners.max(
                        dim=0).values - query_corners.min(dim=0).values
                    radii[batch_index, query_index] = torch.ceil(
                        extent.norm(p=2) / 2).clamp(min=1)
        return dict(centers=centers, radii=radii, valid_mask=valid_mask)

    @staticmethod
    def _should_run_image_attention(valid_mask):
        return bool(valid_mask.any().item())

    @staticmethod
    def _apply_fusion_mask(weights, valid_mask):
        mask = valid_mask
        while mask.ndim < weights.ndim:
            mask = mask.unsqueeze(-1)
        return weights * mask.to(dtype=weights.dtype)

    @staticmethod
    def _fallback_invalid_queries(fusion_prediction, lidar_prediction,
                                  valid_mask):
        result = {}
        for key, fusion_value in fusion_prediction.items():
            lidar_value = lidar_prediction.get(key)
            if (torch.is_tensor(fusion_value) and torch.is_tensor(lidar_value)
                    and fusion_value.shape == lidar_value.shape
                    and fusion_value.shape[-1] == valid_mask.shape[-1]):
                mask = valid_mask[:, None, :]
                result[key] = torch.where(mask, fusion_value, lidar_value)
            else:
                result[key] = fusion_value
        return result

    def _fuse_visible_queries(self, lidar_query, image_feat, projection):
        """Apply Gaussian-constrained query-to-image cross-attention."""
        batch_size, _, image_height, image_width = image_feat.shape
        image_flat = image_feat.reshape(batch_size, image_feat.shape[1], -1)
        image_pos = self.create_2D_grid(image_width, image_height).to(
            device=image_feat.device, dtype=image_feat.dtype)
        image_query = torch.zeros_like(lidar_query)
        valid_mask = projection['valid_mask']
        for batch_index in range(batch_size):
            visible = valid_mask[batch_index]
            if not self._should_run_image_attention(visible):
                continue
            query_pos = projection['centers'][batch_index, visible][None]
            distance = torch.cdist(query_pos[0], image_pos[0], p=2)
            radius = projection['radii'][batch_index, visible]
            sigma = torch.clamp((radius * 2 + 1) / 6.0, min=1.0)
            gaussian = torch.exp(-distance.square() /
                                 (2 * sigma[:, None].square()))
            attention_mask = gaussian.clamp_min(
                torch.finfo(gaussian.dtype).tiny).log()
            fused = self.image_query_decoder(
                lidar_query[batch_index:batch_index + 1, :, visible],
                key=image_flat[batch_index:batch_index + 1],
                query_pos=query_pos,
                key_pos=image_pos,
                cross_attn_mask=attention_mask)
            image_query[batch_index, :, visible] = fused[0]
        return image_query

    def forward_single(self, inputs, metas, img_inputs=None):
        """Forward one LiDAR level with optional original image fusion."""
        batch_size = inputs.shape[0]
        lidar_feat = self.shared_conv(inputs)
        lidar_flat = lidar_feat.reshape(batch_size, lidar_feat.shape[1], -1)
        bev_pos = self.bev_pos.repeat(batch_size, 1, 1).to(
            device=lidar_feat.device, dtype=lidar_feat.dtype)

        with torch.autocast('cuda', enabled=False):
            dense_lidar_heatmap = self.heatmap_head(lidar_feat.float())
        dense_loss_heatmap = dense_lidar_heatmap
        image_feat = None
        if self.fuse_img:
            if img_inputs is None or metas is None:
                raise ValueError('image features and metadata are required')
            image_feat, dense_image_heatmap = self._image_guided_bev_heatmap(
                lidar_flat, img_inputs, lidar_feat.shape, bev_pos)
            heatmap = (dense_lidar_heatmap.detach().sigmoid() +
                       dense_image_heatmap.detach().sigmoid()) / 2
            dense_loss_heatmap = dense_image_heatmap
        else:
            heatmap = dense_lidar_heatmap.detach().sigmoid()

        heatmap = self._local_max_heatmap(heatmap)
        heatmap = heatmap.reshape(batch_size, heatmap.shape[1], -1)
        top_proposals = heatmap.reshape(batch_size, -1).argsort(
            dim=-1, descending=True)[..., :self.num_proposals]
        query_labels = top_proposals // heatmap.shape[-1]
        proposal_indices = top_proposals % heatmap.shape[-1]
        query_feat = lidar_flat.gather(
            2, proposal_indices[:, None].expand(-1, lidar_flat.shape[1], -1))
        class_one_hot = F.one_hot(
            query_labels, num_classes=self.num_classes).permute(0, 2, 1)
        query_feat = query_feat + self.class_encoding(class_one_hot.float())
        query_pos = bev_pos.gather(
            1, proposal_indices[:, :, None].expand(-1, -1, bev_pos.shape[-1]))

        lidar_predictions = []
        for layer_index in range(self.num_decoder_layers):
            query_feat = self.decoder[layer_index](
                query_feat,
                key=lidar_flat,
                query_pos=query_pos,
                key_pos=bev_pos)
            prediction = self.prediction_heads[layer_index](query_feat)
            prediction['center'] = (
                prediction['center'] + query_pos.permute(0, 2, 1))
            lidar_predictions.append(prediction)
            query_pos = prediction['center'].detach().permute(0, 2, 1)

        valid_query_mask = torch.ones((batch_size, self.num_proposals),
                                      dtype=torch.bool,
                                      device=lidar_feat.device)
        if self.fuse_img:
            lidar_prediction = lidar_predictions[-1]
            projection = self._project_queries(lidar_prediction, metas,
                                               image_feat)
            image_query = self._fuse_visible_queries(query_feat.detach(),
                                                     image_feat, projection)
            fusion_prediction = self.fusion_prediction_head(
                torch.cat([image_query, query_feat.detach()], dim=1))
            fusion_prediction['center'] = (
                fusion_prediction['center'] + query_pos.permute(0, 2, 1))
            valid_query_mask = projection['valid_mask']
            ret_dicts = [
                self._fallback_invalid_queries(fusion_prediction,
                                               lidar_prediction,
                                               valid_query_mask)
            ]
        else:
            ret_dicts = lidar_predictions

        query_heatmap_score = heatmap.gather(
            2, proposal_indices[:, None].expand(-1, self.num_classes, -1))
        for prediction in ret_dicts:
            prediction['query_labels'] = query_labels
            prediction['query_heatmap_score'] = query_heatmap_score
            prediction['valid_query_mask'] = valid_query_mask
        ret_dicts[0]['dense_heatmap'] = dense_loss_heatmap

        if not self.auxiliary:
            return [ret_dicts[-1]]
        metadata_keys = {
            'dense_heatmap', 'dense_heatmap_old', 'query_heatmap_score',
            'query_labels', 'valid_query_mask'
        }
        combined = {}
        for key in ret_dicts[0]:
            if key in metadata_keys:
                combined[key] = ret_dicts[0][key]
            else:
                combined[key] = torch.cat(
                    [prediction[key] for prediction in ret_dicts], dim=-1)
        return [combined]

    def forward(self, feats, metas, img_feats=None):
        """Forward pass.

        Args:
            feats (list[torch.Tensor]): Multi-level features, e.g.,
                features produced by FPN.
        Returns:
            tuple(list[dict]): Output results. first index by level, second
            index by layer
        """
        if isinstance(feats, torch.Tensor):
            feats = [feats]
        res = multi_apply(self.forward_single, feats, [metas], [img_feats])
        assert len(res) == 1, 'only support one level features.'
        return res

    def predict(self, batch_feats, batch_input_metas, img_feats=None):
        preds_dicts = self(batch_feats, batch_input_metas, img_feats=img_feats)
        res = self.predict_by_feat(preds_dicts, batch_input_metas)
        return res

    def predict_by_feat(self,
                        preds_dicts,
                        metas,
                        img=None,
                        rescale=False,
                        for_roi=False):
        """Generate bboxes from bbox head predictions.

        Args:
            preds_dicts (tuple[list[dict]]): Prediction results.
        Returns:
            list[list[dict]]: Decoded bbox, scores and labels for each layer
            & each batch.
        """
        rets = []
        for layer_id, preds_dict in enumerate(preds_dicts):
            batch_size = preds_dict[0]['heatmap'].shape[0]
            batch_score = preds_dict[0]['heatmap'][
                ..., -self.num_proposals:].sigmoid()
            # if self.loss_iou.loss_weight != 0:
            #    batch_score = torch.sqrt(batch_score * preds_dict[0]['iou'][..., -self.num_proposals:].sigmoid()) # noqa: E501
            one_hot = F.one_hot(
                preds_dict[0]['query_labels'],
                num_classes=self.num_classes).permute(0, 2, 1)
            batch_score = batch_score * preds_dict[0][
                'query_heatmap_score'] * one_hot

            batch_center = preds_dict[0]['center'][..., -self.num_proposals:]
            batch_height = preds_dict[0]['height'][..., -self.num_proposals:]
            batch_dim = preds_dict[0]['dim'][..., -self.num_proposals:]
            batch_rot = preds_dict[0]['rot'][..., -self.num_proposals:]
            batch_vel = None
            if 'vel' in preds_dict[0]:
                batch_vel = preds_dict[0]['vel'][..., -self.num_proposals:]

            temp = self.bbox_coder.decode(
                batch_score,
                batch_rot,
                batch_dim,
                batch_center,
                batch_height,
                batch_vel,
                filter=True,
            )

            if self.test_cfg['dataset'] == 'nuScenes':
                self.tasks = [
                    dict(
                        num_class=8,
                        class_names=[],
                        indices=[0, 1, 2, 3, 4, 5, 6, 7],
                        radius=-1,
                    ),
                    dict(
                        num_class=1,
                        class_names=['pedestrian'],
                        indices=[8],
                        radius=0.175,
                    ),
                    dict(
                        num_class=1,
                        class_names=['traffic_cone'],
                        indices=[9],
                        radius=0.175,
                    ),
                ]
            elif self.test_cfg['dataset'] == 'Waymo':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[0],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[1],
                        radius=0.7),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[2],
                        radius=0.7),
                ]
            elif self.test_cfg['dataset'] == 'KITTI':
                self.tasks = [
                    dict(
                        num_class=1,
                        class_names=['Pedestrian'],
                        indices=[0],
                        radius=0.5),
                    dict(
                        num_class=1,
                        class_names=['Cyclist'],
                        indices=[1],
                        radius=0.5),
                    dict(
                        num_class=1,
                        class_names=['Car'],
                        indices=[2],
                        radius=0.7),
                ]

            ret_layer = []
            for i in range(batch_size):
                boxes3d = temp[i]['bboxes']
                scores = temp[i]['scores']
                labels = temp[i]['labels']
                # adopt circle nms for different categories
                if self.test_cfg['nms_type'] is not None:
                    keep_mask = torch.zeros_like(scores)
                    for task in self.tasks:
                        task_mask = torch.zeros_like(scores)
                        for cls_idx in task['indices']:
                            task_mask += labels == cls_idx
                        task_mask = task_mask.bool()
                        if task['radius'] > 0:
                            if self.test_cfg['nms_type'] == 'circle':
                                boxes_for_nms = torch.cat(
                                    [
                                        boxes3d[task_mask][:, :2],
                                        scores[:, None][task_mask],
                                    ],
                                    dim=1,
                                )
                                task_keep_indices = torch.tensor(
                                    circle_nms(
                                        boxes_for_nms.detach().cpu().numpy(),
                                        task['radius'],
                                    ))
                            else:
                                boxes_for_nms = xywhr2xyxyr(
                                    metas[i]['box_type_3d'](
                                        boxes3d[task_mask][:, :7], 7).bev)
                                top_scores = scores[task_mask]
                                task_keep_indices = nms_bev(
                                    boxes_for_nms,
                                    top_scores,
                                    thresh=task['radius'],
                                    pre_maxsize=self.test_cfg['pre_maxsize'],
                                    post_max_size=self.
                                    test_cfg['post_maxsize'],
                                )
                        else:
                            task_keep_indices = torch.arange(task_mask.sum())
                        if task_keep_indices.shape[0] != 0:
                            keep_indices = torch.where(
                                task_mask != 0)[0][task_keep_indices]
                            keep_mask[keep_indices] = 1
                    keep_mask = keep_mask.bool()
                    ret = dict(
                        bboxes=boxes3d[keep_mask],
                        scores=scores[keep_mask],
                        labels=labels[keep_mask],
                    )
                else:  # no nms
                    ret = dict(bboxes=boxes3d, scores=scores, labels=labels)

                temp_instances = InstanceData()
                temp_instances.bboxes_3d = metas[i]['box_type_3d'](
                    ret['bboxes'], box_dim=ret['bboxes'].shape[-1])
                temp_instances.scores_3d = ret['scores']
                temp_instances.labels_3d = ret['labels'].int()

                ret_layer.append(temp_instances)

            rets.append(ret_layer)
        assert len(
            rets
        ) == 1, f'only support one layer now, but get {len(rets)} layers'

        return rets[0]

    def get_targets(self, batch_gt_instances_3d: List[InstanceData],
                    preds_dict: List[dict]):
        """Generate training targets.
        Args:
            batch_gt_instances_3d (List[InstanceData]):
            preds_dict (list[dict]): The prediction results. The index of the
                list is the index of layers. The inner dict contains
                predictions of one mini-batch:
                - center: (bs, 2, num_proposals)
                - height: (bs, 1, num_proposals)
                - dim: (bs, 3, num_proposals)
                - rot: (bs, 2, num_proposals)
                - vel: (bs, 2, num_proposals)
                - cls_logit: (bs, num_classes, num_proposals)
                - query_score: (bs, num_classes, num_proposals)
                - heatmap: The original heatmap before fed into transformer
                    decoder, with shape (bs, 10, h, w)
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [BS, num_proposals]
                - torch.Tensor: classification weights (mask)
                    [BS, num_proposals]
                - torch.Tensor: regression target. [BS, num_proposals, 8]
                - torch.Tensor: regression weights. [BS, num_proposals, 8]
        """
        # change preds_dict into list of dict (index by batch_id)
        # preds_dict[0]['center'].shape [bs, 3, num_proposal]
        list_of_pred_dict = []
        for batch_idx in range(len(batch_gt_instances_3d)):
            pred_dict = {}
            for key, value in preds_dict[0].items():
                pred_dict[key] = value[batch_idx:batch_idx + 1]
            list_of_pred_dict.append(pred_dict)

        assert len(batch_gt_instances_3d) == len(list_of_pred_dict)
        res_tuple = multi_apply(
            self.get_targets_single,
            batch_gt_instances_3d,
            list_of_pred_dict,
            np.arange(len(batch_gt_instances_3d)),
        )
        labels = torch.cat(res_tuple[0], dim=0)
        label_weights = torch.cat(res_tuple[1], dim=0)
        bbox_targets = torch.cat(res_tuple[2], dim=0)
        bbox_weights = torch.cat(res_tuple[3], dim=0)
        ious = torch.cat(res_tuple[4], dim=0)
        num_pos = np.sum(res_tuple[5])
        matched_ious = np.mean(res_tuple[6])
        heatmap = torch.cat(res_tuple[7], dim=0)
        return (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        )

    def get_targets_single(self, gt_instances_3d, preds_dict, batch_idx):
        """Generate training targets for a single sample.
        Args:
            gt_instances_3d (:obj:`InstanceData`): ground truth of instances.
            preds_dict (dict): dict of prediction result for a single sample.
        Returns:
            tuple[torch.Tensor]: Tuple of target including \
                the following results in order.
                - torch.Tensor: classification target.  [1, num_proposals]
                - torch.Tensor: classification weights (mask) [1,
                    num_proposals] # noqa: E501
                - torch.Tensor: regression target. [1, num_proposals, 8]
                - torch.Tensor: regression weights. [1, num_proposals, 8]
                - torch.Tensor: iou target. [1, num_proposals]
                - int: number of positive proposals
                - torch.Tensor: heatmap targets.
        """
        # 1. Assignment
        gt_bboxes_3d = gt_instances_3d.bboxes_3d
        gt_labels_3d = gt_instances_3d.labels_3d
        num_proposals = preds_dict['center'].shape[-1]

        # get pred boxes, carefully ! don't change the network outputs
        score = copy.deepcopy(preds_dict['heatmap'].detach())
        center = copy.deepcopy(preds_dict['center'].detach())
        height = copy.deepcopy(preds_dict['height'].detach())
        dim = copy.deepcopy(preds_dict['dim'].detach())
        rot = copy.deepcopy(preds_dict['rot'].detach())
        if 'vel' in preds_dict.keys():
            vel = copy.deepcopy(preds_dict['vel'].detach())
        else:
            vel = None

        boxes_dict = self.bbox_coder.decode(
            score, rot, dim, center, height,
            vel)  # decode the prediction to real world metric bbox
        bboxes_tensor = boxes_dict[0]['bboxes']
        gt_bboxes_tensor = gt_bboxes_3d.tensor.to(score.device)
        # each layer should do label assign separately.
        if self.auxiliary:
            num_layer = self.num_decoder_layers
        else:
            num_layer = 1

        assign_result_list = []
        for idx_layer in range(num_layer):
            bboxes_tensor_layer = bboxes_tensor[self.num_proposals *
                                                idx_layer:self.num_proposals *
                                                (idx_layer + 1), :]
            score_layer = score[
                ...,
                self.num_proposals * idx_layer:self.num_proposals *
                (idx_layer + 1),
            ]

            pred_instances = InstanceData(
                bboxes=bboxes_tensor_layer,
                scores=score_layer[0].transpose(0, 1))
            gt_instances = InstanceData(
                bboxes=gt_bboxes_tensor, labels=gt_labels_3d)
            assign_result = self.bbox_assigner.assign(pred_instances,
                                                      gt_instances,
                                                      self.train_cfg)
            assign_result_list.append(assign_result)

        # combine assign result of each layer
        assign_result_ensemble = AssignResult(
            num_gts=sum([res.num_gts for res in assign_result_list]),
            gt_inds=torch.cat([res.gt_inds for res in assign_result_list]),
            max_overlaps=torch.cat(
                [res.max_overlaps for res in assign_result_list]),
            labels=torch.cat([res.labels for res in assign_result_list]),
        )

        # 2. Sampling. Compatible with the interface of `PseudoSampler` in
        # mmdet.
        gt_instances, pred_instances = InstanceData(
            bboxes=gt_bboxes_tensor), InstanceData(priors=bboxes_tensor)
        sampling_result = self.bbox_sampler.sample(assign_result_ensemble,
                                                   pred_instances,
                                                   gt_instances)
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        assert len(pos_inds) + len(neg_inds) == num_proposals

        # 3. Create target for loss computation
        bbox_targets = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        bbox_weights = torch.zeros([num_proposals, self.bbox_coder.code_size
                                    ]).to(center.device)
        ious = assign_result_ensemble.max_overlaps
        ious = torch.clamp(ious, min=0.0, max=1.0)
        labels = bboxes_tensor.new_zeros(num_proposals, dtype=torch.long)
        label_weights = bboxes_tensor.new_zeros(
            num_proposals, dtype=torch.long)

        if gt_labels_3d is not None:  # default label is -1
            labels += self.num_classes

        # both pos and neg have classification loss, only pos has regression
        # and iou loss
        if len(pos_inds) > 0:
            pos_bbox_targets = self.bbox_coder.encode(
                sampling_result.pos_gt_bboxes)

            bbox_targets[pos_inds, :] = pos_bbox_targets
            bbox_weights[pos_inds, :] = 1.0

            if gt_labels_3d is None:
                labels[pos_inds] = 1
            else:
                labels[pos_inds] = gt_labels_3d[
                    sampling_result.pos_assigned_gt_inds]
            if self.train_cfg.pos_weight <= 0:
                label_weights[pos_inds] = 1.0
            else:
                label_weights[pos_inds] = self.train_cfg.pos_weight

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # # compute dense heatmap targets
        device = labels.device
        gt_bboxes_3d = torch.cat(
            [gt_bboxes_3d.gravity_center, gt_bboxes_3d.tensor[:, 3:]],
            dim=1).to(device)
        grid_size = torch.tensor(self.train_cfg['grid_size'])
        pc_range = torch.tensor(self.train_cfg['point_cloud_range'])
        voxel_size = torch.tensor(self.train_cfg['voxel_size'])
        feature_map_size = (grid_size[:2] // self.train_cfg['out_size_factor']
                            )  # [x_len, y_len]
        heatmap = gt_bboxes_3d.new_zeros(self.num_classes, feature_map_size[1],
                                         feature_map_size[0])
        for idx in range(len(gt_bboxes_3d)):
            width = gt_bboxes_3d[idx][3]
            length = gt_bboxes_3d[idx][4]
            width = width / voxel_size[0] / self.train_cfg['out_size_factor']
            length = length / voxel_size[1] / self.train_cfg['out_size_factor']
            if width > 0 and length > 0:
                radius = gaussian_radius(
                    (length, width),
                    min_overlap=self.train_cfg['gaussian_overlap'])
                radius = max(self.train_cfg['min_radius'], int(radius))
                x, y = gt_bboxes_3d[idx][0], gt_bboxes_3d[idx][1]

                coor_x = ((x - pc_range[0]) / voxel_size[0] /
                          self.train_cfg['out_size_factor'])
                coor_y = ((y - pc_range[1]) / voxel_size[1] /
                          self.train_cfg['out_size_factor'])

                center = torch.tensor([coor_x, coor_y],
                                      dtype=torch.float32,
                                      device=device)
                center_int = center.to(torch.int32)

                # original
                # draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]], center_int, radius) # noqa: E501
                # NOTE: fix
                draw_heatmap_gaussian(heatmap[gt_labels_3d[idx]],
                                      center_int[[1, 0]], radius)

        mean_iou = ious[pos_inds].sum() / max(len(pos_inds), 1)
        return (
            labels[None],
            label_weights[None],
            bbox_targets[None],
            bbox_weights[None],
            ious[None],
            int(pos_inds.shape[0]),
            float(mean_iou),
            heatmap[None],
        )

    def loss(self, batch_feats, batch_data_samples, img_feats=None):
        """Loss function for CenterHead.

        Args:
            batch_feats (): Features in a batch.
            batch_data_samples (List[:obj:`Det3DDataSample`]): The Data
                Samples. It usually includes information such as
                `gt_instance_3d`.
        Returns:
            dict[str:torch.Tensor]: Loss of heatmap and bbox of each task.
        """
        batch_input_metas, batch_gt_instances_3d = [], []
        for data_sample in batch_data_samples:
            batch_input_metas.append(data_sample.metainfo)
            batch_gt_instances_3d.append(data_sample.gt_instances_3d)
        preds_dicts = self(batch_feats, batch_input_metas, img_feats=img_feats)
        loss = self.loss_by_feat(preds_dicts, batch_gt_instances_3d)

        return loss

    def loss_by_feat(self, preds_dicts: Tuple[List[dict]],
                     batch_gt_instances_3d: List[InstanceData], *args,
                     **kwargs):
        (
            labels,
            label_weights,
            bbox_targets,
            bbox_weights,
            ious,
            num_pos,
            matched_ious,
            heatmap,
        ) = self.get_targets(batch_gt_instances_3d, preds_dicts[0])
        preds_dict = preds_dicts[0][0]
        if self.fuse_img:
            valid_query_mask = preds_dict['valid_query_mask']
            label_weights = self._apply_fusion_mask(label_weights,
                                                    valid_query_mask)
            bbox_weights = self._apply_fusion_mask(bbox_weights,
                                                   valid_query_mask)
            num_pos = bbox_weights.max(-1).values.sum()
        loss_dict = dict()

        # compute heatmap loss
        loss_heatmap = self.loss_heatmap(
            clip_sigmoid(preds_dict['dense_heatmap']).float(),
            heatmap.float(),
            avg_factor=max(heatmap.eq(1).float().sum().item(), 1),
        )
        loss_dict['loss_heatmap'] = loss_heatmap

        # compute loss for each layer
        for idx_layer in range(
                self.num_decoder_layers if self.auxiliary else 1):
            if idx_layer == self.num_decoder_layers - 1 or (
                    idx_layer == 0 and self.auxiliary is False):
                prefix = 'layer_-1'
            else:
                prefix = f'layer_{idx_layer}'

            layer_labels = labels[
                ...,
                idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals,
            ].reshape(-1)
            layer_label_weights = label_weights[
                ...,
                idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals,
            ].reshape(-1)
            layer_score = preds_dict['heatmap'][
                ...,
                idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals,
            ]
            layer_cls_score = layer_score.permute(0, 2, 1).reshape(
                -1, self.num_classes)
            layer_loss_cls = self.loss_cls(
                layer_cls_score.float(),
                layer_labels,
                layer_label_weights,
                avg_factor=max(num_pos, 1),
            )

            layer_center = preds_dict['center'][
                ...,
                idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals,
            ]
            layer_height = preds_dict['height'][
                ...,
                idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals,
            ]
            layer_rot = preds_dict['rot'][
                ...,
                idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals,
            ]
            layer_dim = preds_dict['dim'][
                ...,
                idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals,
            ]
            preds = torch.cat(
                [layer_center, layer_height, layer_dim, layer_rot],
                dim=1).permute(0, 2, 1)  # [BS, num_proposals, code_size]
            if 'vel' in preds_dict.keys():
                layer_vel = preds_dict['vel'][
                    ...,
                    idx_layer * self.num_proposals:(idx_layer + 1) *
                    self.num_proposals,
                ]
                preds = torch.cat([
                    layer_center, layer_height, layer_dim, layer_rot, layer_vel
                ],
                                  dim=1).permute(
                                      0, 2,
                                      1)  # [BS, num_proposals, code_size]
            code_weights = self.train_cfg.get('code_weights', None)
            layer_bbox_weights = bbox_weights[
                :,
                idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals,
                :,
            ]
            layer_reg_weights = layer_bbox_weights * layer_bbox_weights.new_tensor(  # noqa: E501
                code_weights)
            layer_bbox_targets = bbox_targets[
                :,
                idx_layer * self.num_proposals:(idx_layer + 1) *
                self.num_proposals,
                :,
            ]
            layer_loss_bbox = self.loss_bbox(
                preds,
                layer_bbox_targets,
                layer_reg_weights,
                avg_factor=max(num_pos, 1))

            loss_dict[f'{prefix}_loss_cls'] = layer_loss_cls
            loss_dict[f'{prefix}_loss_bbox'] = layer_loss_bbox
            # loss_dict[f'{prefix}_loss_iou'] = layer_loss_iou

        loss_dict['matched_ious'] = layer_loss_cls.new_tensor(matched_ious)

        return loss_dict
