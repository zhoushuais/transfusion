"""Unified LiDAR and camera-LiDAR detector for TransFusion KITTI."""

from typing import Optional

from torch import Tensor, nn

from mmdet3d.models.detectors import VoxelNet
from mmdet3d.registry import MODELS


@MODELS.register_module()
class TransFusionKITTIDetector(VoxelNet):
    """VoxelNet-based detector with optional query-level image fusion."""

    def __init__(
        self,
        fuse_img: bool = False,
        img_backbone: Optional[dict] = None,
        img_neck: Optional[dict] = None,
        img_feature_level: int = 0,
        freeze_img: bool = False,
        freeze_lidar: bool = False,
        **kwargs,
    ) -> None:
        super().__init__(**kwargs)
        self.fuse_img = fuse_img
        self.img_feature_level = img_feature_level
        self.freeze_img = freeze_img
        self.freeze_lidar = freeze_lidar
        if fuse_img:
            if img_backbone is None:
                raise ValueError('img_backbone is required when fuse_img=True')
            self.img_backbone = MODELS.build(img_backbone)
            self.img_neck = (
                MODELS.build(img_neck) if img_neck is not None else None)
        elif img_backbone is not None or img_neck is not None:
            raise ValueError('image modules require fuse_img=True')
        self._apply_freeze_policy()

    @staticmethod
    def freeze_module(module: nn.Module) -> None:
        """Freeze parameters and normalization statistics in ``module``."""
        module.eval()
        for parameter in module.parameters():
            parameter.requires_grad = False

    def _apply_freeze_policy(self) -> None:
        if self.freeze_img and self.fuse_img:
            self.freeze_module(self.img_backbone)
            if self.img_neck is not None:
                self.freeze_module(self.img_neck)
        if self.freeze_lidar:
            lidar_modules = [
                self.voxel_encoder,
                self.middle_encoder,
                self.backbone,
            ]
            if self.with_neck:
                lidar_modules.append(self.neck)
            lidar_modules.extend(self.bbox_head.lidar_modules())
            for module in lidar_modules:
                self.freeze_module(module)

    def train(self, mode: bool = True):
        super().train(mode)
        if mode:
            self._apply_freeze_policy()
        return self

    def extract_img_feat(self, imgs: Optional[Tensor]) -> Optional[Tensor]:
        """Extract one FPN level from a single KITTI camera image."""
        if not self.fuse_img:
            return None
        if imgs is None:
            raise ValueError('imgs are required when fuse_img=True')
        if imgs.ndim == 5:
            if imgs.shape[1] != 1:
                raise ValueError('KITTI fusion expects exactly one camera')
            imgs = imgs[:, 0]
        if imgs.ndim != 4:
            raise ValueError('imgs must have shape [B,C,H,W] or [B,1,C,H,W]')
        features = self.img_backbone(imgs.float())
        if self.img_neck is not None:
            features = self.img_neck(features)
        if isinstance(features, Tensor):
            if self.img_feature_level != 0:
                raise IndexError('single image feature only has level zero')
            return features
        return features[self.img_feature_level]

    @staticmethod
    def _metas(batch_data_samples):
        if batch_data_samples is None:
            return None
        return [sample.metainfo for sample in batch_data_samples]

    def loss(self, batch_inputs_dict, batch_data_samples, **kwargs):
        pts_feats = self.extract_feat(batch_inputs_dict)
        img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs'))
        return self.bbox_head.loss(
            pts_feats, batch_data_samples, img_feats=img_feats, **kwargs)

    def predict(self, batch_inputs_dict, batch_data_samples, **kwargs):
        pts_feats = self.extract_feat(batch_inputs_dict)
        img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs'))
        results_list = self.bbox_head.predict(
            pts_feats,
            self._metas(batch_data_samples),
            img_feats=img_feats,
            **kwargs,
        )
        return self.add_pred_to_datasample(batch_data_samples, results_list)

    def _forward(self, batch_inputs_dict, data_samples=None, **kwargs):
        pts_feats = self.extract_feat(batch_inputs_dict)
        img_feats = self.extract_img_feat(batch_inputs_dict.get('imgs'))
        return self.bbox_head(
            pts_feats,
            self._metas(data_samples),
            img_feats=img_feats,
            **kwargs,
        )
