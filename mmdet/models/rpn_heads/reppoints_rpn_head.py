import numpy as np
import torch
import torch.nn as nn
from mmcv.cnn import ConvModule
from mmcv.ops import DeformConv2d
from mmengine.config import ConfigDict
from mmengine.structures import InstanceData
from torch import Tensor
from typing import Sequence, Tuple, List, Dict, Optional, Union

from mmdet.structures.bbox import HorizontalBoxes, BaseBoxes
from mmdet.registry import MODELS, TASK_UTILS
from mmdet.utils import ConfigType, InstanceList, MultiConfig, OptInstanceList
from mmdet.models.task_modules.prior_generators import MlvlPointGenerator
from mmdet.models.task_modules.samplers import PseudoSampler
from mmdet.models.utils import (filter_scores_and_topk, images_to_levels, multi_apply, unmap)
from mmdet.models.dense_heads.anchor_free_head import AnchorFreeHead
from mmdet.models.task_modules.assigners import MaxIoUAssigner
from mmdet.structures.bbox import bbox_overlaps


@MODELS.register_module()
class RepPointsRPNHead(AnchorFreeHead):
    """RepPoints RPN head for generating proposals in Mask R-CNN with BiFPN.

    Args:
        in_channels (int): Number of channels in the input feature map.
        point_feat_channels (int): Number of channels of points features.
        num_points (int): Number of points (default=9).
        gradient_mul (float): Gradient multiplier for points refinement.
        point_strides (Sequence[int]): Points strides for each FPN level.
        point_base_scale (int): Base scale for point generation.
        loss_cls (dict): Config of classification loss.
        loss_bbox_init (dict): Config of initial points loss.
        loss_bbox_refine (dict): Config of refined points loss.
        transform_method (str): Method to transform RepPoints to bbox.
        moment_mul (float): Multiplier for moment transformation.
        topk (int): Number of top scoring proposals to keep.
        num_classes (int): Number of classes (always 1 for RPN).
        init_cfg (dict or list[dict]): Initialization config dict.
    """

    def __init__(self,
                 in_channels: int,
                 point_feat_channels: int = 256,
                 num_points: int = 9,
                 gradient_mul: float = 0.1,
                 point_strides: Sequence[int] = [8, 16, 32, 64, 128],
                 point_base_scale: int = 4,
                 loss_cls: ConfigType = dict(
                     type='CrossEntropyLoss',
                     use_sigmoid=True,
                     loss_weight=1.0),
                 loss_bbox_init: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=0.5),
                 loss_bbox_refine: ConfigType = dict(
                     type='SmoothL1Loss', beta=1.0 / 9.0, loss_weight=1.0),
                 transform_method: str = 'moment',
                 moment_mul: float = 0.01,
                 topk: int = 1000,
                 num_classes: int = 1,
                 init_cfg: MultiConfig = dict(
                     type='Normal',
                     layer='Conv2d',
                     std=0.01,
                     override=dict(
                         type='Normal',
                         name='reppoints_cls_out',
                         std=0.01,
                         bias_prob=0.01)),
                 **kwargs) -> None:
        
        # RPN is binary classification (object vs background)
        self.num_points = num_points
        self.point_feat_channels = point_feat_channels
        self.use_grid_points = False  # Always False for RPN
        self.center_init = True  # Always True for RPN
        
        # Deformable convolution setup
        self.dcn_kernel = int(np.sqrt(num_points))
        self.dcn_pad = int((self.dcn_kernel - 1) / 2)
        assert self.dcn_kernel * self.dcn_kernel == num_points, \
            'The points number should be a square number.'
        assert self.dcn_kernel % 2 == 1, \
            'The points number should be an odd square number.'
        
        # Create base offset for deformable convolution
        dcn_base = np.arange(-self.dcn_pad, self.dcn_pad + 1).astype(np.float64)
        dcn_base_y = np.repeat(dcn_base, self.dcn_kernel)
        dcn_base_x = np.tile(dcn_base, self.dcn_kernel)
        dcn_base_offset = np.stack([dcn_base_y, dcn_base_x], axis=1).reshape((-1))
        self.dcn_base_offset = torch.tensor(dcn_base_offset).view(1, -1, 1, 1)
        
        # Initialize base class (AnchorFreeHead)
        super().__init__(
            num_classes=num_classes,  # 1 for RPN
            in_channels=in_channels,
            loss_cls=loss_cls,
            init_cfg=init_cfg,
            **kwargs)

        self.assigner = TASK_UTILS.build(dict(type='mmdet.MaxIoUAssigner',
                                     pos_iou_thr=0.7,
                                     neg_iou_thr=0.3,
                                     min_pos_iou=0.3,
                                     ignore_iof_thr=-1,
                                     match_low_quality=True,
                                     iou_calculator=dict(type='mmdet.BboxOverlaps2D')))
        self.sampler = PseudoSampler()  # Or another sampler if you prefer
        
        self._init_layers()
        
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
        self.prior_generator = MlvlPointGenerator(self.point_strides, offset=0.)
        self.topk = topk
        
        # For RPN we always use sigmoid classification
        self.use_sigmoid_cls = True
        self.cls_out_channels = self.num_classes
        
        self.loss_bbox_init = MODELS.build(loss_bbox_init)
        self.loss_bbox_refine = MODELS.build(loss_bbox_refine)
        
        self.transform_method = transform_method
        if self.transform_method == 'moment':
            self.moment_transfer = nn.Parameter(
                data=torch.zeros(2), requires_grad=True)
            self.moment_mul = moment_mul

    def get_targets(self,
                    points: List[List[Tensor]],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: OptInstanceList = None,
                    stage: str = 'init',
                    return_sampling_results: bool = False) -> tuple:
        """Compute regression and classification targets for points."""
        # Convert batch_img_metas to the expected format
        img_metas = [{
            'img_shape': meta['img_shape'],
            'scale_factor': meta['scale_factor'],
            'batch_input_shape': meta['img_shape']
        } for meta in batch_img_metas]

        # Get points for each level
        num_levels = len(self.point_strides)
        mlvl_points = []
        for i in range(num_levels):
            points_per_level = []
            for img_id in range(len(batch_img_metas)):
                points_per_level.append(points[img_id][i])
            mlvl_points.append(points_per_level)

        # Get targets for each image
        (labels_list, label_weights_list, 
        bbox_gt_list, candidate_list, 
        bbox_weights_list, avg_factor) = multi_apply(
            self._get_targets_single,
            mlvl_points,
            batch_gt_instances,
            batch_img_metas,
            batch_gt_instances_ignore,
            stage=stage)

        return (labels_list, label_weights_list, bbox_gt_list, 
                candidate_list, bbox_weights_list, avg_factor)

    def _get_targets_single(self,
                          points: List[Tensor],
                          gt_instances: InstanceData,
                          img_meta: dict,
                          gt_instances_ignore: Optional[InstanceData] = None,
                          stage: str = 'init') -> tuple:
        """Compute targets for a single image."""
        # Convert points to bboxes
        bboxes = []
        for points_per_level in points:
            bboxes_per_level = self.points2bbox(points_per_level.unsqueeze(0))
            # Ensure bboxes have shape (N, 4)
            if bboxes_per_level.dim() == 4:
                bboxes_per_level = bboxes_per_level.view(-1, 4)
            bboxes.append(bboxes_per_level.squeeze(0))
        bboxes = torch.cat(bboxes, dim=0)

        # Create InstanceData for predictions
        pred_instances = InstanceData(priors=bboxes)
        
        # Get ground truth boxes
        gt_bboxes = gt_instances.bboxes
        if gt_bboxes.dim() == 3:
            gt_bboxes = gt_bboxes.squeeze(0)
        
        # Create InstanceData for ground truth
        gt_instances = InstanceData(bboxes=gt_bboxes)
        
        # Assign targets
        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore)
        
        # Sample results
        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)
        
        # Prepare targets
        num_points = bboxes.shape[0]
        labels = bboxes.new_full((num_points,), self.num_classes, dtype=torch.long)
        label_weights = bboxes.new_zeros(num_points, dtype=torch.float)
        bbox_gt = bboxes.new_zeros((num_points, 4), dtype=torch.float)
        bbox_weights = bboxes.new_zeros((num_points, 4), dtype=torch.float)
        
        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds
        
        if len(pos_inds) > 0:
            labels[pos_inds] = 0  # RPN binary classification
            bbox_gt[pos_inds, :] = sampling_result.pos_gt_bboxes.tensor
            bbox_weights[pos_inds, :] = 1.0
        
        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0
        
        # Split back into per-level
        start_idx = 0
        labels_list = []
        label_weights_list = []
        bbox_gt_list = []
        bbox_weights_list = []
        for points_per_level in points:
            end_idx = start_idx + len(points_per_level)
            labels_list.append(labels[start_idx:end_idx])
            label_weights_list.append(label_weights[start_idx:end_idx])
            bbox_gt_list.append(bbox_gt[start_idx:end_idx])
            bbox_weights_list.append(bbox_weights[start_idx:end_idx])
            start_idx = end_idx

        print(f"Number of positive samples: {len(pos_inds)}")
        print(f"Number of negative samples: {len(neg_inds)}")
        
        return (labels_list, label_weights_list, bbox_gt_list, 
                points, bbox_weights_list, len(pos_inds))

    def loss_by_feat(self,
                    cls_scores: List[Tensor],
                    pts_preds_init: List[Tensor],
                    pts_preds_refine: List[Tensor],
                    batch_gt_instances: InstanceList,
                    batch_img_metas: List[dict],
                    batch_gt_instances_ignore: Optional[List[Tensor]] = None) -> dict:
        """Calculate loss based on features.
        
        Args:
            cls_scores (list[Tensor]): Classification scores for each level.
            pts_preds_init (list[Tensor]): Initial points predictions.
            pts_preds_refine (list[Tensor]): Refined points predictions.
            batch_gt_instances (list[:obj:`InstanceData`]): Batch of gt instances.
            batch_img_metas (list[dict]): Meta info of each image.
            batch_gt_instances_ignore (list[:obj:`InstanceData`], optional):
                Batch of instances to be ignored.
                
        Returns:
            dict: Dictionary of loss components.
        """
        # Convert batch_img_metas to the expected format
        img_metas = [{
            'img_shape': meta['img_shape'],
            'scale_factor': meta['scale_factor'],
            'batch_input_shape': meta['img_shape']  # Add this if needed
        } for meta in batch_img_metas]

        return self.loss(
            cls_scores,
            pts_preds_init,
            pts_preds_refine,
            batch_gt_instances,
            img_metas,
            batch_gt_instances_ignore=batch_gt_instances_ignore)


    
    def _init_layers(self) -> None:
        """Initialize layers of the head."""
        self.relu = nn.ReLU(inplace=True)
        self.cls_convs = nn.ModuleList()
        self.reg_convs = nn.ModuleList()
        
        # Build stacked convs for classification and regression
        for i in range(self.stacked_convs):
            chn = self.in_channels if i == 0 else self.feat_channels
            self.cls_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
            self.reg_convs.append(
                ConvModule(
                    chn,
                    self.feat_channels,
                    3,
                    stride=1,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg))
        
        # Output dimension for points (9 points * 4 coordinates each)
        pts_out_dim = 4 * self.num_points
        
        # Initialize deformable conv layers
        self.reppoints_cls_conv = DeformConv2d(
            self.feat_channels,
            self.point_feat_channels,
            self.dcn_kernel, 1,
            self.dcn_pad)
        self.reppoints_cls_out = nn.Conv2d(
            self.point_feat_channels,
            self.cls_out_channels, 1, 1, 0)
        
        # Initial points prediction
        self.reppoints_pts_init_conv = nn.Conv2d(
            self.feat_channels,
            self.point_feat_channels, 3, 1, 1)
        self.reppoints_pts_init_out = nn.Conv2d(
            self.point_feat_channels,
            pts_out_dim, 1, 1, 0)
        
        # Refined points prediction
        self.reppoints_pts_refine_conv = DeformConv2d(
            self.feat_channels,
            self.point_feat_channels,
            self.dcn_kernel, 1,
            self.dcn_pad)
        self.reppoints_pts_refine_out = nn.Conv2d(
            self.point_feat_channels,
            pts_out_dim, 1, 1, 0)

    def forward(self, feats: Tuple[Tensor]) -> Tuple[List[Tensor]]:
        """Forward features from BiFPN and return proposals.
        
        Args:
            feats (tuple[Tensor]): Features from BiFPN.
            
        Returns:
            tuple: cls_scores, bbox_preds for each level.
        """
        return multi_apply(self.forward_single, feats)

    def forward_single(self, x: Tensor) -> Tuple[Tensor]:
        """Forward feature map of a single FPN level."""
        dcn_base_offset = self.dcn_base_offset.type_as(x)
        
        # Initialize points
        if self.use_grid_points or not self.center_init:
            scale = self.point_base_scale / 2
            points_init = dcn_base_offset / dcn_base_offset.max() * scale
            bbox_init = x.new_tensor([-scale, -scale, scale, scale]).view(1, 4, 1, 1)
        else:
            points_init = 0
            
        cls_feat = x
        pts_feat = x
        
        # Apply classification convs
        for cls_conv in self.cls_convs:
            cls_feat = cls_conv(cls_feat)
        
        # Apply regression convs
        for reg_conv in self.reg_convs:
            pts_feat = reg_conv(pts_feat)
        
        # Initialize reppoints
        pts_out_init = self.reppoints_pts_init_out(
            self.relu(self.reppoints_pts_init_conv(pts_feat)))
        
        if self.use_grid_points:
            pts_out_init, bbox_out_init = self.gen_grid_from_reg(
                pts_out_init, bbox_init.detach())
        else:
            pts_out_init = pts_out_init + points_init
        
        # Refine and classify reppoints
        pts_out_init_grad_mul = (1 - self.gradient_mul) * pts_out_init.detach() + \
                               self.gradient_mul * pts_out_init
        
        # Use only first 18 channels for DCN offset (x/y for 9 points)
        dcn_offset = pts_out_init_grad_mul[:, :18, :, :] - dcn_base_offset
        
        # Classification output
        cls_out = self.reppoints_cls_out(
            self.relu(self.reppoints_cls_conv(cls_feat, dcn_offset)))
        
        # Refined points output
        pts_out_refine = self.reppoints_pts_refine_out(
            self.relu(self.reppoints_pts_refine_conv(pts_feat, dcn_offset)))
        
        if self.use_grid_points:
            pts_out_refine, bbox_out_refine = self.gen_grid_from_reg(
                pts_out_refine, bbox_out_init.detach())
        else:
            pts_out_refine = pts_out_refine + pts_out_init.detach()

        if self.training:
            return cls_out, pts_out_init, pts_out_refine
        else:
            # For inference, return classification scores and bbox predictions
            return cls_out, self.points2bbox(pts_out_refine)

    def points2bbox(self, pts: Tensor, y_first: bool = True) -> Tensor:
        """Convert point set to bounding box."""
        # Add debug prints
        print(f"Input pts shape: {pts.shape}")
        
        # Handle different input dimensions
        if pts.dim() == 2:
            pts = pts.view(1, pts.shape[0], pts.shape[1], 1)
        elif pts.dim() == 3:
            pts = pts.unsqueeze(-1)
        
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        print(f"Reshaped pts shape: {pts_reshape.shape}")
        
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]
        
        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        elif self.transform_method == 'partial_minmax':
            pts_y = pts_y[:, :4, ...]
            pts_x = pts_x[:, :4, ...]
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)
        elif self.transform_method == 'moment':
            pts_y_mean = pts_y.mean(dim=1, keepdim=True)
            pts_x_mean = pts_x.mean(dim=1, keepdim=True)
            pts_y_std = torch.std(pts_y - pts_y_mean, dim=1, keepdim=True)
            pts_x_std = torch.std(pts_x - pts_x_mean, dim=1, keepdim=True)
            moment_transfer = (self.moment_transfer * self.moment_mul) + \
                            (self.moment_transfer.detach() * (1 - self.moment_mul))
            moment_width_transfer = moment_transfer[0]
            moment_height_transfer = moment_transfer[1]
            half_width = pts_x_std * torch.exp(moment_width_transfer)
            half_height = pts_y_std * torch.exp(moment_height_transfer)
            bbox = torch.cat([
                pts_x_mean - half_width, pts_y_mean - half_height,
                pts_x_mean + half_width, pts_y_mean + half_height
            ], dim=1)
        else:
            raise NotImplementedError
        
        print(f"Raw bbox shape: {bbox.shape}")
        
        # Reshape to (N, 4)
        if bbox.dim() == 4:
            bbox = bbox.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        elif bbox.dim() == 3:
            bbox = bbox.permute(0, 2, 1).contiguous().view(-1, 4)
        elif bbox.dim() == 2:
            if bbox.size(-1) != 4:
                bbox = bbox.view(-1, 4)
        else:
            raise ValueError(f"Unexpected bbox dimension: {bbox.dim()}")
        
        print(f"Final bbox shape: {bbox.shape}")
        return bbox

    def gen_grid_from_reg(self, reg: Tensor, previous_boxes: Tensor) -> Tuple[Tensor]:
        """Generate grid from regression values and previous boxes."""
        b, _, h, w = reg.shape
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.
        bwh = (previous_boxes[:, 2:, ...] - previous_boxes[:, :2, ...]).clamp(min=1e-6)
        
        grid_topleft = bxy + bwh * reg[:, :2, ...] - 0.5 * bwh * torch.exp(reg[:, 2:, ...])
        grid_wh = bwh * torch.exp(reg[:, 2:, ...])
        
        grid_left = grid_topleft[:, [0], ...]
        grid_top = grid_topleft[:, [1], ...]
        grid_width = grid_wh[:, [0], ...]
        grid_height = grid_wh[:, [1], ...]
        
        intervel = torch.linspace(0., 1., self.dcn_kernel).view(
            1, self.dcn_kernel, 1, 1).type_as(reg)
        
        grid_x = grid_left + grid_width * intervel
        grid_x = grid_x.unsqueeze(1).repeat(1, self.dcn_kernel, 1, 1, 1)
        grid_x = grid_x.view(b, -1, h, w)
        
        grid_y = grid_top + grid_height * intervel
        grid_y = grid_y.unsqueeze(2).repeat(1, 1, self.dcn_kernel, 1, 1)
        grid_y = grid_y.view(b, -1, h, w)
        
        grid_yx = torch.stack([grid_y, grid_x], dim=2)
        grid_yx = grid_yx.view(b, -1, h, w)
        
        regressed_bbox = torch.cat([
            grid_left, grid_top, 
            grid_left + grid_width, 
            grid_top + grid_height
        ], 1)
        
        return grid_yx, regressed_bbox

    def get_points(self, featmap_sizes, img_metas_dict, device='cuda'):
        """Get points for all levels in a feature pyramid.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes (h, w).
            img_metas (list[dict]): Image meta info.
            device (str): Device.

        Returns:
            tuple: (mlvl_center_list, mlvl_valid_flag_list)
        """
        mlvl_center_list = []
        mlvl_valid_flag_list = []

        print("img_metas:", img_metas_dict)

        if img_metas_dict is None or len(img_metas_dict) == 0:
            raise ValueError(f"img_metas is {img_metas_dict}. Check if DetDataSample.metainfo was properly set.")
        for i, meta in enumerate(img_metas_dict):
            if 'img_shape' not in meta:
                raise ValueError(f"Missing 'img_shape' in img_meta[{i}]: {meta}")


        for featmap_size, stride in zip(featmap_sizes, self.point_strides):
            h, w = featmap_size
            # Create mesh grid
            y, x = torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=device),
                torch.arange(w, dtype=torch.float32, device=device),
                indexing='ij'
            )
            x = x * stride + stride // 2
            y = y * stride + stride // 2
            points = torch.stack((x, y), dim=-1).view(-1, 2)

            # Repeat for each image in the batch
            center_list = [points.clone() for _ in range(len(img_metas_dict))]
            mlvl_center_list.append(center_list)

            # Valid flags: whether each point lies within the image
            valid_flag_list = []
            for img_meta in img_metas_dict:
                img_h, img_w = img_meta['img_shape'][:2]
                valid_x = ((points[:, 0] >= 0) & (points[:, 0] < img_w))
                valid_y = ((points[:, 1] >= 0) & (points[:, 1] < img_h))
                valid = valid_x & valid_y
                valid_flag_list.append(valid)
            mlvl_valid_flag_list.append(valid_flag_list)

        # Transpose the lists to match structure: list_per_image[level]
        center_list = list(zip(*mlvl_center_list))  # list[num_imgs][num_levels]
        valid_flag_list = list(zip(*mlvl_valid_flag_list))

        return center_list, valid_flag_list

    def loss(self, cls_scores: List[Tensor], pts_preds_init: List[Tensor],
            pts_preds_refine: List[Tensor], batch_gt_instances: InstanceList,
            img_metas_dict, batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        
        """Calculate the loss."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        device = cls_scores[0].device
    
        # Get point centers
        center_list, _ = self.get_points(featmap_sizes, img_metas_dict, device)
    
        # Get targets - make sure we're unpacking the tuple correctly
        (labels_list, label_weights_list, 
         bbox_gt_list_init, candidate_list_init, 
         bbox_weights_list_init, avg_factor_init) = self.get_targets(
            points=center_list,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=img_metas_dict,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            stage='init',
            return_sampling_results=False)
    
        (_, _, 
         bbox_gt_list_refine, _, 
         bbox_weights_list_refine, avg_factor_refine) = self.get_targets(
            points=center_list,
            batch_gt_instances=batch_gt_instances,
            batch_img_metas=img_metas_dict,
            batch_gt_instances_ignore=batch_gt_instances_ignore,
            stage='refine',
            return_sampling_results=False)
    
        # Compute losses
        losses_cls, losses_pts_init, losses_pts_refine = multi_apply(
            self.loss_single,
            cls_scores,
            pts_preds_init,
            pts_preds_refine,
            labels_list,
            label_weights_list,
            bbox_gt_list_init,
            bbox_weights_list_init,
            bbox_gt_list_refine,
            bbox_weights_list_refine,
            self.point_strides,
            avg_factor_init=avg_factor_init,
            avg_factor_refine=avg_factor_refine)
    
        return dict(
            loss_rpn_cls=losses_cls,
            loss_rpn_pts_init=losses_pts_init,
            loss_rpn_pts_refine=losses_pts_refine)


    def loss_single(self, cls_score: Tensor, pts_pred_init: Tensor,
                    pts_pred_refine: Tensor, labels: List[Tensor], label_weights: List[Tensor],
                    bbox_gt_init: List[Tensor], bbox_weights_init: List[Tensor],
                    bbox_gt_refine: List[Tensor], bbox_weights_refine: List[Tensor],
                    stride: int, avg_factor_init: int, avg_factor_refine: int) -> Tuple[Tensor]:
        """Calculate the loss of a single scale level."""
        # Classification loss
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        labels = torch.cat(labels, dim=0).reshape(-1)
        label_weights = torch.cat(label_weights, dim=0).reshape(-1)
        
        valid_idx = (label_weights > 0).nonzero().squeeze(1)
        
        if len(valid_idx) > 0:
            cls_score = cls_score[valid_idx]
            labels = labels[valid_idx]
            label_weights = label_weights[valid_idx]
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights, 
                avg_factor=float(avg_factor_refine))
        else:
            loss_cls = cls_score.sum() * 0

        # Points loss - ensure proper shapes
        bbox_gt_init = torch.cat([b.view(-1, 4) for b in bbox_gt_init], dim=0)
        bbox_weights_init = torch.cat([b.view(-1, 4) for b in bbox_weights_init], dim=0)
        
        pts_pred_init = pts_pred_init.permute(0, 2, 3, 1).reshape(-1, 4 * self.num_points)
        bbox_pred_init = self.points2bbox(pts_pred_init, y_first=False)
        
        # Same for refined points
        bbox_gt_refine = torch.cat([b.view(-1, 4) for b in bbox_gt_refine], dim=0)
        bbox_weights_refine = torch.cat([b.view(-1, 4) for b in bbox_weights_refine], dim=0)
        pts_pred_refine = pts_pred_refine.permute(0, 2, 3, 1).reshape(-1, 4 * self.num_points)
        bbox_pred_refine = self.points2bbox(pts_pred_refine, y_first=False)
        
        normalize_term = self.point_base_scale * stride
        
        loss_pts_init = self.loss_bbox_init(
            bbox_pred_init / normalize_term,
            bbox_gt_init / normalize_term,
            bbox_weights_init,
            avg_factor=float(avg_factor_init))
        
        loss_pts_refine = self.loss_bbox_refine(
            bbox_pred_refine / normalize_term,
            bbox_gt_refine / normalize_term,
            bbox_weights_refine,
            avg_factor=float(avg_factor_refine))
        
        return loss_cls, loss_pts_init, loss_pts_refine

    def get_bboxes(self, cls_scores: List[Tensor], pts_preds_refine: List[Tensor],
                  batch_img_metas: List[dict], cfg: ConfigDict = None,
                  rescale: bool = False, with_nms: bool = True) -> InstanceList:
        """Transform network outputs of a batch into bbox results."""
        assert len(cls_scores) == len(pts_preds_refine)
        cfg = self.test_cfg if cfg is None else cfg
        
        num_levels = len(cls_scores)
        mlvl_priors = [
            self.prior_generator.grid_priors(
                cls_scores[i].shape[-2:],
                self.point_strides[i],
                device=cls_scores[i].device) for i in range(num_levels)
        ]
        
        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(num_levels)
            ]  # Fixed: Added missing closing bracket
            bbox_pred_list = [
                pts_preds_refine[i][img_id].detach() for i in range(num_levels)]
            
            results = self._get_bboxes_single(
                cls_score_list, bbox_pred_list, mlvl_priors, img_meta, cfg,
                rescale, with_nms)
            result_list.append(results)
            
        return result_list

    def _get_bboxes_single(self, cls_score_list: List[Tensor],
                           bbox_pred_list: List[Tensor],
                           mlvl_priors: List[Tensor], img_meta: dict,
                           cfg: ConfigDict, rescale: bool = False,
                           with_nms: bool = True) -> InstanceData:
        """Transform outputs of a single image into bbox predictions."""
        cfg = self.test_cfg if cfg is None else cfg
        img_shape = img_meta['img_shape']
        nms_pre = cfg.get('nms_pre', -1)
        
        mlvl_bboxes = []
        mlvl_scores = []
        
        for level_idx, (cls_score, bbox_pred, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list, mlvl_priors)):
            
            assert cls_score.size()[-2:] == bbox_pred.size()[-2:]
            
            # Reshape predictions
            bbox_pred = bbox_pred.permute(1, 2, 0).reshape(-1, 4 * self.num_points)
            cls_score = cls_score.permute(1, 2, 0).reshape(-1, self.cls_out_channels)
            
            # Apply sigmoid to classification scores
            scores = cls_score.sigmoid().squeeze(1)
            
            # Filter scores and topk
            results = filter_scores_and_topk(
                scores, cfg.score_thr, nms_pre,
                dict(bbox_pred=bbox_pred, priors=priors))
            
            scores, _, _, filtered_results = results
            bbox_pred = filtered_results['bbox_pred']
            priors = filtered_results['priors']
            
            # Decode bbox predictions
            bboxes = self._bbox_decode(priors, bbox_pred, 
                                      self.point_strides[level_idx],
                                      img_shape)
            
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
        
        # Combine predictions from all levels
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        
        # Prepare results
        results = InstanceData()
        results.bboxes = HorizontalBoxes(mlvl_bboxes)
        results.scores = mlvl_scores
        
        # Post-processing (NMS)
        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)

    def _bbox_decode(self, points: Tensor, bbox_pred: Tensor, stride: int,
                     max_shape: Tuple[int, int]) -> Tensor:
        """Decode point predictions to bounding boxes.
        
        Args:
            points (Tensor): Prior points, shape (N, 2).
            bbox_pred (Tensor): Predicted bbox distances, shape (N, num_points*4).
            stride (int): Point stride.
            max_shape (Tuple[int, int]): Image shape (height, width).
            
        Returns:
            Tensor: Decoded bounding boxes, shape (N, 4).
        """
        # Reshape to (N, num_points, 4)
        pred_distances = bbox_pred.view(-1, self.num_points, 4)
        
        # Get centers from points
        x_centers = points[:, 0].unsqueeze(1)  # (N, 1)
        y_centers = points[:, 1].unsqueeze(1)  # (N, 1)
        
        # Extract distances
        left = pred_distances[:, :, 0] * stride
        top = pred_distances[:, :, 1] * stride
        right = pred_distances[:, :, 2] * stride
        bottom = pred_distances[:, :, 3] * stride
        
        # Convert distances to point coordinates
        x1 = x_centers - left
        y1 = y_centers - top
        x2 = x_centers + right
        y2 = y_centers + bottom
        
        # For each sample, take min/max coordinates among all points
        x_min, _ = x1.min(dim=1)
        y_min, _ = y1.min(dim=1)
        x_max, _ = x2.max(dim=1)
        y_max, _ = y2.max(dim=1)
        
        # Clamp to image boundaries
        x_min = x_min.clamp(min=0, max=max_shape[1])
        y_min = y_min.clamp(min=0, max=max_shape[0])
        x_max = x_max.clamp(min=0, max=max_shape[1])
        y_max = y_max.clamp(min=0, max=max_shape[0])
        
        decoded_bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=-1)
        return decoded_bboxes

    def get_proposals(self, cls_scores: List[Tensor], pts_preds_refine: List[Tensor],
                      batch_img_metas: List[dict]) -> InstanceList:
        """Get proposals during training.
        
        Args:
            cls_scores (list[Tensor]): Classification scores for each level.
            pts_preds_refine (list[Tensor]): Refined points predictions.
            batch_img_metas (list[dict]): Meta info of each image.
            
        Returns:
            list[:obj:`InstanceData`]: Proposal results of each image.
        """
        return self.get_bboxes(cls_scores, pts_preds_refine, batch_img_metas)
