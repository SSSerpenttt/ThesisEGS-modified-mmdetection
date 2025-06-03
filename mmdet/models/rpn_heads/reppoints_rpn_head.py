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
                transform_method: str = 'minmax',
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
        self.use_grid_points = True  # Always False for RPN
        self.center_init = False  # Always True for RPN
        self.point_strides = point_strides
        
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
        
        self.prior_generator = MlvlPointGenerator(self.point_strides, offset=0.)

        self.assigner = TASK_UTILS.build(dict(type='mmdet.MaxIoUAssigner',
                                    pos_iou_thr=0.1,
                                    neg_iou_thr=0.05,
                                    min_pos_iou=0,
                                    ignore_iof_thr=-1,
                                    match_low_quality=True,
                                    iou_calculator=dict(type='mmdet.BboxOverlaps2D')))
        self.sampler = PseudoSampler()  # Or another sampler if you prefer
        
        self._init_layers()
        
        self.gradient_mul = gradient_mul
        self.point_base_scale = point_base_scale
        self.point_strides = point_strides
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
        # print("In: get_targets")
        # print(f"[get_targets] stage: {stage}, num images: {len(points)}")

        # Normalize image meta info
        img_metas = [{
            'img_shape': meta['img_shape'],
            'scale_factor': meta['scale_factor'],
            'batch_input_shape': meta['img_shape']
        } for meta in batch_img_metas]

        num_images = len(points)
        num_levels = len(points[0])
        # Transpose points to per image list of per-level points
        points_per_image = [[points[i][lvl] for lvl in range(num_levels)] for i in range(num_images)]

        if batch_gt_instances_ignore is None:
            batch_gt_instances_ignore = [None] * num_images

        # Compute targets per image using multi_apply
        results = multi_apply(
            self._get_targets_single,
            points_per_image,
            batch_gt_instances,
            img_metas,
            batch_gt_instances_ignore,
            stage=stage
        )

        # Unpack lists of per-image targets
        labels_list, label_weights_list, bbox_gt_list, points_list, bbox_weights_list, pos_counts = results

        # Flatten all per-image per-level lists into single lists per type
        labels = []
        label_weights = []
        bbox_gt = []
        bbox_weights = []
        points_flat = []

        for i in range(num_images):
            for lvl in range(num_levels):
                labels.append(labels_list[i][lvl])
                label_weights.append(label_weights_list[i][lvl])
                bbox_gt.append(bbox_gt_list[i][lvl])
                bbox_weights.append(bbox_weights_list[i][lvl])
                points_flat.append(points_list[i][lvl])

        # Concatenate all per-level tensors for all images
        labels = torch.cat(labels, dim=0)
        label_weights = torch.cat(label_weights, dim=0)
        # print("[get_targets] label_weights sum:", label_weights.sum().item(), "shape:", label_weights.shape)
        bbox_gt = torch.cat(bbox_gt, dim=0)
        bbox_weights = torch.cat(bbox_weights, dim=0)
        points_flat = torch.cat(points_flat, dim=0)

        avg_factor = sum(pos_counts)
        # print(f"[get_targets] avg_factor: {avg_factor}, num pos: {[p for p in pos_counts]}")
        if return_sampling_results:
            # Optionally return sampling results if needed downstream
            return labels, label_weights, bbox_gt, points_flat, bbox_weights, avg_factor, results[-1]

        return labels, label_weights, bbox_gt, points_flat, bbox_weights, avg_factor


    def _get_targets_single(self,
                            points: List[Tensor],
                            gt_instances: InstanceData,
                            img_meta: dict,
                            gt_instances_ignore: Optional[InstanceData] = None,
                            stage: str = 'init') -> tuple:
        """Compute targets for a single image."""
        # print(f"[get_targets_single] stage: {stage}, num points: {sum([len(p) for p in points])}")

        # Convert points to bboxes
        bboxes = []
        for points_per_level in points:
            bboxes_per_level = self.points2bbox(points_per_level.unsqueeze(0))
            if bboxes_per_level.dim() == 4:
                bboxes_per_level = bboxes_per_level.view(-1, 4)
            bboxes.append(bboxes_per_level.squeeze(0))
        bboxes = torch.cat(bboxes, dim=0)

        # print("First 5 prior bboxes:", bboxes[:5].cpu().numpy())
        # print("All GT bboxes:", gt_instances.bboxes.cpu().numpy())

        ious = bbox_overlaps(bboxes, gt_instances.bboxes)
        # print("IoU stats: min", ious.min().item(), "max", ious.max().item(), "mean", ious.mean().item())
        
        # print("[RPN _get_targets_single] GT bboxes (first 5):", gt_instances.bboxes[:5])
        # print("[RPN _get_targets_single] GT labels (first 5):", gt_instances.labels[:5])
        
        # if hasattr(gt_instances, 'masks'):
        #     print("[RPN _get_targets_single] GT masks type:", type(gt_instances.masks))

        pred_instances = InstanceData(priors=bboxes)

        # Validate gt_instances bboxes and labels
        if not hasattr(gt_instances, 'bboxes'):
            raise ValueError("gt_instances must have 'bboxes' attribute.")
        gt_bboxes = gt_instances.bboxes
        if gt_bboxes.dim() == 3:
            gt_instances.bboxes = gt_bboxes.squeeze(0)

        if not hasattr(gt_instances, 'labels'):
            raise AttributeError("gt_instances missing 'labels' attribute.")

        assign_result = self.assigner.assign(
            pred_instances=pred_instances,
            gt_instances=gt_instances,
            gt_instances_ignore=gt_instances_ignore)
        
        # print("[_get_targets_single] assign_result.num_gts:", assign_result.num_gts)
        # print("[_get_targets_single] assign_result.gt_inds (first 10):", assign_result.gt_inds[:10])

        sampling_result = self.sampler.sample(
            assign_result=assign_result,
            pred_instances=pred_instances,
            gt_instances=gt_instances)

        num_points = bboxes.shape[0]
        labels = bboxes.new_full((num_points,), self.num_classes, dtype=torch.long)
        label_weights = bboxes.new_zeros(num_points, dtype=torch.float)
        bbox_gt = bboxes.new_zeros((num_points, 4), dtype=torch.float)
        bbox_weights = bboxes.new_zeros((num_points, 4), dtype=torch.float)

        pos_inds = sampling_result.pos_inds
        neg_inds = sampling_result.neg_inds

        # print("[RPN _get_targets_single] Num pos_inds:", len(pos_inds), "Num neg_inds:", len(neg_inds))

        if len(pos_inds) > 0:
            # print("Assigning GT boxes to positive priors:")
            # print("pos_inds:", pos_inds)
            # print("sampling_result.pos_gt_bboxes:", sampling_result.pos_gt_bboxes)
            labels[pos_inds] = 0  # For RPN binary classification
            pos_gt_bboxes = sampling_result.pos_gt_bboxes
            if hasattr(pos_gt_bboxes, "tensor"):
                pos_gt_bboxes = pos_gt_bboxes.tensor
            bbox_gt[pos_inds, :] = pos_gt_bboxes
            bbox_weights[pos_inds, :] = 1.0
            label_weights[pos_inds] = 1.0

        if len(neg_inds) > 0:
            label_weights[neg_inds] = 1.0

        # Split targets back per feature level based on points length for each level
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

        # print("[RPN _get_targets_single] GT bboxes (first 5):", gt_instances.bboxes[:5])
        # print("[RPN _get_targets_single] GT labels (first 5):", gt_instances.labels[:5])
        # if hasattr(gt_instances, 'masks'):
        #     print("[RPN _get_targets_single] GT masks type:", type(gt_instances.masks))
        # print("[RPN _get_targets_single] Num pos_inds:", len(pos_inds), "Num neg_inds:", len(neg_inds))

        return (labels_list, label_weights_list, bbox_gt_list, points, bbox_weights_list, len(pos_inds))


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

    def forward(self, feats: Tuple[Tensor]) -> Union[
            Tuple[List[Tensor], List[Tensor], List[Tensor]],
            Tuple[List[Tensor], List[Tensor]]]:
        
        # for i, f in enumerate(feats):
            # print(f"Input feat[{i}]: shape={f.shape}")

        outputs = multi_apply(self.forward_single, feats)

        # print("[RepPointsRPNHead] Received feature maps:", [f.shape for f in feats])
        # print("[RepPointsRPNHead] Configured point_strides:", self.point_strides)

        if self.training:
            cls_scores, pts_preds_init, pts_preds_refine = outputs
            # print("RPN training mode")
            return cls_scores, pts_preds_init, pts_preds_refine
        else:
            cls_scores, bbox_preds = outputs
            # print("RPN inference mode")
            return cls_scores, bbox_preds


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
            # For inference, return classification scores and raw bbox predictions
            return cls_out, pts_out_refine


    def points2bbox(self, pts: Tensor, y_first: bool = True) -> Tensor:
        """Convert a set of points to bounding boxes.

        Args:
            pts (Tensor): Input points tensor, shape can be (N, num_points*2) or (batch_size, N, num_points*2), etc.
            y_first (bool): Whether the first coordinate is y (True) or x (False).

        Returns:
            Tensor: Bounding boxes in (x1, y1, x2, y2) format, shape (num_bboxes, 4).
        """
        # print(f"Input pts shape: {pts.shape}")

        # Normalize input dimensions: expecting something like (batch_size, num_points*2, 1) or (batch_size, num_points, 2)
        if pts.dim() == 2:
            # Reshape (N, num_points*2) -> (1, N, num_points*2, 1) for consistent processing
            pts = pts.view(1, pts.shape[0], pts.shape[1], 1)
        elif pts.dim() == 3:
            # Add trailing dimension for uniform shape (batch_size, N, num_points*2, 1)
            pts = pts.unsqueeze(-1)

        # Now reshape to separate x and y coords: (batch, N, 2, ...)
        pts_reshape = pts.view(pts.shape[0], -1, 2, *pts.shape[2:])
        # print(f"Reshaped pts shape: {pts_reshape.shape}")

        # Extract y and x depending on y_first
        pts_y = pts_reshape[:, :, 0, ...] if y_first else pts_reshape[:, :, 1, ...]
        pts_x = pts_reshape[:, :, 1, ...] if y_first else pts_reshape[:, :, 0, ...]

        if self.transform_method == 'minmax':
            bbox_left = pts_x.min(dim=1, keepdim=True)[0]
            bbox_right = pts_x.max(dim=1, keepdim=True)[0]
            bbox_up = pts_y.min(dim=1, keepdim=True)[0]
            bbox_bottom = pts_y.max(dim=1, keepdim=True)[0]
            bbox = torch.cat([bbox_left, bbox_up, bbox_right, bbox_bottom], dim=1)

        elif self.transform_method == 'partial_minmax':
            # Use only first 4 points to compute bbox
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
                pts_x_mean - half_width,
                pts_y_mean - half_height,
                pts_x_mean + half_width,
                pts_y_mean + half_height
            ], dim=1)
        else:
            raise NotImplementedError(f"Unknown transform method: {self.transform_method}")

        # print(f"Raw bbox shape: {bbox.shape}")

        # Reshape bbox to (N, 4)
        if bbox.dim() == 4:
            # (batch, something, something, 4) -> flatten to (N, 4)
            bbox = bbox.permute(0, 2, 3, 1).contiguous().view(-1, 4)
        elif bbox.dim() == 3:
            # (batch, something, 4) -> flatten
            bbox = bbox.permute(0, 2, 1).contiguous().view(-1, 4)
        elif bbox.dim() == 2:
            if bbox.size(-1) != 4:
                bbox = bbox.view(-1, 4)
        else:
            raise ValueError(f"Unexpected bbox dimension: {bbox.dim()}")

        # print(f"Final bbox shape: {bbox.shape}")
        return bbox


    def gen_grid_from_reg(self, reg: torch.Tensor, previous_boxes: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Generate grid from regression values and previous boxes.
        
        Args:
            reg (Tensor): regression output of shape (B, 2*N, H, W), where N = number of points
            previous_boxes (Tensor): boxes of shape (B, 4, H, W), format: [x1, y1, x2, y2]
            
        Returns:
            grid_yx (Tensor): generated grid points (B, 2*N*dcn_kernel, H, W)
            regressed_bbox (Tensor): regressed bounding boxes (B, 4, H, W)
        """
        b, c, h, w = reg.shape
        N = c // 2  # number of points
        
        # Center (x, y) of previous boxes, shape (B, 2, H, W)
        bxy = (previous_boxes[:, :2, ...] + previous_boxes[:, 2:, ...]) / 2.0  # (B, 2, H, W)
        # Width and height of previous boxes, shape (B, 2, H, W)
        bwh = (previous_boxes[:, 2:, ...] - previous_boxes[:, :2, ...]).clamp(min=1e-6)  # (B, 2, H, W)
        
        # Expand bxy and bwh to shape (B, 2, N, H, W) for each point
        bxy_exp = bxy.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, 2, N, H, W)
        bwh_exp = bwh.unsqueeze(2).expand(-1, -1, N, -1, -1)  # (B, 2, N, H, W)
        
        # Split reg into x and y: (B, N, H, W)
        reg_x = reg[:, :N, ...]  # (B, N, H, W)
        reg_y = reg[:, N:, ...]  # (B, N, H, W)
        
        # Stack reg_x and reg_y to (B, 2, N, H, W)
        reg_offsets = torch.stack([reg_x, reg_y], dim=1)  # (B, 2, N, H, W)
        
        # Calculate offset points relative to box center with box width/height scaling
        # reg_offsets assumed normalized offsets in [-0.5, 0.5] or similar, scaled by box size
        points = bxy_exp + reg_offsets * bwh_exp  # (B, 2, N, H, W)
        
        # Prepare interpolation interval for dcn_kernel subdivisions [0, 1]
        interval = torch.linspace(0., 1., self.dcn_kernel, device=reg.device, dtype=reg.dtype)  # (dcn_kernel,)
        
        # Interpolate points along x and y for each point
        # points shape: (B, 2, N, H, W)
        # We want to generate finer grid for each point: (B, 2, N, dcn_kernel, H, W)
        points = points.unsqueeze(3)  # (B, 2, N, 1, H, W)
        
        # Interpolate along x dimension (dim=3), and y dimension (dim=1)
        # For grid_x: interpolate between points_x and points_x + some step - but here we only have single points.
        # Usually, interpolation needs start and end points, but your original code interpolates between left and left+width for each point.
        # We assume you want to generate a grid around each point in local coordinate, so we create a small grid around each point.
        # Here, for simplicity, we tile points along the new dcn_kernel dimension (no actual interpolation, because only one point)
        points_interp = points.expand(-1, -1, -1, self.dcn_kernel, -1, -1)  # (B, 2, N, dcn_kernel, H, W)
        
        # Rearrange to (B, 2 * N * dcn_kernel, H, W)
        points_interp = points_interp.permute(0, 2, 3, 1, 4, 5)  # (B, N, dcn_kernel, 2, H, W)
        points_interp = points_interp.reshape(b, N * self.dcn_kernel * 2, h, w)  # (B, 2*N*dcn_kernel, H, W)
        
        # Now separate y and x channels for stacking in (y, x) order if needed
        # If your model expects [y1, x1, y2, x2, ...], reorder here:
        # Currently channels are [x1, y1, x2, y2, ...] — swap pairs:
        x = points_interp[:, 0::2, :, :]
        y = points_interp[:, 1::2, :, :]
        grid_yx = torch.stack([y, x], dim=2).reshape(b, -1, h, w)
        
        # For simplicity, return original boxes as regressed bbox
        regressed_bbox = previous_boxes.clone()
        
        return grid_yx, regressed_bbox



    def convert_results_to_img_meta(results):
        img_meta = {
            'img_shape': results['img_shape'],           # required for valid_flag
            'ori_shape': results['ori_shape'],           # optional, for scaling back
            'scale_factor': results['scale_factor'],     # optional, useful for post-processing
            'img_id': results['img_id'],                 # optional
            'img_info': results['img_info'],             # optional
        }

        # Debug print statement
        # print(f"[DEBUG] Converted img_meta for img_id {img_meta.get('img_id', 'N/A')}:\n{img_meta}\n")

        return img_meta

    def get_points(self, featmap_sizes, img_metas_dict, device='cuda'):
        """Get points for all levels using the RepPoints prior generator."""
        mlvl_points = self.prior_generator.grid_priors(featmap_sizes, device=device)

        # for lvl, p in enumerate(mlvl_points):
        #     print(f"[RepPointsRPNHead] Level {lvl} points shape: {p.shape}, stride: {self.point_strides[lvl]}")
       
        # mlvl_points is a list of tensors, one per level, shape (num_points, 2)
        # Repeat for each image
        center_list = [[p.clone() for p in mlvl_points] for _ in range(len(img_metas_dict))]
        valid_flag_list = [[torch.ones(p.size(0), dtype=torch.bool, device=device) for p in mlvl_points] for _ in range(len(img_metas_dict))]
        return center_list, valid_flag_list


    def loss(self, cls_scores: List[Tensor], pts_preds_init: List[Tensor],
            pts_preds_refine: List[Tensor], batch_gt_instances: InstanceList,
            img_metas_dict, batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        
        """Calculate the loss."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
        # print("[DEBUG] Feature map sizes:", featmap_sizes)
        device = cls_scores[0].device

        # Get point centers
        center_list, _ = self.get_points(featmap_sizes, img_metas_dict, device)

        # Get targets
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

        # Convert avg_factor lists to scalars
        if isinstance(avg_factor_init, int):
            # Already an int, no need to sum
            avg_factor_init_sum = avg_factor_init
        else:
            # Assume iterable
            avg_factor_init_sum = sum(avg_factor_init)

        if isinstance(avg_factor_refine, int):
            # Already an int, no need to sum
            avg_factor_refine_sum = avg_factor_refine
        else:
            # Assume iterable
            avg_factor_refine_sum = sum(avg_factor_refine)

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
            avg_factor_init=avg_factor_init_sum,
            avg_factor_refine=avg_factor_refine_sum)

        # print(f"[loss] cls_scores lens: {len(cls_scores)}, pts_preds_init lens: {len(pts_preds_init)}, pts_preds_refine lens: {len(pts_preds_refine)}")
        # print(f"[loss] avg_factor_init_sum: {avg_factor_init_sum}, avg_factor_refine_sum: {avg_factor_refine_sum}")
        return dict(
            loss_rpn_cls=losses_cls,
            loss_rpn_pts_init=losses_pts_init,
            loss_rpn_pts_refine=losses_pts_refine)


    def loss_single(self, cls_score: Tensor, pts_pred_init: Tensor,
                    pts_pred_refine: Tensor, labels: List[Tensor], label_weights: List[Tensor],
                    gt_bboxes_init: List[Tensor], gt_weights_init: List[Tensor],
                    gt_bboxes_refine: List[Tensor], gt_weights_refine: List[Tensor],
                    stride: int, avg_factor_init: int, avg_factor_refine: int
                    ) -> Tuple[Tensor, Tensor, Tensor]:
        
        # print(f"[loss_single] stride for this level: {stride}")

        # print("[loss_single] labels shape:", labels.shape)
        # print("[loss_single] label_weights shape:", label_weights.shape)

        if isinstance(labels, (list, tuple)):
            labels = torch.cat(labels, dim=0)
        if isinstance(label_weights, (list, tuple)):
            label_weights = torch.cat(label_weights, dim=0)

        # Only print if labels is not empty and has at least 1 dimension
        # if labels.numel() > 0 and labels.dim() > 0:
        #     # print("[loss_single] labels (first 10):", labels[:10].cpu().numpy())
        # else:
        #     print("[loss_single] labels: EMPTY or scalar", labels.item() if labels.numel() == 1 else labels)


        # ----- Classification Loss -----
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)
        labels = torch.cat(labels, dim=0).reshape(-1) if isinstance(labels, (list, tuple)) else labels.reshape(-1)
        label_weights = torch.cat(label_weights, dim=0).reshape(-1) if isinstance(label_weights, (list, tuple)) else label_weights.reshape(-1)
        valid_class_idx = (label_weights > 0).nonzero(as_tuple=False).view(-1)

        # print("[loss_single] valid_class_idx:", valid_class_idx.shape, "num valid:", valid_class_idx.numel())

        if valid_class_idx.numel() > 0:
            cls_score = cls_score[valid_class_idx]
            labels = labels[valid_class_idx]
            label_weights = label_weights[valid_class_idx]
            loss_cls = self.loss_cls(
                cls_score, labels, label_weights,
                avg_factor=float(avg_factor_refine)
            )
        else:
            loss_cls = cls_score.sum() * 0.0  # Avoid NaN in loss

        # ----- Bounding Box Loss -----
        gt_bboxes_init = torch.cat(gt_bboxes_init, dim=0) if isinstance(gt_bboxes_init, (list, tuple)) else gt_bboxes_init
        gt_weights_init = torch.cat(gt_weights_init, dim=0) if isinstance(gt_weights_init, (list, tuple)) else gt_weights_init
        gt_bboxes_refine = torch.cat(gt_bboxes_refine, dim=0) if isinstance(gt_bboxes_refine, (list, tuple)) else gt_bboxes_refine
        gt_weights_refine = torch.cat(gt_weights_refine, dim=0) if isinstance(gt_weights_refine, (list, tuple)) else gt_weights_refine

        pred_bboxes_init = self.points2bbox(pts_pred_init)
        pred_bboxes_refine = self.points2bbox(pts_pred_refine)

        # Align lengths
        min_len_init = min(pred_bboxes_init.size(0), gt_bboxes_init.size(0), gt_weights_init.size(0))
        pred_bboxes_init = pred_bboxes_init[:min_len_init]
        gt_bboxes_init = gt_bboxes_init[:min_len_init]
        gt_weights_init = gt_weights_init[:min_len_init]

        min_len_refine = min(pred_bboxes_refine.size(0), gt_bboxes_refine.size(0), gt_weights_refine.size(0))
        pred_bboxes_refine = pred_bboxes_refine[:min_len_refine]
        gt_bboxes_refine = gt_bboxes_refine[:min_len_refine]
        gt_weights_refine = gt_weights_refine[:min_len_refine]

        # Filter valid samples
        valid_init = gt_weights_init.sum(dim=1) > 0 if gt_weights_init.dim() > 1 else gt_weights_init > 0
        pred_bboxes_init = pred_bboxes_init[valid_init]
        gt_bboxes_init = gt_bboxes_init[valid_init]
        gt_weights_init = gt_weights_init[valid_init]

        valid_refine = gt_weights_refine.sum(dim=1) > 0 if gt_weights_refine.dim() > 1 else gt_weights_refine > 0
        pred_bboxes_refine = pred_bboxes_refine[valid_refine]
        gt_bboxes_refine = gt_bboxes_refine[valid_refine]
        gt_weights_refine = gt_weights_refine[valid_refine]

        normalize_factor = self.point_base_scale * stride

        # Ensure shapes match for loss computation
        if gt_bboxes_init.dim() == 1:
            gt_bboxes_init = gt_bboxes_init.unsqueeze(0)
        if pred_bboxes_init.dim() == 1:
            pred_bboxes_init = pred_bboxes_init.unsqueeze(0)
        if gt_bboxes_refine.dim() == 1:
            gt_bboxes_refine = gt_bboxes_refine.unsqueeze(0)
        if pred_bboxes_refine.dim() == 1:
            pred_bboxes_refine = pred_bboxes_refine.unsqueeze(0)

        # --- Initial points loss ---
        goto_skip_init = False
        if pred_bboxes_init.size(0) != gt_bboxes_init.size(0):
            if gt_bboxes_init.size(0) == 1 and pred_bboxes_init.size(0) > 1:
                gt_bboxes_init = gt_bboxes_init.expand_as(pred_bboxes_init)
                gt_weights_init = gt_weights_init.expand_as(pred_bboxes_init)
            else:
                loss_pts_init = pred_bboxes_init.sum() * 0.0
                goto_skip_init = True
        if not goto_skip_init:
            loss_pts_init = self.loss_bbox_init(
                pred_bboxes_init / normalize_factor,
                gt_bboxes_init / normalize_factor,
                gt_weights_init,
                avg_factor=float(avg_factor_init)
            )

        # --- Refined points loss ---
        goto_skip_refine = False
        if pred_bboxes_refine.size(0) != gt_bboxes_refine.size(0):
            if gt_bboxes_refine.size(0) == 1 and pred_bboxes_refine.size(0) > 1:
                gt_bboxes_refine = gt_bboxes_refine.expand_as(pred_bboxes_refine)
                gt_weights_refine = gt_weights_refine.expand_as(pred_bboxes_refine)
            else:
                loss_pts_refine = pred_bboxes_refine.sum() * 0.0
                goto_skip_refine = True
        if not goto_skip_refine:
            loss_pts_refine = self.loss_bbox_refine(
                pred_bboxes_refine / normalize_factor,
                gt_bboxes_refine / normalize_factor,
                gt_weights_refine,
                avg_factor=float(avg_factor_refine)
            )

        # print(f"[loss_single] labels shape: {labels.shape}, label_weights shape: {label_weights.shape}")
        # print(f"[loss_single] valid_class_idx: {valid_class_idx.shape}, num valid: {valid_class_idx.numel()}")
        # print(f"[loss_single] GT boxes (init): {gt_bboxes_init.shape}, GT boxes (refine): {gt_bboxes_refine.shape}")
        # print(f"[loss_single] pred_bboxes_init: {pred_bboxes_init.shape}, pred_bboxes_refine: {pred_bboxes_refine.shape}")
        # print(f"[loss_single] gt_weights_init sum: {gt_weights_init.sum().item()}, gt_weights_refine sum: {gt_weights_refine.sum().item()}")
        # print(f"[loss_single] valid_init: {valid_init.sum().item()}, valid_refine: {valid_refine.sum().item()}")

        return loss_cls, loss_pts_init, loss_pts_refine


    def get_bboxes(self, cls_scores: List[Tensor], pts_preds_refine: List[Tensor],
                batch_img_metas: List[dict], cfg: ConfigDict = None,
                rescale: bool = False, with_nms: bool = True) -> InstanceList:
        # print("DEBUG: get_bboxes called")
        # print("DEBUG: len(cls_scores) =", len(cls_scores))
        # print("DEBUG: len(pts_preds_refine) =", len(pts_preds_refine))
        """Transform network outputs of a batch into bbox results."""
        assert len(cls_scores) == len(pts_preds_refine)
        cfg = self.test_cfg if cfg is None else cfg

        # FIX: Get all feature map sizes and call grid_priors ONCE
        featmap_sizes = [cs.shape[-2:] for cs in cls_scores]
        mlvl_priors = self.prior_generator.grid_priors(
            featmap_sizes,
            device=cls_scores[0].device
        )

        result_list = []
        for img_id in range(len(batch_img_metas)):
            img_meta = batch_img_metas[img_id]
            # Fix: if img_meta is a list, get the first element
            if isinstance(img_meta, list):
                img_meta = img_meta[0]
            cls_score_list = [
                cls_scores[i][img_id].detach() for i in range(len(cls_scores))
            ]
            bbox_pred_list = [
                pts_preds_refine[i][img_id].detach() for i in range(len(pts_preds_refine))]
            
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
        if hasattr(img_meta, 'metainfo'):
            img_shape = img_meta.metainfo.get('img_shape', None)
        else:
            img_shape = img_meta.get('img_shape', None)
        nms_pre = cfg.get('nms_pre', -1)
        
        mlvl_bboxes = []
        mlvl_scores = []
        
        for level_idx, (cls_score, bbox_pred, priors) in enumerate(
                zip(cls_score_list, bbox_pred_list, mlvl_priors)):

            # print(f"[DEBUG] Level {level_idx}: cls_score.shape={cls_score.shape}, bbox_pred.shape={bbox_pred.shape}")
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

            # Robust unpacking for RPN (single-class) and multi-class
            if isinstance(results, tuple) and len(results) == 4:
                # RPN/single-class: (scores, keep_idxs, _, filtered_results)
                scores, keep_idxs, _, filtered_results = results
                bbox_pred = filtered_results['bbox_pred']
                priors = filtered_results['priors']
                labels = torch.zeros_like(scores, dtype=torch.long)
            elif isinstance(results, tuple) and len(results) == 5:
                # Multi-class: (scores, labels, keep_idxs, filtered_results, ...)
                scores, labels, keep_idxs, filtered_results, _ = results
                bbox_pred = filtered_results['bbox_pred']
                priors = filtered_results['priors']
            else:
                # Multi-class: (scores, labels, keep_idxs, filtered_results)
                scores, labels, keep_idxs, filtered_results = results
                bbox_pred = filtered_results['bbox_pred']
                priors = filtered_results['priors']
            
            # Decode bbox predictions
            bboxes = self._bbox_decode(priors, bbox_pred, 
                                    self.point_strides[level_idx],
                                    img_shape)
            
            mlvl_bboxes.append(bboxes)
            mlvl_scores.append(scores)
            if 'labels' not in locals():
                labels = torch.zeros_like(scores, dtype=torch.long)
            if level_idx == 0:
                mlvl_labels = labels
            else:
                mlvl_labels = torch.cat([mlvl_labels, labels], dim=0)
        
        # Combine predictions from all levels
        mlvl_bboxes = torch.cat(mlvl_bboxes)
        mlvl_scores = torch.cat(mlvl_scores)
        mlvl_labels = mlvl_labels if 'mlvl_labels' in locals() else torch.zeros_like(mlvl_scores, dtype=torch.long)

        # print("DEBUG: mlvl_bboxes shape:", mlvl_bboxes.shape)
        # print("DEBUG: mlvl_scores shape:", mlvl_scores.shape)
        
        # Prepare results
        results = InstanceData()
        results.bboxes = HorizontalBoxes(mlvl_bboxes)
        results.scores = mlvl_scores
        results.labels = mlvl_labels
        
        # Post-processing (NMS)
        return self._bbox_post_process(
            results=results,
            cfg=cfg,
            rescale=rescale,
            with_nms=with_nms,
            img_meta=img_meta)


    def _bbox_decode(self, points: Tensor, bbox_pred: Tensor, stride: int,
                     max_shape: Tuple[int, int]) -> Tensor:
        """Decode point predictions to bounding boxes."""
        pred_distances = bbox_pred.view(-1, self.num_points, 4)
        
        # ✅ Clamp distances to avoid exploding predictions
        max_dist = max(max_shape) / stride
        pred_distances = torch.clamp(pred_distances, min=0, max=max_dist)
    
        x_centers = points[:, 0].unsqueeze(1)
        y_centers = points[:, 1].unsqueeze(1)
    
        left   = pred_distances[:, :, 0] * stride
        top    = pred_distances[:, :, 1] * stride
        right  = pred_distances[:, :, 2] * stride
        bottom = pred_distances[:, :, 3] * stride
    
        x1 = x_centers - left
        y1 = y_centers - top
        x2 = x_centers + right
        y2 = y_centers + bottom
    
        x_min, _ = x1.min(dim=1)
        y_min, _ = y1.min(dim=1)
        x_max, _ = x2.max(dim=1)
        y_max, _ = y2.max(dim=1)
    
        x_min = x_min.clamp(min=0, max=max_shape[1])
        y_min = y_min.clamp(min=0, max=max_shape[0])
        x_max = x_max.clamp(min=0, max=max_shape[1])
        y_max = y_max.clamp(min=0, max=max_shape[0])

        if torch.any((x_max - x_min) > max_shape[1] * 0.9):
            print("[!] Detected extremely wide box:", decoded_bboxes)
        if torch.any((y_max - y_min) > max_shape[0] * 0.9):
            print("[!] Detected extremely tall box:", decoded_bboxes)
    
        decoded_bboxes = torch.stack([x_min, y_min, x_max, y_max], dim=1)
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
