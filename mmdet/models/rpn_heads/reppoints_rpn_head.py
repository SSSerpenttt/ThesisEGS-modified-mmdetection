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
                                     pos_iou_thr=0.4,
                                     neg_iou_thr=0.3,
                                     min_pos_iou=0.2,
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
        # print("In: get_targets")
        print(f"[get_targets] stage: {stage}, num images: {len(points)}")

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
        bbox_gt = torch.cat(bbox_gt, dim=0)
        bbox_weights = torch.cat(bbox_weights, dim=0)
        points_flat = torch.cat(points_flat, dim=0)

        avg_factor = sum(pos_counts)
        print(f"[get_targets] avg_factor: {avg_factor}, num pos: {[p for p in pos_counts]}")
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
        print(f"[get_targets_single] stage: {stage}, num points: {sum([len(p) for p in points])}")

        # Convert points to bboxes
        bboxes = []
        for points_per_level in points:
            bboxes_per_level = self.points2bbox(points_per_level.unsqueeze(0))
            if bboxes_per_level.dim() == 4:
                bboxes_per_level = bboxes_per_level.view(-1, 4)
            bboxes.append(bboxes_per_level.squeeze(0))
        bboxes = torch.cat(bboxes, dim=0)

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

        if len(pos_inds) > 0:
            labels[pos_inds] = 0  # For RPN binary classification
            bbox_gt[pos_inds, :] = sampling_result.pos_gt_bboxes.tensor
            bbox_weights[pos_inds, :] = 1.0

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

        print(f"[get_targets_single] GT bboxes: {gt_instances.bboxes.shape if hasattr(gt_instances, 'bboxes') else None}")
        print(f"[get_targets_single] pos_inds: {len(pos_inds)}, neg_inds: {len(neg_inds)}")
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
            # For inference, return classification scores and bbox predictions
            return cls_out, self.points2bbox(pts_out_refine)


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
        """
        Get points for all levels in a feature pyramid.

        Args:
            featmap_sizes (list[tuple]): List of feature map sizes (h, w) for each FPN level.
            img_metas_dict (list[dict]): List of image metadata dictionaries.
            device (str): Device to create the tensors on.

        Returns:
            tuple: (center_list, valid_flag_list)
                - center_list: List[num_imgs][num_levels] of center points.
                - valid_flag_list: List[num_imgs][num_levels] of bool masks for valid locations.
        """

        if not img_metas_dict:
            raise ValueError("img_metas_dict is empty or None. Make sure 'metainfo' is correctly set in DetDataSample.")

        mlvl_center_list = []
        mlvl_valid_flag_list = []

        for level_idx, (featmap_size, stride) in enumerate(zip(featmap_sizes, self.point_strides)):
            h, w = featmap_size

            # Create mesh grid of shape (h, w)
            y, x = torch.meshgrid(
                torch.arange(h, dtype=torch.float32, device=device),
                torch.arange(w, dtype=torch.float32, device=device),
                indexing='ij'
            )

            # Convert to absolute pixel coordinates (center of each grid cell)
            x = x * stride + stride // 2
            y = y * stride + stride // 2

            points = torch.stack((x, y), dim=-1).view(-1, 2)  # (h*w, 2)

            # Repeat for each image
            center_list = [points.clone() for _ in range(len(img_metas_dict))]
            mlvl_center_list.append(center_list)

            # Compute valid flags for each image
            valid_flag_list = []
            for meta in img_metas_dict:
                if 'img_shape' not in meta:
                    raise ValueError(f"Missing 'img_shape' in image metadata: {meta}")

                img_h, img_w = meta['img_shape'][:2]
                valid_x = (points[:, 0] >= 0) & (points[:, 0] < img_w)
                valid_y = (points[:, 1] >= 0) & (points[:, 1] < img_h)
                valid = valid_x & valid_y
                valid_flag_list.append(valid)
            mlvl_valid_flag_list.append(valid_flag_list)

        # Transpose list so it's per-image: list[num_imgs][num_levels]
        center_list = list(map(list, zip(*mlvl_center_list)))
        valid_flag_list = list(map(list, zip(*mlvl_valid_flag_list)))

        print(f"[get_points] featmap_sizes: {featmap_sizes}, num images: {len(img_metas_dict)}")
        print(f"[get_points] center_list lens: {[len(lvl) for lvl in center_list]}")
        print(f"[get_points] valid_flag_list lens: {[len(lvl) for lvl in valid_flag_list]}")
        return center_list, valid_flag_list


    def loss(self, cls_scores: List[Tensor], pts_preds_init: List[Tensor],
            pts_preds_refine: List[Tensor], batch_gt_instances: InstanceList,
            img_metas_dict, batch_gt_instances_ignore: OptInstanceList = None) -> dict:
        
        """Calculate the loss."""
        featmap_sizes = [featmap.size()[-2:] for featmap in cls_scores]
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

        print(f"[loss] cls_scores lens: {len(cls_scores)}, pts_preds_init lens: {len(pts_preds_init)}, pts_preds_refine lens: {len(pts_preds_refine)}")
        print(f"[loss] avg_factor_init_sum: {avg_factor_init_sum}, avg_factor_refine_sum: {avg_factor_refine_sum}")
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

        print(f"[loss_single] cls_score shape: {cls_score.shape}, pts_pred_init shape: {pts_pred_init.shape}, pts_pred_refine shape: {pts_pred_refine.shape}")

        # ----- Classification Loss -----
        cls_score = cls_score.permute(0, 2, 3, 1).reshape(-1, self.cls_out_channels)

        labels = torch.cat(labels, dim=0).reshape(-1) if isinstance(labels, (list, tuple)) else labels.reshape(-1)
        label_weights = torch.cat(label_weights, dim=0).reshape(-1) if isinstance(label_weights, (list, tuple)) else label_weights.reshape(-1)

        valid_class_idx = (label_weights > 0).nonzero(as_tuple=False).view(-1)

        # Debug: Print classes
        print("GT classes:", labels.cpu().numpy())
        print("Pred classes (logits):", cls_score.detach().cpu().numpy())

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

        pred_bboxes_init = pts_pred_init.reshape(-1, 4)
        pred_bboxes_refine = pts_pred_refine.reshape(-1, 4)

        # Debug: Print GT and predicted boxes and points
        print("GT boxes (init):", gt_bboxes_init.cpu().numpy())
        print("Pred boxes (init):", pred_bboxes_init.detach().cpu().numpy())
        print("GT boxes (refine):", gt_bboxes_refine.cpu().numpy())
        print("Pred boxes (refine):", pred_bboxes_refine.detach().cpu().numpy())
        print("Pred points (init):", pts_pred_init.detach().cpu().numpy())
        print("Pred points (refine):", pts_pred_refine.detach().cpu().numpy())

        min_len_init = min(pred_bboxes_init.size(0), gt_bboxes_init.size(0), gt_weights_init.size(0))
        pred_bboxes_init = pred_bboxes_init[:min_len_init]
        gt_bboxes_init = gt_bboxes_init[:min_len_init]
        gt_weights_init = gt_weights_init[:min_len_init]

        min_len_refine = min(pred_bboxes_refine.size(0), gt_bboxes_refine.size(0), gt_weights_refine.size(0))
        pred_bboxes_refine = pred_bboxes_refine[:min_len_refine]
        gt_bboxes_refine = gt_bboxes_refine[:min_len_refine]
        gt_weights_refine = gt_weights_refine[:min_len_refine]

        valid_init = gt_weights_init.sum(dim=1) > 0 if gt_weights_init.dim() > 1 else gt_weights_init > 0
        pred_bboxes_init = pred_bboxes_init[valid_init]
        gt_bboxes_init = gt_bboxes_init[valid_init]
        gt_weights_init = gt_weights_init[valid_init]

        valid_refine = gt_weights_refine.sum(dim=1) > 0 if gt_weights_refine.dim() > 1 else gt_weights_refine > 0
        pred_bboxes_refine = pred_bboxes_refine[valid_refine]
        gt_bboxes_refine = gt_bboxes_refine[valid_refine]
        gt_weights_refine = gt_weights_refine[valid_refine]

        normalize_factor = self.point_base_scale * stride

        loss_pts_init = self.loss_bbox_init(
            pred_bboxes_init / normalize_factor,
            gt_bboxes_init / normalize_factor,
            gt_weights_init,
            avg_factor=float(avg_factor_init)
        )

        loss_pts_refine = self.loss_bbox_refine(
            pred_bboxes_refine / normalize_factor,
            gt_bboxes_refine / normalize_factor,
            gt_weights_refine,
            avg_factor=float(avg_factor_refine)
        )

        print(f"[loss_single] labels shape: {labels.shape}, label_weights shape: {label_weights.shape}")
        print(f"[loss_single] valid_class_idx: {valid_class_idx.shape}, num valid: {valid_class_idx.numel()}")
        print(f"[loss_single] GT boxes (init): {gt_bboxes_init.shape}, GT boxes (refine): {gt_bboxes_refine.shape}")
        print(f"[loss_single] pred_bboxes_init: {pred_bboxes_init.shape}, pred_bboxes_refine: {pred_bboxes_refine.shape}")
        print(f"[loss_single] gt_weights_init sum: {gt_weights_init.sum().item()}, gt_weights_refine sum: {gt_weights_refine.sum().item()}")
        print(f"[loss_single] valid_init: {valid_init.sum().item()}, valid_refine: {valid_refine.sum().item()}")

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
