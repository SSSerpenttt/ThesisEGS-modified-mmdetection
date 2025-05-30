# Copyright (c) OpenMMLab. All rights reserved.
import copy
import warnings
from typing import List, Tuple, Union

import torch
from torch import Tensor

from mmdet.registry import MODELS
from mmdet.structures import SampleList
from mmdet.utils import ConfigType, OptConfigType, OptMultiConfig
from .base import BaseDetector

from mmengine.structures import InstanceData
from mmdet.structures import DetDataSample


@MODELS.register_module()
class TwoStageDetector(BaseDetector):
    """Base class for two-stage detectors.

    Two-stage detectors typically consisting of a region proposal network and a
    task-specific regression head.
    """

    def __init__(self,
                 backbone: ConfigType,
                 neck: OptConfigType = None,
                 rpn_head: OptConfigType = None,
                 roi_head: OptConfigType = None,
                 train_cfg: OptConfigType = None,
                 test_cfg: OptConfigType = None,
                 data_preprocessor: OptConfigType = None,
                 init_cfg: OptMultiConfig = None) -> None:
        super().__init__(
            data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        self.backbone = MODELS.build(backbone)

        if neck is not None:
            self.neck = MODELS.build(neck)

        if rpn_head is not None:
            rpn_train_cfg = train_cfg.rpn if train_cfg is not None else None
            rpn_head_ = rpn_head.copy()
            rpn_head_.update(train_cfg=rpn_train_cfg, test_cfg=test_cfg.rpn)
            rpn_head_num_classes = rpn_head_.get('num_classes', None)
            if rpn_head_num_classes is None:
                rpn_head_.update(num_classes=1)
            else:
                if rpn_head_num_classes != 1:
                    warnings.warn(
                        'The `num_classes` should be 1 in RPN, but get '
                        f'{rpn_head_num_classes}, please set '
                        'rpn_head.num_classes = 1 in your config file.')
                    rpn_head_.update(num_classes=1)
            self.rpn_head = MODELS.build(rpn_head_)

        if roi_head is not None:
            # update train and test cfg here for now
            # TODO: refactor assigner & sampler
            rcnn_train_cfg = train_cfg.rcnn if train_cfg is not None else None
            roi_head.update(train_cfg=rcnn_train_cfg)
            roi_head.update(test_cfg=test_cfg.rcnn)
            self.roi_head = MODELS.build(roi_head)

        self.train_cfg = train_cfg
        self.test_cfg = test_cfg

    def _load_from_state_dict(self, state_dict: dict, prefix: str,
                              local_metadata: dict, strict: bool,
                              missing_keys: Union[List[str], str],
                              unexpected_keys: Union[List[str], str],
                              error_msgs: Union[List[str], str]) -> None:
        """Exchange bbox_head key to rpn_head key when loading single-stage
        weights into two-stage model."""
        bbox_head_prefix = prefix + '.bbox_head' if prefix else 'bbox_head'
        bbox_head_keys = [
            k for k in state_dict.keys() if k.startswith(bbox_head_prefix)
        ]
        rpn_head_prefix = prefix + '.rpn_head' if prefix else 'rpn_head'
        rpn_head_keys = [
            k for k in state_dict.keys() if k.startswith(rpn_head_prefix)
        ]
        if len(bbox_head_keys) != 0 and len(rpn_head_keys) == 0:
            for bbox_head_key in bbox_head_keys:
                rpn_head_key = rpn_head_prefix + \
                               bbox_head_key[len(bbox_head_prefix):]
                state_dict[rpn_head_key] = state_dict.pop(bbox_head_key)
        super()._load_from_state_dict(state_dict, prefix, local_metadata,
                                      strict, missing_keys, unexpected_keys,
                                      error_msgs)

    @property
    def with_rpn(self) -> bool:
        """bool: whether the detector has RPN"""
        return hasattr(self, 'rpn_head') and self.rpn_head is not None

    @property
    def with_roi_head(self) -> bool:
        """bool: whether the detector has a RoI head"""
        return hasattr(self, 'roi_head') and self.roi_head is not None

    def extract_feat(self, batch_inputs):
        if isinstance(batch_inputs, list):
            batch_inputs = torch.stack(batch_inputs, dim=0)
        x = self.backbone(batch_inputs)
        print("[extract_feat] Backbone outputs:", [f.shape for f in x])
        if self.with_neck:
            x = self.neck(x)
            print("[extract_feat] Neck outputs:", [f.shape for f in x])
        #   print("Neck output shapes:", [feat.shape for feat in x])
        # else:
        #     print("No neck used")
        expected_strides = [4, 8, 16, 32, 64]
        for i, feat in enumerate(x):
            print(f"[extract_feat] Feature map {i} stride (expected {expected_strides[i]}): shape={feat.shape}")
        
        return x

    def _forward(self, batch_inputs: Tensor,
                batch_data_samples: SampleList) -> tuple:
        """Network forward process. Usually includes backbone, neck and head
        forward without any post-processing.

        Args:
            batch_inputs (Tensor): Inputs with shape (N, C, H, W).
            batch_data_samples (list[:obj:`DetDataSample`]): Each item contains
                the meta information of each image and corresponding
                annotations.

        Returns:
            tuple: A tuple of features from ``rpn_head`` and ``roi_head``
            forward.
        """
        # Unwrap if batch_data_samples itself is a tuple
        if isinstance(batch_data_samples, tuple):
            batch_data_samples = batch_data_samples[0]

        # Unwrap inner elements if they are tuples
        batch_data_samples = [
            data_sample[0] if isinstance(data_sample, tuple) else data_sample
            for data_sample in batch_data_samples
        ]

        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_outputs = self.rpn_head.forward(x)
            if self.rpn_head.training:
                cls_scores, _, pts_preds_refine = rpn_outputs
                rpn_results_list = self.rpn_head.get_proposals(
                    cls_scores, pts_preds_refine, batch_data_samples)
                for i, rpn_results in enumerate(rpn_results_list):
                    print(f"[RPN Proposals] Image {i}: bboxes shape {rpn_results.bboxes.shape}, scores: {getattr(rpn_results, 'scores', None)}")
            else:
                cls_scores, bbox_preds = rpn_outputs
                rpn_results_list = self.rpn_head.get_proposals(
                    cls_scores, bbox_preds, batch_data_samples)
                for i, rpn_results in enumerate(rpn_results_list):
                    print(f"[RPN Proposals] Image {i}: bboxes shape {rpn_results.bboxes.shape}, scores: {getattr(rpn_results, 'scores', None)}")
        else:
            # If RPN is not used, use existing proposals
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]
            for i, rpn_results in enumerate(rpn_results_list):
                print(f"[RPN Proposals] Image {i}: bboxes shape {rpn_results.bboxes.shape}, scores: {getattr(rpn_results, 'scores', None)}")
        # Forward the RoI head
        roi_outs = self.roi_head.forward(x, rpn_results_list, batch_data_samples)
        return roi_outs



    def loss(self, batch_inputs: Tensor, batch_data_samples: SampleList) -> dict:
        """Calculate losses from a batch of inputs and data samples."""
        # print(f"Type of batch_data_samples: {type(batch_data_samples)}")
        
        # Unwrap if batch_data_samples itself is a tuple
        if isinstance(batch_data_samples, tuple):
            # print(f"Length of tuple batch_data_samples: {len(batch_data_samples)}")
            batch_data_samples = batch_data_samples[0]

        # Unwrap inner elements if they are tuples
        batch_data_samples = [
            data_sample[0] if isinstance(data_sample, tuple) else data_sample
            for data_sample in batch_data_samples
        ]
        # print(f"Type after adjustment: {type(batch_data_samples)}")
        # print("DEBUG: batch_data_samples type:", type(batch_data_samples))
        # print("DEBUG: First entry type:", type(batch_data_samples[0]))
        # print("DEBUG: First entry contents:", batch_data_samples[0])

        x = self.extract_feat(batch_inputs)
        losses = dict()

        # RPN forward and loss
        if self.with_rpn:
            proposal_cfg = self.train_cfg.get('rpn_proposal', self.test_cfg.rpn)

            # Create RPN-specific data samples
            rpn_data_samples = []
            for data_sample in batch_data_samples:
                rpn_instances = InstanceData()
                rpn_instances.bboxes = data_sample.gt_instances.bboxes
                rpn_instances.labels = torch.zeros_like(data_sample.gt_instances.labels)

                rpn_sample = DetDataSample()
                rpn_sample.gt_instances = rpn_instances
                rpn_sample.set_metainfo(data_sample.metainfo)
                rpn_data_samples.append(rpn_sample)

            rpn_losses, rpn_results_list = self.rpn_head.loss_and_predict(
                x, rpn_data_samples, proposal_cfg=proposal_cfg)

            # Rename losses to include 'rpn_' prefix
            for key in list(rpn_losses.keys()):
                if 'loss' in key and 'rpn' not in key:
                    rpn_losses[f'rpn_{key}'] = rpn_losses.pop(key)
            losses.update(rpn_losses)
        else:
            rpn_results_list = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        # ROI head losses
        roi_losses = self.roi_head.loss(x, rpn_results_list, batch_data_samples)
        losses.update(roi_losses)

        return losses


    def predict(self, batch_inputs: Tensor, batch_data_samples: SampleList, **kwargs) -> list:
        # Unwrap outer tuple if needed
        if isinstance(batch_data_samples, tuple):
            batch_data_samples = batch_data_samples[0]

        # Flatten and unwrap any nested lists/tuples
        flat_samples = []
        for sample in batch_data_samples:
            if isinstance(sample, (list, tuple)):
                sample = sample[0]
            flat_samples.append(sample)
        batch_data_samples = flat_samples

        x = self.extract_feat(batch_inputs)

        if self.with_rpn:
            rpn_outputs = self.rpn_head.forward(x)
            cls_scores, bbox_preds = rpn_outputs
            proposals = self.rpn_head.get_proposals(cls_scores, bbox_preds, batch_data_samples)
            # for p in proposals:
            #     print("DEBUG: proposals bboxes shape:", p.bboxes.shape)
            #     print("DEBUG: proposals scores:", getattr(p, 'scores', None))
        else:
            proposals = [
                data_sample.proposals for data_sample in batch_data_samples
            ]

        results_list = self.roi_head.predict(x, proposals, batch_data_samples, rescale=True)

        # --- PATCH: Return DetDataSample objects with pred_instances and meta info ---
        wrapped_results = []
        for result, data_sample in zip(results_list, batch_data_samples):
            # Always create a new DetDataSample to avoid mutation
            det_sample = DetDataSample()
            # Copy meta info (img_path, ori_shape, etc.)
            if hasattr(data_sample, 'metainfo'):
                det_sample.set_metainfo(data_sample.metainfo)
            elif isinstance(data_sample, dict):
                det_sample.set_metainfo(data_sample)
            # Set prediction results
            det_sample.pred_instances = result
            wrapped_results.append(det_sample)
        return wrapped_results
