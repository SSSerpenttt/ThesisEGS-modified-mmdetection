# Copyright (c) OpenMMLab. All rights reserved.
from typing import List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from mmcv.cnn import ConvModule, build_conv_layer, build_upsample_layer
from mmcv.ops.carafe import CARAFEPack
from mmengine.config import ConfigDict
from mmengine.model import BaseModule, ModuleList
from mmengine.structures import InstanceData
from torch import Tensor
from torch.nn.modules.utils import _pair

from mmdet.models.losses import CrossEntropyLoss
from mmdet.models.task_modules.samplers import SamplingResult
from mmdet.models.utils import empty_instances
from mmdet.registry import MODELS
from mmdet.structures.mask import mask_target
from mmdet.utils import ConfigType, InstanceList, OptConfigType, OptMultiConfig

BYTES_PER_FLOAT = 4
GPU_MEM_LIMIT = 1024**3  # 1 GB memory limit


@MODELS.register_module()
class FCNMaskHead(BaseModule):

    def __init__(self,
                 num_convs: int = 4,
                 roi_feat_size: int = 14,
                 in_channels: int = 256,
                 conv_kernel_size: int = 3,
                 conv_out_channels: int = 256,
                 num_classes: int = 80,
                 class_agnostic: int = False,
                 upsample_cfg: ConfigType = dict(
                     type='deconv', scale_factor=2),
                 conv_cfg: OptConfigType = None,
                 norm_cfg: OptConfigType = None,
                 mask_size: int = 28,
                 predictor_cfg: ConfigType = dict(type='Conv'),
                 loss_mask: ConfigType = dict(
                     type='mmdet.CrossEntropyLoss', use_mask=True, loss_weight=1.0),
                 init_cfg: OptMultiConfig = None) -> None:
        assert init_cfg is None, 'To prevent abnormal initialization ' \
                                 'behavior, init_cfg is not allowed to be set'
        super().__init__(init_cfg=init_cfg)
        self.upsample_cfg = upsample_cfg.copy()
        if self.upsample_cfg['type'] not in [
                None, 'deconv', 'nearest', 'bilinear', 'carafe'
        ]:
            raise ValueError(
                f'Invalid upsample method {self.upsample_cfg["type"]}, '
                'accepted methods are "deconv", "nearest", "bilinear", '
                '"carafe"')
        self.num_convs = num_convs
        self.roi_feat_size = _pair(roi_feat_size)
        self.in_channels = in_channels
        self.conv_kernel_size = conv_kernel_size
        self.conv_out_channels = conv_out_channels
        self.upsample_method = self.upsample_cfg.get('type')
        self.scale_factor = self.upsample_cfg.pop('scale_factor', None)
        self.num_classes = num_classes
        self.class_agnostic = class_agnostic
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.mask_size = mask_size
        self.predictor_cfg = predictor_cfg
        self.loss_mask = MODELS.build(loss_mask)

        self.convs = ModuleList()
        for i in range(self.num_convs):
            in_channels = (
                self.in_channels if i == 0 else self.conv_out_channels)
            padding = (self.conv_kernel_size - 1) // 2
            self.convs.append(
                ConvModule(
                    in_channels,
                    self.conv_out_channels,
                    self.conv_kernel_size,
                    padding=padding,
                    conv_cfg=conv_cfg,
                    norm_cfg=norm_cfg))
        upsample_in_channels = (
            self.conv_out_channels if self.num_convs > 0 else in_channels)
        upsample_cfg_ = self.upsample_cfg.copy()
        if self.upsample_method is None:
            self.upsample = None
        elif self.upsample_method == 'deconv':
            upsample_cfg_.update(
                in_channels=upsample_in_channels,
                out_channels=self.conv_out_channels,
                kernel_size=self.scale_factor,
                stride=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        elif self.upsample_method == 'carafe':
            upsample_cfg_.update(
                channels=upsample_in_channels, scale_factor=self.scale_factor)
            self.upsample = build_upsample_layer(upsample_cfg_)
        else:
            align_corners = (None
                             if self.upsample_method == 'nearest' else False)
            upsample_cfg_.update(
                scale_factor=self.scale_factor,
                mode=self.upsample_method,
                align_corners=align_corners)
            self.upsample = build_upsample_layer(upsample_cfg_)

        out_channels = 1 if self.class_agnostic else self.num_classes
        logits_in_channel = (
            self.conv_out_channels
            if self.upsample_method == 'deconv' else upsample_in_channels)
        self.conv_logits = build_conv_layer(self.predictor_cfg,
                                            logits_in_channel, out_channels, 1)
        self.relu = nn.ReLU(inplace=True)
        self.debug_imgs = None

    # [Rest of the methods remain unchanged...]
    # ... (keep all other methods exactly as they were in the original)
