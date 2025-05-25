# Copyright (c) OpenMMLab. All rights reserved.
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn import ConvModule
from mmcv.cnn.bricks import DropPath
from mmengine.model import BaseModule

from mmdet.models.layers.se_layer import SELayer


class InvertedResidual(BaseModule):
    """Fixed Inverted Residual Block with proper channel validation."""

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 stride=1,
                 se_cfg=None,
                 with_expand_conv=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.,
                 with_cp=False,
                 init_cfg=None):
        super(InvertedResidual, self).__init__(init_cfg)
        self.out_channels = out_channels
        
        # Validate channel dimensions
        assert mid_channels % 8 == 0, f"mid_channels ({mid_channels}) must be divisible by 8"
        assert in_channels % 8 == 0, f"in_channels ({in_channels}) must be divisible by 8"
        assert out_channels % 8 == 0, f"out_channels ({out_channels}) must be divisible by 8"
        
        self.with_res_shortcut = (stride == 1 and in_channels == out_channels)
        assert stride in [1, 2], f'stride must in [1, 2]. Received {stride}.'
        self.with_cp = with_cp
        self.drop_path = DropPath(drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.with_se = se_cfg is not None
        self.with_expand_conv = with_expand_conv

        if self.with_se:
            assert isinstance(se_cfg, dict)
        if not self.with_expand_conv:
            assert mid_channels == in_channels

        if self.with_expand_conv:
            self.expand_conv = ConvModule(
                in_channels=in_channels,
                out_channels=mid_channels,
                kernel_size=1,
                stride=1,
                padding=0,
                conv_cfg=conv_cfg,
                norm_cfg=norm_cfg,
                act_cfg=act_cfg)
        
        # Depthwise convolution with proper group validation
        self.depthwise_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=kernel_size // 2,
            groups=mid_channels,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if self.with_se:
            self.se = SELayer(**se_cfg)

        self.linear_conv = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

    def forward(self, x):
        def _inner_forward(x):
            out = x

            if self.with_expand_conv:
                out = self.expand_conv(out)
                # Validate expanded channels match groups
                assert out.size(1) == self.depthwise_conv.conv.groups, \
                    f"Channels {out.size(1)} must match groups {self.depthwise_conv.conv.groups}"

            out = self.depthwise_conv(out)

            if self.with_se:
                out = self.se(out)

            out = self.linear_conv(out)

            if self.with_res_shortcut:
                return x + self.drop_path(out)
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out
