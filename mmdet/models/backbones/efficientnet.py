# Copyright (c) OpenMMLab. All rights reserved.
import copy
import math
from functools import partial

import torch
import torch.nn as nn
import torch.utils.checkpoint as cp
from mmcv.cnn.bricks import ConvModule, DropPath
from mmengine.model import BaseModule, Sequential

from mmdet.registry import MODELS
from mmdet.models.layers import InvertedResidual, SELayer
from mmdet.models.utils import make_divisible


class EdgeResidual(BaseModule):
    """Edge Residual Block.

    Args:
        in_channels (int): The input channels of this module.
        out_channels (int): The output channels of this module.
        mid_channels (int): The input channels of the second convolution.
        kernel_size (int): The kernel size of the first convolution.
            Defaults to 3.
        stride (int): The stride of the first convolution. Defaults to 1.
        se_cfg (dict, optional): Config dict for se layer. Defaults to None,
            which means no se layer.
        with_residual (bool): Use residual connection. Defaults to True.
        conv_cfg (dict, optional): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to ``dict(type='BN')``.
        act_cfg (dict): Config dict for activation layer.
            Defaults to ``dict(type='ReLU')``.
        drop_path_rate (float): stochastic depth rate. Defaults to 0.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
        init_cfg (dict | list[dict], optional): Initialization config dict.
    """

    def __init__(self,
                 in_channels,
                 out_channels,
                 mid_channels,
                 kernel_size=3,
                 stride=1,
                 se_cfg=None,
                 with_residual=True,
                 conv_cfg=None,
                 norm_cfg=dict(type='BN'),
                 act_cfg=dict(type='ReLU'),
                 drop_path_rate=0.,
                 with_cp=False,
                 init_cfg=None,
                 **kwargs):
        super(EdgeResidual, self).__init__(init_cfg=init_cfg)
        assert stride in [1, 2]
        self.with_cp = with_cp
        self.drop_path = DropPath(
            drop_path_rate) if drop_path_rate > 0 else nn.Identity()
        self.with_se = se_cfg is not None
        self.with_residual = (
            stride == 1 and in_channels == out_channels and with_residual)

        if self.with_se:
            assert isinstance(se_cfg, dict)

        self.conv1 = ConvModule(
            in_channels=in_channels,
            out_channels=mid_channels,
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=act_cfg)

        if self.with_se:
            self.se = SELayer(**se_cfg)

        self.conv2 = ConvModule(
            in_channels=mid_channels,
            out_channels=out_channels,
            kernel_size=1,
            stride=stride,
            padding=0,
            conv_cfg=conv_cfg,
            norm_cfg=norm_cfg,
            act_cfg=None)

        self.out_channels = out_channels

    def forward(self, x):

        def _inner_forward(x):
            out = x
            out = self.conv1(out)

            if self.with_se:
                out = self.se(out)

            out = self.conv2(out)

            if self.with_residual:
                return x + self.drop_path(out)
            else:
                return out

        if self.with_cp and x.requires_grad:
            out = cp.checkpoint(_inner_forward, x)
        else:
            out = _inner_forward(x)

        return out


def model_scaling(layer_setting, arch_setting):
    """Scaling operation to the layer's parameters according to the
    arch_setting."""
    # scale width
    new_layer_setting = copy.deepcopy(layer_setting)
    for layer_cfg in new_layer_setting:
        for block_cfg in layer_cfg:
            block_cfg[1] = make_divisible(block_cfg[1] * arch_setting[0], 8)

    # scale depth
    split_layer_setting = [new_layer_setting[0]]
    for layer_cfg in new_layer_setting[1:-1]:
        tmp_index = [0]
        for i in range(len(layer_cfg) - 1):
            if layer_cfg[i + 1][1] != layer_cfg[i][1]:
                tmp_index.append(i + 1)
        tmp_index.append(len(layer_cfg))
        for i in range(len(tmp_index) - 1):
            split_layer_setting.append(layer_cfg[tmp_index[i]:tmp_index[i +
                                                                        1]])
    split_layer_setting.append(new_layer_setting[-1])

    num_of_layers = [len(layer_cfg) for layer_cfg in split_layer_setting[1:-1]]
    new_layers = [
        int(math.ceil(arch_setting[1] * num)) for num in num_of_layers
    ]

    merge_layer_setting = [split_layer_setting[0]]
    for i, layer_cfg in enumerate(split_layer_setting[1:-1]):
        if new_layers[i] <= num_of_layers[i]:
            tmp_layer_cfg = layer_cfg[:new_layers[i]]
        else:
            tmp_layer_cfg = copy.deepcopy(layer_cfg) + [layer_cfg[-1]] * (
                new_layers[i] - num_of_layers[i])
        if tmp_layer_cfg[0][3] == 1 and i != 0:
            merge_layer_setting[-1] += tmp_layer_cfg.copy()
        else:
            merge_layer_setting.append(tmp_layer_cfg.copy())
    merge_layer_setting.append(split_layer_setting[-1])

    return merge_layer_setting


@MODELS.register_module()
class EfficientNet(BaseModule):
    """EfficientNet backbone.

    Args:
        arch (str): Architecture of efficientnet. Defaults to b0.
        out_indices (Sequence[int]): Output from which stages.
            Defaults to (6, ).
        frozen_stages (int): Stages to be frozen (all param fixed).
            Defaults to 0, which means not freezing any parameters.
        conv_cfg (dict): Config dict for convolution layer.
            Defaults to None, which means using conv2d.
        norm_cfg (dict): Config dict for normalization layer.
            Defaults to dict(type='BN').
        act_cfg (dict): Config dict for activation layer.
            Defaults to dict(type='Swish').
        norm_eval (bool): Whether to set norm layers to eval mode, namely,
            freeze running stats (mean and var). Note: Effect on Batch Norm
            and its variants only. Defaults to False.
        with_cp (bool): Use checkpoint or not. Using checkpoint will save some
            memory while slowing down the training speed. Defaults to False.
    """

    # Parameters to build layers.
    # 'b' represents the architecture of normal EfficientNet family includes
    # 'b0', 'b1', 'b2', 'b3', 'b4', 'b5', 'b6', 'b7', 'b8'.
    # 'e' represents the architecture of EfficientNet-EdgeTPU including 'es',
    # 'em', 'el'.
    # 6 parameters are needed to construct a layer, From left to right:
    # - kernel_size: The kernel size of the block
    # - out_channel: The number of out_channels of the block
    # - se_ratio: The sequeeze ratio of SELayer.
    # - stride: The stride of the block
    # - expand_ratio: The expand_ratio of the mid_channels
    # - block_type: -1: Not a block, 0: InvertedResidual, 1: EdgeResidual
    layer_settings = {
        'b': [[[3, 32, 0, 2, 0, -1]],
              [[3, 16, 4, 1, 1, 0]],
              [[3, 24, 4, 2, 6, 0], [3, 24, 4, 1, 6, 0]],
              [[5, 40, 4, 2, 6, 0], [5, 40, 4, 1, 6, 0]],
              [[3, 80, 4, 2, 6, 0], [3, 80, 4, 1, 6, 0], 
               [3, 80, 4, 1, 6, 0], [5, 112, 4, 1, 6, 0],
               [5, 112, 4, 1, 6, 0], [5, 112, 4, 1, 6, 0]],
              [[5, 192, 4, 2, 6, 0], [5, 192, 4, 1, 6, 0],
               [5, 192, 4, 1, 6, 0], [5, 192, 4, 1, 6, 0],
               [3, 320, 4, 1, 6, 0]],
              [[1, 1280, 0, 1, 0, -1]]],
        'e': [[[3, 32, 0, 2, 0, -1]],
              [[3, 24, 0, 1, 3, 1]],
              [[3, 32, 0, 2, 8, 1], [3, 32, 0, 1, 8, 1]],
              [[3, 48, 0, 2, 8, 1], [3, 48, 0, 1, 8, 1],
               [3, 48, 0, 1, 8, 1], [3, 48, 0, 1, 8, 1]],
              [[5, 96, 0, 2, 8, 0], [5, 96, 0, 1, 8, 0],
               [5, 96, 0, 1, 8, 0], [5, 96, 0, 1, 8, 0],
               [5, 96, 0, 1, 8, 0], [5, 144, 0, 1, 8, 0],
               [5, 144, 0, 1, 8, 0], [5, 144, 0, 1, 8, 0],
               [5, 144, 0, 1, 8, 0]],
              [[5, 192, 0, 2, 8, 0], [5, 192, 0, 1, 8, 0]],
              [[1, 1280, 0, 1, 0, -1]]]
    }

    arch_settings = {
        'b0': (1.0, 1.0, 224),
        'b1': (1.0, 1.1, 240),
        'b2': (1.1, 1.2, 260),
        'b3': (1.2, 1.4, 300),
        'b4': (1.4, 1.8, 380),
        'b5': (1.6, 2.2, 456),
        'b6': (1.8, 2.6, 528),
        'b7': (2.0, 3.1, 600),
        'b8': (2.2, 3.6, 672),
        'es': (1.0, 1.0, 224),
        'em': (1.0, 1.1, 240),
        'el': (1.2, 1.4, 300)
    }

    def __init__(self,
                 arch='b3',
                 drop_path_rate=0.,
                 out_indices=(3, 4, 5),  # Default for BiFPN integration
                 frozen_stages=0,
                 conv_cfg=dict(type='Conv2dAdaptivePadding'),
                 norm_cfg=dict(type='BN', eps=1e-3),
                 act_cfg=dict(type='Swish'),
                 norm_eval=False,
                 with_cp=False,
                 init_cfg=[
                     dict(type='Kaiming', layer='Conv2d'),
                     dict(type='Constant', layer=['_BatchNorm', 'GroupNorm'], val=1)
                 ]):
        super().__init__(init_cfg)
        
        # Validation checks
        assert arch in self.arch_settings, \
            f'Invalid arch: {arch}. Choices: {list(self.arch_settings.keys())}'
        assert all(i in range(len(self.layer_settings[arch[0]])) for i in out_indices), \
            f'Invalid out_indices {out_indices} for arch {arch}'
            
        self.arch_setting = self.arch_settings[arch]
        self.layer_setting = self.layer_settings[arch[0]]
        self.drop_path_rate = drop_path_rate
        self.out_indices = out_indices
        self.frozen_stages = frozen_stages
        self.conv_cfg = conv_cfg
        self.norm_cfg = norm_cfg
        self.act_cfg = act_cfg
        self.norm_eval = norm_eval
        self.with_cp = with_cp
        
        # Build model
        self._build_model()

    def _build_model(self):
        """Construct the model architecture."""
        self.layer_setting = model_scaling(self.layer_setting, self.arch_setting)
        
        # Initial convolution
        block_cfg_0 = self.layer_setting[0][0]
        self.in_channels = make_divisible(block_cfg_0[1], 8)
        self.layers = nn.ModuleList([
            ConvModule(
                in_channels=3,
                out_channels=self.in_channels,
                kernel_size=block_cfg_0[0],
                stride=block_cfg_0[3],
                padding=block_cfg_0[0] // 2,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg,
                act_cfg=self.act_cfg)
        ])
        
        # Intermediate layers
        self._build_intermediate_layers()
        
        # Final convolution if needed
        if len(self.layers) < max(self.out_indices) + 1:
            block_cfg_last = self.layer_setting[-1][0]
            self.layers.append(
                ConvModule(
                    in_channels=self.in_channels,
                    out_channels=block_cfg_last[1],
                    kernel_size=block_cfg_last[0],
                    stride=block_cfg_last[3],
                    padding=block_cfg_last[0] // 2,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.act_cfg))

    def _build_intermediate_layers(self):
        """Construct the intermediate layers with drop path."""
        layer_setting = self.layer_setting[1:-1]
        total_blocks = sum(len(x) for x in layer_setting)
        dpr = torch.linspace(0, self.drop_path_rate, total_blocks).tolist()
        
        block_idx = 0
        for i, layer_cfg in enumerate(layer_setting):
            if i > max(self.out_indices) - 1:
                break
                
            layer = []
            for block_cfg in layer_cfg:
                layer.append(self._build_block(block_cfg, dpr[block_idx]))
                block_idx += 1
                
            self.layers.append(Sequential(*layer))
            self.in_channels = layer[-1].out_channels

    def _build_block(self, block_cfg, drop_path_rate):
      """Build a single block with strict channel validation."""
      kernel_size, out_channels, se_ratio, stride, expand_ratio, block_type = block_cfg
      
      # Ensure all channels are divisible by 8
      out_channels = make_divisible(out_channels, 8)
      mid_channels = make_divisible(self.in_channels * expand_ratio, 8)
      
      # For EfficientNet-b3, we need to ensure specific channel counts
      if self.arch_setting == self.arch_settings['b3']:
          if expand_ratio == 6:  # Standard expansion for b3
              mid_channels = make_divisible(self.in_channels * 6, 8)
      
      # Configure SE layer
      se_cfg = None
      if se_ratio > 0:
          se_cfg = dict(
              channels=mid_channels,
              ratio=se_ratio,
              act_cfg=(self.act_cfg, dict(type='Sigmoid')))
      
      # Select block type
      if block_type == 1:  # EdgeResidual
          with_residual = not (expand_ratio == 3 and stride > 1)
          if not with_residual:
              expand_ratio = 4  # Adjust for non-residual path
          mid_channels = make_divisible(self.in_channels * expand_ratio, 8)
          
          return EdgeResidual(
              in_channels=self.in_channels,
              out_channels=out_channels,
              mid_channels=mid_channels,
              kernel_size=kernel_size,
              stride=stride,
              se_cfg=se_cfg,
              with_residual=with_residual,
              conv_cfg=self.conv_cfg,
              norm_cfg=self.norm_cfg,
              act_cfg=self.act_cfg,
              drop_path_rate=drop_path_rate,
              with_cp=self.with_cp)
      else:  # InvertedResidual
          return InvertedResidual(
              in_channels=self.in_channels,
              out_channels=out_channels,
              mid_channels=mid_channels,
              kernel_size=kernel_size,
              stride=stride,
              se_cfg=se_cfg,
              conv_cfg=self.conv_cfg,
              norm_cfg=self.norm_cfg,
              act_cfg=self.act_cfg,
              drop_path_rate=drop_path_rate,
              with_cp=self.with_cp,
              with_expand_conv=(mid_channels != self.in_channels))

    def forward(self, x):
        """Enhanced forward pass with better input handling."""
        if isinstance(x, (list, tuple)):
            x = x[0]  # Handle multi-input cases
            
        outs = []
        for i, layer in enumerate(self.layers):
            x = layer(x)
            if i in self.out_indices:
                outs.append(x)
                if len(outs) == len(self.out_indices):
                    break  # Early exit if we have all required features
                    
        return tuple(outs)

    def _freeze_stages(self):
        """Freeze stages for transfer learning."""
        for i in range(self.frozen_stages):
            m = self.layers[i]
            m.eval()
            for param in m.parameters():
                param.requires_grad = False

    def train(self, mode=True):
        """Enhanced train mode with norm freezing option."""
        super().train(mode)
        self._freeze_stages()
        if mode and self.norm_eval:
            for m in self.modules():
                if isinstance(m, nn.BatchNorm2d):
                    m.eval()
