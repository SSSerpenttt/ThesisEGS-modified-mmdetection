import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init
import torch

from mmdet.registry import MODELS
from mmcv.cnn import ConvModule

@MODELS.register_module()
class BiFPN(nn.Module):
    """Enhanced BiFPN implementation with improved features:
    - Better memory efficiency
    - More stable training
    - Cleaner code organization
    - Support for different fusion operations
    """
    
    def __init__(self,
                 in_channels,
                 out_channels,
                 num_outs,
                 start_level=0,
                 end_level=-1,
                 stack=1,
                 add_extra_convs=False,
                 extra_convs_on_inputs=True,
                 relu_before_extra_convs=False,
                 no_norm_on_lateral=False,
                 conv_cfg=None,
                 norm_cfg=dict(type='SyncBN'),  # Changed to SyncBN for multi-GPU
                 act_cfg=dict(type='Swish'),  # Swish often works better than ReLU
                 fusion_method='fast_attention'):  # Added fusion method option
        super(BiFPN, self).__init__()
        
        assert isinstance(in_channels, list)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.num_ins = len(in_channels)
        self.num_outs = num_outs
        self.activation = act_cfg
        self.relu_before_extra_convs = relu_before_extra_convs
        self.no_norm_on_lateral = no_norm_on_lateral
        self.stack = stack
        self.fusion_method = fusion_method
        self.conv_cfg = conv_cfg

        if end_level == -1:
            self.backbone_end_level = self.num_ins
            assert num_outs >= self.num_ins - start_level
        else:
            self.backbone_end_level = end_level
            assert end_level <= len(in_channels)
            assert num_outs == end_level - start_level
            
        self.start_level = start_level
        self.end_level = end_level
        self.add_extra_convs = add_extra_convs
        self.extra_convs_on_inputs = extra_convs_on_inputs

        self._init_layers()
        self.init_weights()

    def _init_layers(self):
        """Initialize layers in a more organized way."""
        # Lateral convolutions
        self.lateral_convs = nn.ModuleList()
        for i in range(self.start_level, self.backbone_end_level):
            l_conv = ConvModule(
                self.in_channels[i],
                self.out_channels,
                1,
                conv_cfg=self.conv_cfg,
                norm_cfg=self.norm_cfg if not self.no_norm_on_lateral else None,
                act_cfg=self.activation,
                inplace=False)
            self.lateral_convs.append(l_conv)

        # Stacked BiFPN modules
        self.stack_bifpn_convs = nn.ModuleList()
        for _ in range(self.stack):
            self.stack_bifpn_convs.append(
                BiFPNModule(
                    channels=self.out_channels,
                    levels=self.backbone_end_level - self.start_level,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    activation=self.activation,
                    fusion_method=self.fusion_method))

        # Extra convolutions
        self.fpn_convs = nn.ModuleList()
        extra_levels = self.num_outs - self.backbone_end_level + self.start_level
        if self.add_extra_convs and extra_levels >= 1:
            for i in range(extra_levels):
                in_channels = (self.in_channels[self.backbone_end_level - 1] 
                             if i == 0 and self.extra_convs_on_inputs 
                             else self.out_channels)
                extra_fpn_conv = ConvModule(
                    in_channels,
                    self.out_channels,
                    3,
                    stride=2,
                    padding=1,
                    conv_cfg=self.conv_cfg,
                    norm_cfg=self.norm_cfg,
                    act_cfg=self.activation,
                    inplace=False)
                self.fpn_convs.append(extra_fpn_conv)

    def init_weights(self):
        """Improved weight initialization."""
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

    def forward(self, inputs):
        """Enhanced forward pass with better memory management."""
        assert len(inputs) == len(self.in_channels)

        # Build laterals
        laterals = [
            lateral_conv(inputs[i + self.start_level])
            for i, lateral_conv in enumerate(self.lateral_convs)
        ]

        # Process through stacked BiFPN modules
        for bifpn_module in self.stack_bifpn_convs:
            laterals = bifpn_module(laterals)

        # Handle extra levels
        outs = laterals
        if self.num_outs > len(outs):
            if not self.add_extra_convs:
                # Use max pool
                for _ in range(self.num_outs - len(outs)):
                    outs.append(F.max_pool2d(outs[-1], 1, stride=2))
            else:
                # Add extra convs
                if self.extra_convs_on_inputs:
                    orig = inputs[self.backbone_end_level - 1]
                    outs.append(self.fpn_convs[0](orig))
                else:
                    outs.append(self.fpn_convs[0](outs[-1]))
                
                for i in range(1, self.num_outs - len(outs)):
                    if self.relu_before_extra_convs:
                        outs.append(self.fpn_convs[i](F.relu(outs[-1])))
                    else:
                        outs.append(self.fpn_convs[i](outs[-1]))

        return tuple(outs)


class BiFPNModule(nn.Module):
    """Enhanced BiFPN module with multiple fusion options."""
    
    def __init__(self,
                 channels,
                 levels,
                 init=0.5,
                 conv_cfg=None,
                 norm_cfg=None,
                 activation=None,
                 eps=1e-4,
                 fusion_method='fast_attention'):
        super(BiFPNModule, self).__init__()
        self.activation = activation
        self.eps = eps
        self.levels = levels
        self.fusion_method = fusion_method
        
        # Initialize fusion parameters based on method
        if fusion_method == 'fast_attention':
            self.w1 = nn.Parameter(torch.Tensor(2, levels).fill_(init))
            self.w2 = nn.Parameter(torch.Tensor(3, levels - 2).fill_(init))
            self.relu = nn.ReLU(inplace=False)
        elif fusion_method == 'softmax':
            self.w1 = nn.Parameter(torch.Tensor(2, levels))
            self.w2 = nn.Parameter(torch.Tensor(3, levels - 2))
            nn.init.constant_(self.w1, 1.0)
            nn.init.constant_(self.w2, 1.0)
        
        # Initialize convolutions
        self.bifpn_convs = nn.ModuleList()
        for _ in range(2):  # Top-down and bottom-up passes
            for _ in range(levels - 1):
                self.bifpn_convs.append(
                    ConvModule(
                        channels,
                        channels,
                        3,
                        padding=1,
                        conv_cfg=conv_cfg,
                        norm_cfg=norm_cfg,
                        act_cfg=activation,
                        inplace=False))

    def _fuse_features(self, weights, *features):
        """Fuse features using the specified method."""
        if self.fusion_method == 'fast_attention':
            weights = self.relu(weights)
            norm = torch.sum(weights, dim=0, keepdim=True) + self.eps
            weights = weights / norm
            fused = sum(w * f for w, f in zip(weights, features))
        elif self.fusion_method == 'softmax':
            weights = F.softmax(weights, dim=0)
            fused = sum(w * f for w, f in zip(weights, features))
        else:
            raise ValueError(f"Unknown fusion method: {self.fusion_method}")
        return fused

    def forward(self, inputs):
        assert len(inputs) == self.levels
        
        # Clone inputs to avoid in-place modification
        td_features = [x.clone() for x in inputs]  # Top-down features
        bu_features = td_features.copy()           # Bottom-up features
        
        conv_idx = 0
        
        # Top-down path
        for i in range(self.levels - 1, 0, -1):
            if self.fusion_method in ['fast_attention', 'softmax']:
                fused = self._fuse_features(
                    self.w1[:, i-1],
                    td_features[i-1],
                    F.interpolate(td_features[i], scale_factor=2, mode='nearest'))
            else:
                # Default weighted sum
                fused = (td_features[i-1] + F.interpolate(td_features[i], scale_factor=2, mode='nearest')) / 2
            
            td_features[i-1] = self.bifpn_convs[conv_idx](fused)
            conv_idx += 1

        # Bottom-up path
        for i in range(self.levels - 2):
            if self.fusion_method in ['fast_attention', 'softmax']:
                fused = self._fuse_features(
                    self.w2[:, i],
                    bu_features[i+1],
                    F.max_pool2d(bu_features[i], kernel_size=2),
                    inputs[i+1])  # Residual connection
            else:
                # Default weighted sum
                fused = (bu_features[i+1] + F.max_pool2d(bu_features[i], kernel_size=2) + inputs[i+1]) / 3
            
            bu_features[i+1] = self.bifpn_convs[conv_idx](fused)
            conv_idx += 1

        # Final level
        if self.fusion_method in ['fast_attention', 'softmax']:
            fused = self._fuse_features(
                self.w1[:, -1],
                bu_features[-1],
                F.max_pool2d(bu_features[-2], kernel_size=2))
        else:
            fused = (bu_features[-1] + F.max_pool2d(bu_features[-2], kernel_size=2)) / 2
        
        bu_features[-1] = self.bifpn_convs[conv_idx](fused)

        return bu_features
