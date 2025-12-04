"""
YOLOv12-BDA Custom Modules (YAML-Compatible Version)
Based on: "YOLOv12-BDA: A Dynamic Multi-Scale Architecture for Small Weed Detection in Sesame Fields"

This file implements:
1. HGStem - Shared stem for dual backbone
2. HGBlock - Dense feature extraction block
3. DLUBlock - Dynamic Learning Unit for adaptive fusion
4. DGCSBlock - Dynamic Grouped Convolution and Channel Mixing
5. DASINeck - Dynamic Adaptive Scale-aware Interactive Neck

YAML Compatibility Notes:
- Modules accept arguments in format: [arg1, arg2, ...]
- Forward methods handle both single and multiple inputs
- Compatible with Ultralytics YOLO parsing system
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ============================================================================
# 1. HGStem - Shared Stem for Dual Backbone
# ============================================================================

class HGStem(nn.Module):
    """
    HGStem: Shared stem structure for dual-backbone architecture.
    
    From paper Section 2.2.1:
    - Reduces computational load by sharing initial feature extraction
    - Equations (1-4) in paper
    
    Args:
        c1: Input channels (from YAML)
        c2: Output channels (from YAML)
    """
    def __init__(self, c1=3, c2=32):
        super().__init__()
        
        # Fstem1 = ReLU(Conv_s=2_3x3(X0))
        self.conv1 = nn.Conv2d(c1, c2, 3, 2, 1, bias=False)
        self.bn1 = nn.BatchNorm2d(c2)
        self.act1 = nn.SiLU()
        
        # Branch 1: MaxPool
        self.maxpool = nn.MaxPool2d(2, 1, 0)
        
        # Branch 2: Conv path
        self.conv2_1 = nn.Conv2d(c2, c2, 2, 1, 0, bias=False)
        self.bn2_1 = nn.BatchNorm2d(c2)
        self.act2_1 = nn.SiLU()
        
        self.conv2_2 = nn.Conv2d(c2, c2, 2, 1, 0, bias=False)
        self.bn2_2 = nn.BatchNorm2d(c2)
        self.act2_2 = nn.SiLU()
        
        # Final fusion
        self.conv3 = nn.Conv2d(c2 * 2, c2, 3, 2, 1, bias=False)
        self.bn3 = nn.BatchNorm2d(c2)
        self.act3 = nn.SiLU()
        
        self.conv4 = nn.Conv2d(c2, c2, 1, 1, 0, bias=False)
        self.bn4 = nn.BatchNorm2d(c2)
        self.act4 = nn.SiLU()
    
    def forward(self, x):
        # Fstem1: Initial convolution with stride 2
        x = self.act1(self.bn1(self.conv1(x)))
        
        # Branch 1: MaxPool path
        branch1 = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        branch1 = self.maxpool(branch1)
        
        # Branch 2: Conv path  
        branch2 = F.pad(x, (0, 1, 0, 1), mode='constant', value=0)
        branch2 = self.act2_1(self.bn2_1(self.conv2_1(branch2)))
        branch2 = self.act2_2(self.bn2_2(self.conv2_2(branch2)))
        
        # Ensure both branches have same spatial size
        h1, w1 = branch1.shape[2], branch1.shape[3]
        h2, w2 = branch2.shape[2], branch2.shape[3]
        
        if h1 != h2 or w1 != w2:
            target_h, target_w = min(h1, h2), min(w1, w2)
            branch1 = branch1[:, :, :target_h, :target_w]
            branch2 = branch2[:, :, :target_h, :target_w]
        
        # Concatenate and fuse
        x = torch.cat([branch1, branch2], dim=1)
        x = self.act3(self.bn3(self.conv3(x)))
        x = self.act4(self.bn4(self.conv4(x)))
        
        return x


# ============================================================================
# 2. HGBlock - Dense Feature Extraction
# ============================================================================

class HGBlock(nn.Module):
    """
    HGBlock: Densely connected block for detail-oriented feature extraction.
    
    From paper Section 2.2.1, Equations (5-7):
    - Uses dense connections for multi-scale feature aggregation
    - Applies SE (Squeeze-and-Excitation) for channel attention
    
    Args:
        c1: Input channels (inferred from previous layer)
        c2: Output channels (from YAML)
        n: Number of convolutional units (from YAML, default=3)
        shortcut: Use residual connection (from YAML, default=True)
    """
    def __init__(self, c1, c2, n=3, shortcut=True):
        super().__init__()
        self.n = n
        self.shortcut = shortcut
        
        # n convolutional units
        self.convs = nn.ModuleList()
        for i in range(n):
            self.convs.append(
                nn.Sequential(
                    nn.Conv2d(c1 if i == 0 else c2, 
                             c2, 3, 1, 1, bias=False),
                    nn.BatchNorm2d(c2),
                    nn.SiLU()
                )
            )
        
        # Channel attention (SE module)
        concat_channels = c1 + n * c2
        reduction = max(concat_channels // 4, 16)  # Ensure minimum reduction
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(concat_channels, reduction, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(reduction, concat_channels, 1, bias=False),
            nn.Sigmoid()
        )
        
        # Final 1x1 conv to adjust channels
        self.final_conv = nn.Conv2d(concat_channels, c2, 1, 1, 0, bias=False)
        self.final_bn = nn.BatchNorm2d(c2)
        
    def forward(self, x):
        outputs = [x]
        
        # Dense connections: each layer receives all previous features
        for conv in self.convs:
            x = conv(outputs[-1])
            outputs.append(x)
        
        # Concatenate all features
        x_concat = torch.cat(outputs, dim=1)
        
        # Apply SE attention
        se_weight = self.se(x_concat)
        x_concat = x_concat * se_weight
        
        # Final fusion
        out = self.final_bn(self.final_conv(x_concat))
        
        # Residual connection if shapes match
        if self.shortcut and outputs[0].shape == out.shape:
            out = out + outputs[0]
        
        return out


# ============================================================================
# 3. DLUBlock - Dynamic Learning Unit
# ============================================================================

class DLUBlock(nn.Module):
    """
    DLU: Dynamic Learning Unit for adaptive dual-backbone fusion.
    
    From paper Section 2.2.1, Figure 2:
    - Generates dynamic weights using sigmoid activation
    - Performs element-wise weighted fusion
    - α = sigmoid(MLP([GAP(feat_A), GAP(feat_B)]))
    - fused = α * feat_A + (1 - α) * feat_B
    
    Args:
        c1: Input channels (inferred or provided, should match both inputs)
        
    YAML Compatibility:
    - Can accept either [[layerA, layerB], ...] format
    - Forward handles list of tensors or single tensor
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        # c2 is ignored, output channels = input channels
        channels = c1
        
        # Global pooling for weight generation
        self.gap = nn.AdaptiveAvgPool2d(1)
        
        # MLP for dynamic weight generation
        # Takes concatenated GAP features from both branches
        self.mlp = nn.Sequential(
            nn.Conv2d(channels * 2, channels, 1, bias=False),
            nn.SiLU(),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.Sigmoid()  # α ∈ [0, 1]
        )
        
        # Feature refinement before fusion
        self.conv_a = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        self.conv_b = nn.Sequential(
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
        # Final fusion conv
        self.fusion = nn.Sequential(
            nn.Conv2d(channels, channels, 3, 1, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.SiLU()
        )
        
    def forward(self, x):
        """
        Forward pass - handles both single tensor and list of tensors
        
        Args:
            x: Either a single tensor or list of [feat_a, feat_b]
        """
        # Handle YAML input format: x can be a list [feat_a, feat_b]
        if isinstance(x, list):
            if len(x) != 2:
                raise ValueError(f"DLUBlock expects 2 inputs, got {len(x)}")
            feat_a, feat_b = x
        else:
            # Fallback: if single input, return as-is (shouldn't happen in dual-backbone)
            return x
        
        # Ensure both features have same spatial dimensions
        if feat_a.shape != feat_b.shape:
            # Resize feat_b to match feat_a
            feat_b = F.interpolate(feat_b, size=feat_a.shape[2:], 
                                  mode='bilinear', align_corners=False)
        
        # Refine features
        feat_a_refined = self.conv_a(feat_a)
        feat_b_refined = self.conv_b(feat_b)
        
        # Generate dynamic weights using GAP + MLP
        gap_a = self.gap(feat_a_refined)
        gap_b = self.gap(feat_b_refined)
        gap_concat = torch.cat([gap_a, gap_b], dim=1)
        
        # α = sigmoid(MLP([GAP(A), GAP(B)]))
        alpha = self.mlp(gap_concat)
        
        # Adaptive fusion: fused = α * A + (1-α) * B
        fused = alpha * feat_a_refined + (1 - alpha) * feat_b_refined
        
        # Final refinement
        output = self.fusion(fused)
        
        return output


# ============================================================================
# 4. DGCSBlock - Dynamic Grouped Conv + Channel Mixing
# ============================================================================

class DGCSBlock(nn.Module):
    """
    DGCS: Dynamic Grouped Convolution and Channel-Mixing Block.
    
    From paper Section 2.2.2:
    - Replaces C3K2 blocks in backbone
    - 1:3 channel split ratio
    - Grouped convolution on smaller split
    - Channel shuffling for information exchange
    
    Args:
        c1: Input channels (inferred from previous layer)
        c2: Output channels (from YAML, usually same as c1)
        shortcut: Use residual connection (from YAML, default=True)
    """
    def __init__(self, c1, c2=None, shortcut=True):
        super().__init__()
        c2 = c2 or c1  # Output channels default to input channels
        self.shortcut = shortcut and c1 == c2
        
        # Initial 1x1 conv for dimension transformation
        self.conv1 = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
        # Calculate split channels (1:3 ratio)
        # 25% goes through grouped conv, 75% passes through
        self.split_channels = c2 // 4
        self.pass_channels = c2 - self.split_channels
        
        # Grouped convolution (depthwise)
        self.grouped_conv = nn.Sequential(
            nn.Conv2d(self.split_channels, self.split_channels, 
                     3, 1, 1, groups=self.split_channels, bias=False),
            nn.BatchNorm2d(self.split_channels),
            nn.SiLU()
        )
        
        # Channel mixing (after shuffle)
        self.mix_conv1 = nn.Sequential(
            nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
        self.mix_conv2 = nn.Sequential(
            nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
        # Final output conv
        self.conv_out = nn.Sequential(
            nn.Conv2d(c2, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2)
        )
        
    def channel_shuffle(self, x, groups=4):
        """Channel shuffling operation"""
        batch, channels, height, width = x.size()
        channels_per_group = channels // groups
        
        x = x.view(batch, groups, channels_per_group, height, width)
        x = x.transpose(1, 2).contiguous()
        x = x.view(batch, channels, height, width)
        
        return x
    
    def forward(self, x):
        identity = x
        
        # Initial transformation
        x = self.conv1(x)
        
        # Split into two parts (1:3 ratio)
        x_grouped = x[:, :self.split_channels, :, :]
        x_pass = x[:, self.split_channels:, :, :]
        
        # Apply grouped convolution to smaller split
        x_grouped = self.grouped_conv(x_grouped)
        
        # Concatenate
        x = torch.cat([x_grouped, x_pass], dim=1)
        
        # Channel shuffle
        x = self.channel_shuffle(x, groups=4)
        
        # Channel mixing with residual
        x_mixed = self.mix_conv1(x)
        x_mixed = self.mix_conv2(x_mixed)
        x = x + x_mixed  # Internal residual
        
        # Final output
        x = self.conv_out(x)
        
        # Shortcut connection if dimensions match
        if self.shortcut:
            x = x + identity
        
        return x


# ============================================================================
# 5. DASINeck - Dynamic Adaptive Scale-aware Interactive Neck
# ============================================================================

class DASINeck(nn.Module):
    """
    DASI: Dynamic Adaptive Scale-aware Interactive module.
    
    From paper Section 2.2.3, Equations (8-11), Figure 4:
    - Receives three feature maps (high, current, low resolution)
    - Partitions into 4 channel sub-blocks
    - Dynamic weighting: α = sigmoid(mi)
    - Adaptive fusion: m'i = α ⊙ li + (1-α) ⊙ hi
    
    Args:
        c1: Input channels (from YAML, should be same for all 3 inputs)
        
    YAML Compatibility:
    - Expects input as list: [high_res, current_res, low_res]
    """
    def __init__(self, c1, c2=None):
        super().__init__()
        c2 = c2 or c1  # Output channels = input channels
        self.channels = c1
        
        # Scale alignment operations
        # For high-resolution features (downsample)
        self.align_high = nn.Sequential(
            nn.Conv2d(c1, c1, 3, 2, 1, bias=False),
            nn.BatchNorm2d(c1),
            nn.SiLU()
        )
        
        # Weight generation for each channel group
        channels_per_group = c1 // 4
        self.weight_gens = nn.ModuleList([
            nn.Sequential(
                nn.Conv2d(channels_per_group, channels_per_group, 1, bias=False),
                nn.BatchNorm2d(channels_per_group),
                nn.Sigmoid()
            ) for _ in range(4)
        ])
        
        # Final processing
        self.final_conv = nn.Sequential(
            nn.Conv2d(c1, c2, 1, 1, 0, bias=False),
            nn.BatchNorm2d(c2),
            nn.SiLU()
        )
        
    def forward(self, x):
        """
        Forward pass - handles list of 3 feature maps
        
        Args:
            x: List of [high_res, current_res, low_res] feature maps
        """
        # Handle YAML input format
        if isinstance(x, list):
            if len(x) != 3:
                raise ValueError(f"DASINeck expects 3 inputs, got {len(x)}")
            high_res, current_res, low_res = x
        else:
            # Fallback: single input
            return x
        
        batch, channels, H, W = current_res.shape
        
        # Step 1: Align all features to current resolution
        # High-res: downsample
        hi = self.align_high(high_res)
        
        # Low-res: upsample
        li = F.interpolate(low_res, size=(H, W), mode='bilinear', align_corners=False)
        
        # Current: no alignment
        mi = current_res
        
        # Step 2: Partition into 4 equal channel groups
        channels_per_group = channels // 4
        
        fused_parts = []
        
        for i in range(4):
            start_ch = i * channels_per_group
            end_ch = (i + 1) * channels_per_group
            
            # Extract i-th partition
            hi_part = hi[:, start_ch:end_ch, :, :]
            li_part = li[:, start_ch:end_ch, :, :]
            mi_part = mi[:, start_ch:end_ch, :, :]
            
            # Generate dynamic weight: α = sigmoid(mi)
            alpha = self.weight_gens[i](mi_part)
            
            # Adaptive fusion: m'i = α ⊙ li + (1-α) ⊙ hi
            fused_part = alpha * li_part + (1 - alpha) * hi_part
            
            fused_parts.append(fused_part)
        
        # Step 3: Concatenate all fused sub-blocks
        fused = torch.cat(fused_parts, dim=1)
        
        # Step 4: Final convolution
        output = self.final_conv(fused)
        
        return output


# ============================================================================
# Utility: Conv module for compatibility
# ============================================================================

class Conv(nn.Module):
    """Standard convolution with BatchNorm and activation"""
    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, act=True):
        super().__init__()
        if p is None:
            p = k // 2
        self.conv = nn.Conv2d(c1, c2, k, s, p, groups=g, bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = nn.SiLU() if act else nn.Identity()
    
    def forward(self, x):
        return self.act(self.bn(self.conv(x)))


# ============================================================================
# Export all modules
# ============================================================================

__all__ = ['HGStem', 'HGBlock', 'DLUBlock', 'DGCSBlock', 'DASINeck', 'Conv']