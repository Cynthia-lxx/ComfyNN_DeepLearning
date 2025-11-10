import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any

class TransposedConv2DNode:
    """
    转置卷积（反卷积）节点
    
    转置卷积常用于上采样操作，特别是在语义分割和生成模型中。
    它可以将低分辨率特征图上采样到高分辨率。
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "in_channels": ("INT", {"default": 64, "min": 1, "max": 1024, "step": 1}),
                "out_channels": ("INT", {"default": 32, "min": 1, "max": 1024, "step": 1}),
                "kernel_size": ("INT", {"default": 4, "min": 1, "max": 15, "step": 2}),
                "stride": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "padding": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1}),
            },
            "optional": {
                "output_padding": ("INT", {"default": 0, "min": 0, "max": 8, "step": 1}),
                "dilation": ("INT", {"default": 1, "min": 1, "max": 8, "step": 1}),
                "activation": (["relu", "leaky_relu", "sigmoid", "tanh", "none"], {"default": "relu"}),
                "use_bias": ("BOOLEAN", {"default": True}),
                "groups": ("INT", {"default": 1, "min": 1, "max": 128, "step": 1})
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output_tensor",)
    FUNCTION = "transposed_conv2d"
    CATEGORY = "ComfyNN/DeepLearning/ComputerVision"

    def transposed_conv2d(
        self,
        input_tensor: torch.Tensor,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 2,
        padding: int = 1,
        output_padding: int = 0,
        dilation: int = 1,
        activation: str = "relu",
        use_bias: bool = True,
        groups: int = 1
    ) -> Tuple[torch.Tensor]:
        """
        执行2D转置卷积操作
        
        Args:
            input_tensor: 输入张量 [N, C, H, W]
            in_channels: 输入通道数
            out_channels: 输出通道数
            kernel_size: 卷积核大小
            stride: 步长
            padding: 填充
            output_padding: 输出填充
            dilation: 膨胀系数
            activation: 激活函数类型
            use_bias: 是否使用偏置
            groups: 分组卷积组数
            
        Returns:
            output_tensor: 输出张量 [N, out_channels, H_out, W_out]
        """
        # 确保输入张量维度正确
        if input_tensor.dim() != 4:
            raise ValueError(f"输入张量应为4维 [N, C, H, W]，当前维度: {input_tensor.dim()}")
        
        # 创建转置卷积层
        transposed_conv = nn.ConvTranspose2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            output_padding=output_padding,
            dilation=dilation,
            bias=use_bias,
            groups=groups
        )
        
        # 应用转置卷积
        output_tensor = transposed_conv(input_tensor)
        
        # 应用激活函数
        if activation == "relu":
            output_tensor = F.relu(output_tensor)
        elif activation == "leaky_relu":
            output_tensor = F.leaky_relu(output_tensor)
        elif activation == "sigmoid":
            output_tensor = torch.sigmoid(output_tensor)
        elif activation == "tanh":
            output_tensor = torch.tanh(output_tensor)
        # "none" 情况下不应用激活函数
        
        return (output_tensor,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False


class MultiScaleTransposedConvNode:
    """
    多尺度转置卷积节点
    
    同时使用多个不同尺度的转置卷积核进行上采样，
    然后将结果融合以获得更丰富的特征表示。
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "base_channels": ("INT", {"default": 64, "min": 1, "max": 512, "step": 1}),
                "target_channels": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "base_kernel_size": ("INT", {"default": 4, "min": 2, "max": 15, "step": 2}),
            },
            "optional": {
                "num_scales": ("INT", {"default": 3, "min": 1, "max": 5, "step": 1}),
                "stride": ("INT", {"default": 2, "min": 1, "max": 8, "step": 1}),
                "padding": ("INT", {"default": 1, "min": 0, "max": 8, "step": 1}),
                "activation": (["relu", "leaky_relu", "sigmoid", "tanh", "none"], {"default": "relu"}),
                "use_bias": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("output_tensor",)
    FUNCTION = "multi_scale_transposed_conv"
    CATEGORY = "ComfyNN/DeepLearning/ComputerVision"

    def multi_scale_transposed_conv(
        self,
        input_tensor: torch.Tensor,
        base_channels: int,
        target_channels: int,
        base_kernel_size: int,
        num_scales: int = 3,
        stride: int = 2,
        padding: int = 1,
        activation: str = "relu",
        use_bias: bool = True
    ) -> Tuple[torch.Tensor]:
        """
        执行多尺度转置卷积操作
        
        Args:
            input_tensor: 输入张量 [N, C, H, W]
            base_channels: 基础通道数
            target_channels: 目标通道数
            base_kernel_size: 基础卷积核大小
            num_scales: 尺度数量
            stride: 步长
            padding: 填充
            activation: 激活函数类型
            use_bias: 是否使用偏置
            
        Returns:
            output_tensor: 输出张量 [N, target_channels, H_out, W_out]
        """
        # 确保输入张量维度正确
        if input_tensor.dim() != 4:
            raise ValueError(f"输入张量应为4维 [N, C, H, W]，当前维度: {input_tensor.dim()}")
        
        batch_size, channels, height, width = input_tensor.shape
        
        # 创建多个不同尺度的转置卷积层
        transposed_convs = []
        for i in range(num_scales):
            kernel_size = base_kernel_size + 2 * i  # 逐步增加核大小
            out_channels = max(1, target_channels // num_scales)  # 分配输出通道
            
            conv = nn.ConvTranspose2d(
                in_channels=channels,
                out_channels=out_channels,
                kernel_size=kernel_size,
                stride=stride,
                padding=padding,
                bias=use_bias
            )
            transposed_convs.append(conv)
        
        # 应用所有转置卷积
        outputs = []
        for conv in transposed_convs:
            out = conv(input_tensor)
            outputs.append(out)
        
        # 调整所有输出到相同尺寸（使用最大的尺寸）
        max_height = max([out.shape[2] for out in outputs])
        max_width = max([out.shape[3] for out in outputs])
        
        resized_outputs = []
        for out in outputs:
            if out.shape[2] != max_height or out.shape[3] != max_width:
                out = F.interpolate(out, size=(max_height, max_width), mode='bilinear', align_corners=False)
            resized_outputs.append(out)
        
        # 合并所有输出
        combined_output = torch.cat(resized_outputs, dim=1)
        
        # 如果合并后的通道数与目标通道数不匹配，添加一个1x1卷积调整通道数
        if combined_output.shape[1] != target_channels:
            channel_adjust = nn.Conv2d(
                in_channels=combined_output.shape[1],
                out_channels=target_channels,
                kernel_size=1,
                bias=use_bias
            )
            combined_output = channel_adjust(combined_output)
        
        # 应用激活函数
        if activation == "relu":
            combined_output = F.relu(combined_output)
        elif activation == "leaky_relu":
            combined_output = F.leaky_relu(combined_output)
        elif activation == "sigmoid":
            combined_output = torch.sigmoid(combined_output)
        elif activation == "tanh":
            combined_output = torch.tanh(combined_output)
        # "none" 情况下不应用激活函数
        
        return (combined_output,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False


# Node导出映射
NODE_CLASS_MAPPINGS = {
    "TransposedConv2DNode": TransposedConv2DNode,
    "MultiScaleTransposedConvNode": MultiScaleTransposedConvNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "TransposedConv2DNode": "Transposed Conv 2D 🐱",
    "MultiScaleTransposedConvNode": "Multi-Scale Transposed Conv 🐱"
}