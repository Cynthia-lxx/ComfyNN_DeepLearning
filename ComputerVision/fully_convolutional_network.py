import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List

class FCNNode:
    """
    全卷积网络 (Fully Convolutional Network) 节点
    
    FCN是用于语义分割的经典网络架构，将传统CNN中的全连接层替换为卷积层，
    使得网络可以接受任意尺寸的输入，并产生相应尺寸的输出。
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "fcn_variant": (["fcn8s", "fcn16s", "fcn32s"], {"default": "fcn8s"}),
                "num_classes": ("INT", {"default": 10, "min": 2, "max": 1000, "step": 1}),
                "backbone": (["vgg16", "resnet50", "resnet101"], {"default": "vgg16"})
            },
            "optional": {
                "pretrained": ("BOOLEAN", {"default": True}),
                "upsample_method": (["bilinear", "transposed_conv"], {"default": "bilinear"}),
                "output_stride": ("INT", {"default": 32, "min": 8, "max": 64, "step": 8}),
                "use_dropout": ("BOOLEAN", {"default": False}),
                "dropout_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.9, "step": 0.05})
            }
        }

    RETURN_TYPES = ("MASK", "TENSOR", "IMAGE")
    RETURN_NAMES = ("segmentation_masks", "class_probabilities", "overlay_image")
    FUNCTION = "fcn_segment"
    CATEGORY = "ComfyNN/DeepLearning/ComputerVision"

    def fcn_segment(
        self,
        image_batch: torch.Tensor,
        fcn_variant: str,
        num_classes: int,
        backbone: str,
        pretrained: bool = True,
        upsample_method: str = "bilinear",
        output_stride: int = 32,
        use_dropout: bool = False,
        dropout_rate: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行FCN语义分割
        
        Args:
            image_batch: 输入图像批次 [B, H, W, C]
            fcn_variant: FCN变体 (fcn8s, fcn16s, fcn32s)
            num_classes: 类别数量
            backbone: 骨干网络
            pretrained: 是否使用预训练权重
            upsample_method: 上采样方法
            output_stride: 输出步长
            use_dropout: 是否使用dropout
            dropout_rate: Dropout比率
            
        Returns:
            segmentation_masks: 分割掩码 [B, H, W]
            class_probabilities: 类别概率 [B, num_classes, H, W]
            overlay_image: 叠加图像 [B, H, W, C]
        """
        # 确保输入图像为正确的格式
        batch_size, height, width, channels = image_batch.shape
        
        # 转换图像格式为 [B, C, H, W]
        if channels in [1, 3]:  # 灰度图或RGB图
            input_tensor = image_batch.permute(0, 3, 1, 2)
        else:
            raise ValueError(f"不支持的图像通道数: {channels}")
        
        # 模拟FCN推理过程
        # 实际实现中这里会加载预训练的FCN模型并进行推理
        
        # 特征提取（模拟）
        features = torch.rand(batch_size, 512, height // output_stride, width // output_stride)
        
        # 分类头（模拟）
        logits = torch.rand(batch_size, num_classes, height // output_stride, width // output_stride)
        
        # 上采样到原始尺寸
        if upsample_method == "bilinear":
            upsampled_logits = F.interpolate(
                logits, 
                size=(height, width), 
                mode='bilinear', 
                align_corners=False
            )
        else:  # transposed_conv
            # 使用转置卷积上采样
            transposed_conv = nn.ConvTranspose2d(
                in_channels=num_classes,
                out_channels=num_classes,
                kernel_size=output_stride,
                stride=output_stride // 8,  # 简化处理
                padding=output_stride // 16
            )
            upsampled_logits = transposed_conv(logits)
            # 调整到目标尺寸
            upsampled_logits = F.interpolate(
                upsampled_logits,
                size=(height, width),
                mode='bilinear',
                align_corners=False
            )
        
        # 应用softmax获取概率分布
        class_probabilities = F.softmax(upsampled_logits, dim=1)
        
        # 获取分割掩码（最大概率类别）
        segmentation_masks = torch.argmax(class_probabilities, dim=1)
        
        # 生成叠加图像（将分割结果叠加到原始图像上）
        overlay_image = image_batch.clone()
        
        # 添加一些可视化效果（简化实现）
        # 在实际应用中，这里会根据分割结果为图像添加颜色编码
        alpha = 0.7
        overlay_image = overlay_image * alpha + (1 - alpha) * overlay_image
        
        return (
            segmentation_masks.long(),  # 分割掩码
            class_probabilities,        # 类别概率
            overlay_image               # 叠加图像
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False


class EncoderDecoderNode:
    """
    编码器-解码器结构节点
    
    这是FCN的通用架构，包含编码器（下采样）和解码器（上采样）部分，
    广泛应用于语义分割任务中。
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "num_classes": ("INT", {"default": 10, "min": 2, "max": 1000, "step": 1}),
                "encoder_depth": ("INT", {"default": 5, "min": 3, "max": 8, "step": 1}),
                "base_channels": ("INT", {"default": 64, "min": 16, "max": 256, "step": 16})
            },
            "optional": {
                "decoder_type": (["upsample", "transposed_conv", "unet_style"], {"default": "upsample"}),
                "use_skip_connections": ("BOOLEAN", {"default": True}),
                "activation": (["relu", "leaky_relu", "elu"], {"default": "relu"}),
                "use_batch_norm": ("BOOLEAN", {"default": True}),
                "dropout_rate": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.9, "step": 0.05})
            }
        }

    RETURN_TYPES = ("MASK", "TENSOR", "TENSOR")
    RETURN_NAMES = ("segmentation_masks", "class_probabilities", "feature_map")
    FUNCTION = "encode_decode"
    CATEGORY = "ComfyNN/DeepLearning/ComputerVision"

    def encode_decode(
        self,
        input_tensor: torch.Tensor,
        num_classes: int,
        encoder_depth: int,
        base_channels: int,
        decoder_type: str = "upsample",
        use_skip_connections: bool = True,
        activation: str = "relu",
        use_batch_norm: bool = True,
        dropout_rate: float = 0.0
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行编码器-解码器结构的前向传播
        
        Args:
            input_tensor: 输入张量 [B, C, H, W]
            num_classes: 类别数量
            encoder_depth: 编码器深度
            base_channels: 基础通道数
            decoder_type: 解码器类型
            use_skip_connections: 是否使用跳跃连接
            activation: 激活函数类型
            use_batch_norm: 是否使用批归一化
            dropout_rate: Dropout比率
            
        Returns:
            segmentation_masks: 分割掩码 [B, H, W]
            class_probabilities: 类别概率 [B, num_classes, H, W]
            feature_map: 特征图 [B, C', H, W]
        """
        # 确保输入张量维度正确
        if input_tensor.dim() != 4:
            raise ValueError(f"输入张量应为4维 [B, C, H, W]，当前维度: {input_tensor.dim()}")
        
        batch_size, channels, height, width = input_tensor.shape
        
        # 编码器阶段（下采样）
        encoder_features = []
        x = input_tensor
        
        for i in range(encoder_depth):
            out_channels = base_channels * (2 ** min(i, 4))  # 逐步增加通道数
            
            # 卷积块
            conv1 = nn.Conv2d(
                in_channels=x.shape[1],
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
            
            conv2 = nn.Conv2d(
                in_channels=out_channels,
                out_channels=out_channels,
                kernel_size=3,
                padding=1
            )
            
            x = conv1(x)
            if use_batch_norm:
                bn = nn.BatchNorm2d(out_channels)
                x = bn(x)
            
            if activation == "relu":
                x = F.relu(x)
            elif activation == "leaky_relu":
                x = F.leaky_relu(x)
            elif activation == "elu":
                x = F.elu(x)
            
            x = conv2(x)
            if use_batch_norm:
                bn = nn.BatchNorm2d(out_channels)
                x = bn(x)
            
            if activation == "relu":
                x = F.relu(x)
            elif activation == "leaky_relu":
                x = F.leaky_relu(x)
            elif activation == "elu":
                x = F.elu(x)
            
            # 保存跳跃连接特征
            if use_skip_connections:
                encoder_features.append(x)
            
            # 下采样（最后一个阶段不进行下采样）
            if i < encoder_depth - 1:
                x = F.max_pool2d(x, kernel_size=2, stride=2)
        
        # 解码器阶段（上采样）
        for i in range(encoder_depth - 1):
            target_channels = base_channels * (2 ** min(encoder_depth - 2 - i, 4))
            
            # 上采样
            if decoder_type == "upsample":
                x = F.interpolate(x, scale_factor=2, mode='bilinear', align_corners=False)
            elif decoder_type == "transposed_conv":
                transposed_conv = nn.ConvTranspose2d(
                    in_channels=x.shape[1],
                    out_channels=target_channels,
                    kernel_size=2,
                    stride=2
                )
                x = transposed_conv(x)
            # unet_style 会结合跳跃连接
            
            # 如果使用跳跃连接，合并特征
            if use_skip_connections and len(encoder_features) > 0:
                skip_feature = encoder_features.pop()
                # 调整尺寸以匹配
                if x.shape[2:] != skip_feature.shape[2:]:
                    x = F.interpolate(x, size=skip_feature.shape[2:], mode='bilinear', align_corners=False)
                x = torch.cat([x, skip_feature], dim=1)
            
            # 卷积块
            conv1 = nn.Conv2d(
                in_channels=x.shape[1],
                out_channels=target_channels,
                kernel_size=3,
                padding=1
            )
            
            conv2 = nn.Conv2d(
                in_channels=target_channels,
                out_channels=target_channels,
                kernel_size=3,
                padding=1
            )
            
            x = conv1(x)
            if use_batch_norm:
                bn = nn.BatchNorm2d(target_channels)
                x = bn(x)
            
            if activation == "relu":
                x = F.relu(x)
            elif activation == "leaky_relu":
                x = F.leaky_relu(x)
            elif activation == "elu":
                x = F.elu(x)
            
            if dropout_rate > 0:
                x = F.dropout(x, p=dropout_rate)
            
            x = conv2(x)
            if use_batch_norm:
                bn = nn.BatchNorm2d(target_channels)
                x = bn(x)
            
            if activation == "relu":
                x = F.relu(x)
            elif activation == "leaky_relu":
                x = F.leaky_relu(x)
            elif activation == "elu":
                x = F.elu(x)
            
            if dropout_rate > 0:
                x = F.dropout(x, p=dropout_rate)
        
        # 最终分类层
        final_conv = nn.Conv2d(
            in_channels=x.shape[1],
            out_channels=num_classes,
            kernel_size=1
        )
        logits = final_conv(x)
        
        # 上采样到原始输入尺寸
        if logits.shape[2:] != (height, width):
            logits = F.interpolate(logits, size=(height, width), mode='bilinear', align_corners=False)
        
        # 应用softmax获取概率分布
        class_probabilities = F.softmax(logits, dim=1)
        
        # 获取分割掩码（最大概率类别）
        segmentation_masks = torch.argmax(class_probabilities, dim=1)
        
        return (
            segmentation_masks.long(),  # 分割掩码
            class_probabilities,        # 类别概率
            x                           # 最后一层特征图
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False


# Node导出映射
NODE_CLASS_MAPPINGS = {
    "FCNNode": FCNNode,
    "EncoderDecoderNode": EncoderDecoderNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FCNNode": "Fully Convolutional Network 🐱",
    "EncoderDecoderNode": "Encoder-Decoder Network 🐱"
}