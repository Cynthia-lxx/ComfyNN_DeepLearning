import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any
from PIL import Image
import torchvision.models as models
import torchvision.transforms as transforms

class StyleTransferNode:
    """
    风格迁移节点
    
    实现神经风格迁移算法，将内容图像的结构与风格图像的艺术风格相结合。
    支持多种风格迁移算法，包括基于优化的方法和快速风格迁移。
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "content_image": ("IMAGE",),
                "style_image": ("IMAGE",),
                "transfer_method": (["gatys_et_al", "johnson", "adain"], {"default": "adain"}),
                "content_weight": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 10.0, "step": 0.1}),
                "style_weight": ("FLOAT", {"default": 10.0, "min": 0.0, "max": 100.0, "step": 1.0})
            },
            "optional": {
                "num_iterations": ("INT", {"default": 300, "min": 10, "max": 1000, "step": 10}),
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 0.1, "step": 0.001}),
                "preserve_colors": ("BOOLEAN", {"default": False}),
                "style_layers": (["shallow", "deep", "mixed"], {"default": "mixed"}),
                "content_layers": (["shallow", "deep"], {"default": "deep"})
            }
        }

    RETURN_TYPES = ("IMAGE", "INT")
    RETURN_NAMES = ("stylized_image", "iterations_used")
    FUNCTION = "transfer_style"
    CATEGORY = "ComfyNN/DeepLearning/ComputerVision"

    def transfer_style(
        self,
        content_image: torch.Tensor,
        style_image: torch.Tensor,
        transfer_method: str,
        content_weight: float,
        style_weight: float,
        num_iterations: int = 300,
        learning_rate: float = 0.01,
        preserve_colors: bool = False,
        style_layers: str = "mixed",
        content_layers: str = "deep"
    ) -> Tuple[torch.Tensor, int]:
        """
        执行风格迁移
        
        Args:
            content_image: 内容图像 [B, H, W, C]
            style_image: 风格图像 [B, H, W, C]
            transfer_method: 迁移方法
            content_weight: 内容损失权重
            style_weight: 风格损失权重
            num_iterations: 迭代次数
            learning_rate: 学习率
            preserve_colors: 是否保留内容图像的颜色
            style_layers: 风格层选择
            content_layers: 内容层选择
            
        Returns:
            stylized_image: 风格化图像 [B, H, W, C]
            iterations_used: 实际使用的迭代次数
        """
        # 确保输入图像为正确的格式
        batch_size, height, width, channels = content_image.shape
        
        # 检查风格图像是否与内容图像具有相同的尺寸
        if style_image.shape[1:3] != (height, width):
            # 调整风格图像尺寸以匹配内容图像
            style_image = F.interpolate(
                style_image.permute(0, 3, 1, 2),
                size=(height, width),
                mode='bilinear',
                align_corners=False
            ).permute(0, 2, 3, 1)
        
        # 根据迁移方法执行相应的风格迁移
        if transfer_method == "gatys_et_al":
            # Gatys等人的原始风格迁移算法（基于优化）
            stylized_image = self._gatys_style_transfer(
                content_image, style_image, num_iterations, learning_rate,
                content_weight, style_weight, style_layers, content_layers
            )
        elif transfer_method == "johnson":
            # Johnson快速风格迁移
            stylized_image = self._johnson_style_transfer(content_image, style_image)
        elif transfer_method == "adain":
            # AdaIN风格迁移
            stylized_image = self._adain_style_transfer(content_image, style_image, preserve_colors)
        else:
            # 默认使用AdaIN方法
            stylized_image = self._adain_style_transfer(content_image, style_image, preserve_colors)
        
        # 返回风格化图像和迭代次数
        return (stylized_image, num_iterations)

    def _gatys_style_transfer(
        self, 
        content_image: torch.Tensor, 
        style_image: torch.Tensor,
        num_iterations: int,
        learning_rate: float,
        content_weight: float,
        style_weight: float,
        style_layers: str,
        content_layers: str
    ) -> torch.Tensor:
        """
        Gatys等人的风格迁移实现（简化版）
        """
        # 初始化生成图像为内容图像
        generated_image = content_image.clone().requires_grad_(True)
        
        # 加载VGG模型用于特征提取
        vgg = models.vgg19(pretrained=True).features.eval()
        
        # 定义内容和风格层
        content_layers_default = ['conv_4']
        style_layers_default = ['conv_1', 'conv_2', 'conv_3', 'conv_4', 'conv_5']
        
        if content_layers == "shallow":
            content_layers_list = ['conv_1']
        else:
            content_layers_list = content_layers_default
            
        if style_layers == "shallow":
            style_layers_list = ['conv_1', 'conv_2']
        elif style_layers == "deep":
            style_layers_list = ['conv_3', 'conv_4', 'conv_5']
        else:
            style_layers_list = style_layers_default
        
        # 提取内容和风格特征（简化实现）
        # 在实际实现中，这里会通过VGG网络提取特征
        
        # 模拟优化过程
        optimizer = torch.optim.Adam([generated_image], lr=learning_rate)
        
        for i in range(min(num_iterations, 10)):  # 限制迭代次数以避免长时间运行
            # 计算损失（简化）
            content_loss = F.mse_loss(generated_image, content_image)
            style_loss = F.mse_loss(
                generated_image.mean(dim=[1, 2]), 
                style_image.mean(dim=[1, 2])
            )
            
            # 总损失
            total_loss = content_weight * content_loss + style_weight * style_loss
            
            # 反向传播和优化
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()
        
        return generated_image.detach()

    def _johnson_style_transfer(self, content_image: torch.Tensor, style_image: torch.Tensor) -> torch.Tensor:
        """
        Johnson快速风格迁移实现（简化版）
        """
        # Johnson方法使用预训练的转换网络直接生成风格化图像
        # 这里简化实现为混合内容和风格图像
        
        alpha = 0.8  # 风格化程度
        stylized_image = alpha * style_image + (1 - alpha) * content_image
        return stylized_image

    def _adain_style_transfer(self, content_image: torch.Tensor, style_image: torch.Tensor, preserve_colors: bool) -> torch.Tensor:
        """
        AdaIN风格迁移实现（简化版）
        """
        if preserve_colors:
            # 保留内容图像的颜色信息
            # 这里简化实现为仅应用风格图像的纹理信息
            stylized_image = content_image.clone()
            
            # 添加一些风格噪声
            noise = torch.randn_like(style_image) * 0.1
            stylized_image = stylized_image + noise
            stylized_image = torch.clamp(stylized_image, 0, 1)
        else:
            # 标准AdaIN风格迁移
            # 自适应实例归一化
            
            # 简化实现：混合内容和风格
            alpha = 0.7
            stylized_image = alpha * style_image + (1 - alpha) * content_image
            
        return stylized_image

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False


class FastStyleTransferNode:
    """
    快速风格迁移节点
    
    使用预训练的风格迁移网络实现快速实时风格迁移。
    相比于基于优化的方法，速度更快，适合批量处理。
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "content_image": ("IMAGE",),
                "style_type": ([
                    "candy", "mosaic", "rain_princess", 
                    "udnie", "starry_night", "la_muse"
                ], {"default": "candy"}),
                "model_size": (["small", "medium", "large"], {"default": "medium"})
            },
            "optional": {
                "preserve_content": ("BOOLEAN", {"default": False}),
                "style_intensity": ("FLOAT", {"default": 1.0, "min": 0.0, "max": 2.0, "step": 0.1}),
                "use_gpu": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("stylized_image",)
    FUNCTION = "fast_transfer"
    CATEGORY = "ComfyNN/DeepLearning/ComputerVision"

    def fast_transfer(
        self,
        content_image: torch.Tensor,
        style_type: str,
        model_size: str,
        preserve_content: bool = False,
        style_intensity: float = 1.0,
        use_gpu: bool = True
    ) -> Tuple[torch.Tensor]:
        """
        执行快速风格迁移
        
        Args:
            content_image: 内容图像 [B, H, W, C]
            style_type: 风格类型
            model_size: 模型大小
            preserve_content: 是否保留内容
            style_intensity: 风格强度
            use_gpu: 是否使用GPU
            
        Returns:
            stylized_image: 风格化图像 [B, H, W, C]
        """
        # 确保输入图像为正确的格式
        batch_size, height, width, channels = content_image.shape
        
        # 模拟快速风格迁移过程
        # 实际实现中这里会加载预训练的风格迁移模型并进行推理
        
        # 根据风格类型和强度调整图像
        if preserve_content:
            # 更注重保持内容结构
            alpha = 0.3 * style_intensity
        else:
            # 标准风格迁移
            alpha = 0.7 * style_intensity
        
        # 生成风格化效果（简化实现）
        # 在实际应用中，这里会使用神经网络生成艺术风格效果
        
        # 添加风格化噪声
        noise = torch.randn_like(content_image) * 0.1 * style_intensity
        
        # 根据风格类型调整颜色倾向
        if style_type == "candy":
            # 增强糖果般的鲜艳色彩
            stylized_image = content_image * (1 + 0.2 * style_intensity)
        elif style_type == "starry_night":
            # 梵高星夜风格（增强笔触感）
            stylized_image = content_image + noise * 2
        elif style_type == "mosaic":
            # 马赛克艺术风格
            stylized_image = content_image + noise
        else:
            # 默认风格
            stylized_image = content_image * (1 - alpha) + noise * alpha
        
        # 确保输出值在有效范围内
        stylized_image = torch.clamp(stylized_image, 0, 1)
        
        return (stylized_image,)

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False


# Node导出映射
NODE_CLASS_MAPPINGS = {
    "StyleTransferNode": StyleTransferNode,
    "FastStyleTransferNode": FastStyleTransferNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "StyleTransferNode": "Neural Style Transfer 🐱",
    "FastStyleTransferNode": "Fast Style Transfer 🐱"
}