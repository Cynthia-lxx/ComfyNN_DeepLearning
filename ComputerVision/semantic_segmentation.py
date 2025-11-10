import torch
import torch.nn.functional as F
import numpy as np
from typing import Tuple, Dict, Any, List

class SemanticSegmentationNode:
    """
    语义分割节点
    
    对输入图像执行语义分割任务。支持多种基础模型架构，
    可用于CIFAR-10或ImageNet Dogs等任务的像素级分类。
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "segmentation_model": ([
                    "fcn_resnet50", 
                    "fcn_resnet101", 
                    "deeplabv3_resnet50", 
                    "deeplabv3_resnet101",
                    "unet"
                ], {"default": "fcn_resnet50"}),
                "num_classes": ("INT", {"default": 10, "min": 2, "max": 1000, "step": 1}),
                "output_size": ("INT", {"default": 224, "min": 32, "max": 1024, "step": 32})
            },
            "optional": {
                "pretrained": ("BOOLEAN", {"default": True}),
                "background_class": ("BOOLEAN", {"default": True}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01})
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "TENSOR")
    RETURN_NAMES = ("segmentation_masks", "overlay_image", "class_probabilities")
    FUNCTION = "segment"
    CATEGORY = "ComfyNN/DeepLearning/ComputerVision"

    def segment(
        self, 
        image_batch: torch.Tensor,
        segmentation_model: str,
        num_classes: int,
        output_size: int,
        pretrained: bool = True,
        background_class: bool = True,
        confidence_threshold: float = 0.5
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        执行语义分割
        
        Args:
            image_batch: 输入图像批次 [B, H, W, C]
            segmentation_model: 分割模型类型
            num_classes: 类别数量
            output_size: 输出尺寸
            pretrained: 是否使用预训练权重
            background_class: 是否包含背景类别
            confidence_threshold: 置信度阈值
            
        Returns:
            segmentation_masks: 分割掩码 [B, H, W]
            overlay_image: 带分割结果叠加的图像 [B, H, W, C]
            class_probabilities: 各类别的概率分布 [B, num_classes, H, W]
        """
        # 确保输入图像为正确的格式
        batch_size, height, width, channels = image_batch.shape
        
        # 如果包含背景类，则实际类别数+1
        actual_num_classes = num_classes + 1 if background_class else num_classes
        
        # 模拟语义分割过程（实际实现中这里会加载模型并进行推理）
        # 这里为了演示目的，我们将生成伪分割结果
        segmentation_masks = torch.randint(0, actual_num_classes, (batch_size, output_size, output_size))
        
        # 生成伪概率图
        probabilities = torch.rand(batch_size, actual_num_classes, output_size, output_size)
        probabilities = F.softmax(probabilities, dim=1)  # 归一化概率
        
        # 生成带分割结果叠加的图像
        overlay_image = image_batch.clone()
        
        # 如果需要调整输出大小
        if output_size != height or output_size != width:
            overlay_image = F.interpolate(
                overlay_image.permute(0, 3, 1, 2), 
                size=(output_size, output_size), 
                mode='bilinear'
            ).permute(0, 2, 3, 1)
        
        # 应用置信度阈值过滤低置信度预测
        max_probs, _ = torch.max(probabilities, dim=1)
        mask = max_probs > confidence_threshold
        segmentation_masks = segmentation_masks.float() * mask.float()
        
        return (
            segmentation_masks.long(),  # 分割掩码
            overlay_image,              # 叠加图像
            probabilities               # 类别概率
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False


class InstanceSegmentationNode:
    """
    实例分割节点
    
    与语义分割不同，实例分割不仅区分不同的语义类别，
    还能区分同一类别中的不同个体。
    """
    
    @classmethod
    def INPUT_TYPES(cls) -> Dict[str, Dict[str, Any]]:
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "instance_model": ([
                    "maskrcnn_resnet50_fpn",
                    "maskrcnn_resnet101_fpn",
                    "cascade_mask_rcnn"
                ], {"default": "maskrcnn_resnet50_fpn"}),
                "max_detections": ("INT", {"default": 100, "min": 1, "max": 1000, "step": 1})
            },
            "optional": {
                "score_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 1.0, "step": 0.01}),
                "pretrained": ("BOOLEAN", {"default": True})
            }
        }

    RETURN_TYPES = ("MASK", "IMAGE", "TENSOR", "STRING")
    RETURN_NAMES = ("instance_masks", "detection_boxes", "class_scores", "labels")
    FUNCTION = "detect_instances"
    CATEGORY = "ComfyNN/DeepLearning/ComputerVision"

    def detect_instances(
        self,
        image_batch: torch.Tensor,
        instance_model: str,
        max_detections: int,
        score_threshold: float = 0.5,
        nms_threshold: float = 0.3,
        pretrained: bool = True
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[str]]:
        """
        执行实例分割
        
        Args:
            image_batch: 输入图像批次 [B, H, W, C]
            instance_model: 实例分割模型
            max_detections: 最大检测数量
            score_threshold: 分数阈值
            nms_threshold: 非极大抑制阈值
            pretrained: 是否使用预训练权重
            
        Returns:
            instance_masks: 实例掩码 [B, N, H, W]
            detection_boxes: 检测框 [B, N, 4]
            class_scores: 类别分数 [B, N, num_classes]
            labels: 标签列表
        """
        batch_size, height, width, channels = image_batch.shape
        
        # 生成伪实例分割结果
        num_instances = min(max_detections, 10)  # 限制实例数量以便演示
        
        # 生成伪实例掩码
        instance_masks = torch.rand(batch_size, num_instances, height, width) > 0.5
        instance_masks = instance_masks.float()
        
        # 生成伪检测框 (x1, y1, x2, y2)
        detection_boxes = torch.rand(batch_size, num_instances, 4)
        detection_boxes[:, :, 2:] += detection_boxes[:, :, :2]  # 确保x2>x1, y2>y1
        detection_boxes = torch.clamp(detection_boxes, 0, 1)
        
        # 生成伪类别分数
        class_scores = torch.rand(batch_size, num_instances, 10)  # 假设有10个类别
        class_scores = F.softmax(class_scores, dim=-1)
        
        # 生成标签
        labels = [f"instance_{i}" for i in range(num_instances)]
        
        return (
            instance_masks,
            detection_boxes,
            class_scores,
            labels
        )

    @classmethod
    def IS_CHANGED(cls, **kwargs):
        return False


# Node导出映射
NODE_CLASS_MAPPINGS = {
    "SemanticSegmentationNode": SemanticSegmentationNode,
    "InstanceSegmentationNode": InstanceSegmentationNode
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SemanticSegmentationNode": "Semantic Segmentation 🐱",
    "InstanceSegmentationNode": "Instance Segmentation 🐱"
}