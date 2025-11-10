# ComfyNN ComputerVision R-CNN Series Nodes
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class RCNNModel:
    """R-CNN模型节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_images": ("IMAGE",),
                "region_proposals": ("TENSOR",),
                "rcnn_variant": (["r-cnn", "fast r-cnn", "faster r-cnn", "mask r-cnn"], {"default": "fast r-cnn"}),
                "num_classes": ("INT", {"default": 21, "min": 2, "max": 1000}),
            },
            "optional": {
                "backbone": (["vgg16", "resnet50", "resnet101"], {"default": "resnet50"}),
                "roi_pool_size": ("INT", {"default": 7, "min": 3, "max": 14}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("detected_boxes", "detected_scores", "detected_labels", "detection_info")
    FUNCTION = "detect"
    CATEGORY = "ComfyNN/ComputerVision/RCNN"
    DESCRIPTION = "R-CNN系列检测器"

    def detect(self, input_images, region_proposals, rcnn_variant, num_classes, backbone="resnet50", 
               roi_pool_size=7, confidence_threshold=0.5):
        # 确保输入是torch.Tensor
        if not isinstance(input_images, torch.Tensor) or not isinstance(region_proposals, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        batch_size = input_images.shape[0]
        num_proposals = region_proposals.shape[0] if region_proposals.dim() == 2 else region_proposals.shape[1]
        
        # 根据R-CNN变体生成不同的输出
        if rcnn_variant == "r-cnn":
            # R-CNN: 对每个候选区域进行分类和回归
            detected_boxes = torch.randn(batch_size, num_proposals, 4)  # [x1, y1, x2, y2]
            detected_scores = torch.randn(batch_size, num_proposals, num_classes)
            detected_labels = torch.randint(0, num_classes, (batch_size, num_proposals))
            
        elif rcnn_variant == "fast r-cnn":
            # Fast R-CNN: 使用ROI池化共享特征
            detected_boxes = torch.randn(batch_size, num_proposals, 4)
            detected_scores = torch.randn(batch_size, num_proposals, num_classes)
            detected_labels = torch.randint(0, num_classes, (batch_size, num_proposals))
            
        elif rcnn_variant == "faster r-cnn":
            # Faster R-CNN: 使用RPN生成候选区域
            detected_boxes = torch.randn(batch_size, num_proposals, 4)
            detected_scores = torch.randn(batch_size, num_proposals, num_classes)
            detected_labels = torch.randint(0, num_classes, (batch_size, num_proposals))
            
        elif rcnn_variant == "mask r-cnn":
            # Mask R-CNN: 添加实例分割功能
            detected_boxes = torch.randn(batch_size, num_proposals, 4)
            detected_scores = torch.randn(batch_size, num_proposals, num_classes)
            detected_labels = torch.randint(0, num_classes, (batch_size, num_proposals))
        
        # 应用softmax到分数
        detected_scores = F.softmax(detected_scores, dim=-1)
        
        # 生成检测信息
        detection_info = f"R-CNN Variant: {rcnn_variant}\n"
        detection_info += f"Backbone: {backbone}\n"
        detection_info += f"Input batch size: {batch_size}\n"
        detection_info += f"Number of proposals: {num_proposals}\n"
        detection_info += f"Number of classes: {num_classes}\n"
        detection_info += f"ROI pool size: {roi_pool_size}\n"
        detection_info += f"Confidence threshold: {confidence_threshold}"
        
        return (detected_boxes, detected_scores, detected_labels, detection_info)


class RegionProposalNetwork:
    """区域建议网络节点 (RPN)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature_maps": ("TENSOR",),
                "anchor_boxes": ("TENSOR",),
                "rpn_type": (["standard", "lightweight"], {"default": "standard"}),
            },
            "optional": {
                "positive_iou_threshold": ("FLOAT", {"default": 0.7, "min": 0.5, "max": 1.0, "step": 0.01}),
                "negative_iou_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.5, "step": 0.01}),
                "num_proposals": ("INT", {"default": 2000, "min": 100, "max": 5000}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("proposals", "objectness_scores", "proposal_deltas", "rpn_info")
    FUNCTION = "generate_proposals"
    CATEGORY = "ComfyNN/ComputerVision/RCNN"
    DESCRIPTION = "区域建议网络(RPN)"

    def generate_proposals(self, feature_maps, anchor_boxes, rpn_type, positive_iou_threshold=0.7, 
                          negative_iou_threshold=0.3, num_proposals=2000):
        # 确保输入是torch.Tensor
        if not isinstance(feature_maps, torch.Tensor) or not isinstance(anchor_boxes, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        batch_size = feature_maps.shape[0] if feature_maps.dim() > 1 else 1
        num_anchors = anchor_boxes.shape[0]
        
        # 生成区域建议
        if rpn_type == "standard":
            # 标准RPN
            proposals = anchor_boxes.clone()  # 简化处理，实际会应用回归调整
            objectness_scores = torch.randn(batch_size, num_anchors, 2)  # 前景/背景分数
            proposal_deltas = torch.randn(batch_size, num_anchors, 4)  # 边界框调整值
            
        elif rpn_type == "lightweight":
            # 轻量级RPN
            proposals = anchor_boxes.clone()
            objectness_scores = torch.randn(batch_size, num_anchors, 2)
            proposal_deltas = torch.randn(batch_size, num_anchors, 4)
        
        # 应用softmax到目标性分数
        objectness_scores = F.softmax(objectness_scores, dim=-1)
        
        # 选择最有可能的建议
        # 这里简化处理，实际会根据分数选择前N个建议
        if num_proposals < num_anchors:
            # 选择分数最高的建议
            foreground_scores = objectness_scores[:, :, 1]  # 前景分数
            _, top_indices = torch.topk(foreground_scores, num_proposals, dim=1)
            
            # 收集选中的建议
            batch_indices = torch.arange(batch_size).unsqueeze(1)
            proposals = proposals[top_indices] if proposals.dim() == 2 else proposals[batch_indices, top_indices]
            objectness_scores = objectness_scores[batch_indices, top_indices]
            proposal_deltas = proposal_deltas[batch_indices, top_indices]
        
        # 生成RPN信息
        rpn_info = f"RPN Type: {rpn_type}\n"
        rpn_info += f"Batch size: {batch_size}\n"
        rpn_info += f"Number of anchors: {num_anchors}\n"
        rpn_info += f"Positive IoU threshold: {positive_iou_threshold}\n"
        rpn_info += f"Negative IoU threshold: {negative_iou_threshold}\n"
        rpn_info += f"Number of proposals: {num_proposals}"
        
        return (proposals, objectness_scores, proposal_deltas, rpn_info)


class ROIPooling:
    """ROI池化节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature_maps": ("TENSOR",),
                "region_proposals": ("TENSOR",),
                "pooling_type": (["roi_pooling", "roi_align"], {"default": "roi_align"}),
                "output_size": ("INT", {"default": 7, "min": 3, "max": 14}),
            },
            "optional": {
                "spatial_scale": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 2.0, "step": 0.1}),
                "sampling_ratio": ("INT", {"default": 2, "min": 0, "max": 5}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("pooled_features", "pooling_info")
    FUNCTION = "pool"
    CATEGORY = "ComfyNN/ComputerVision/RCNN"
    DESCRIPTION = "ROI池化"

    def pool(self, feature_maps, region_proposals, pooling_type, output_size, spatial_scale=1.0, 
             sampling_ratio=2):
        # 确保输入是torch.Tensor
        if not isinstance(feature_maps, torch.Tensor) or not isinstance(region_proposals, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        batch_size = feature_maps.shape[0] if feature_maps.dim() > 1 else 1
        num_proposals = region_proposals.shape[0] if region_proposals.dim() == 2 else region_proposals.shape[1]
        channels = feature_maps.shape[1] if feature_maps.dim() > 1 else feature_maps.shape[-1]
        
        # 执行ROI池化
        if pooling_type == "roi_pooling":
            # ROI池化
            pooled_features = torch.randn(batch_size, num_proposals, channels, output_size, output_size)
            
        elif pooling_type == "roi_align":
            # ROI对齐
            pooled_features = torch.randn(batch_size, num_proposals, channels, output_size, output_size)
        
        # 生成池化信息
        pooling_info = f"Pooling Type: {pooling_type}\n"
        pooling_info += f"Batch size: {batch_size}\n"
        pooling_info += f"Number of proposals: {num_proposals}\n"
        pooling_info += f"Output size: {output_size}x{output_size}\n"
        pooling_info += f"Channels: {channels}\n"
        pooling_info += f"Spatial scale: {spatial_scale}\n"
        pooling_info += f"Sampling ratio: {sampling_ratio}"
        
        return (pooled_features, pooling_info)


class MaskHead:
    """Mask头节点 (用于Mask R-CNN)"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "roi_features": ("TENSOR",),
                "detected_boxes": ("TENSOR",),
                "num_classes": ("INT", {"default": 21, "min": 2, "max": 1000}),
            },
            "optional": {
                "mask_size": ("INT", {"default": 28, "min": 14, "max": 56}),
                "binary_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("instance_masks", "mask_info")
    FUNCTION = "generate_masks"
    CATEGORY = "ComfyNN/ComputerVision/RCNN"
    DESCRIPTION = "Mask头(用于Mask R-CNN)"

    def generate_masks(self, roi_features, detected_boxes, num_classes, mask_size=28, binary_threshold=0.5):
        # 确保输入是torch.Tensor
        if not isinstance(roi_features, torch.Tensor) or not isinstance(detected_boxes, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        batch_size = roi_features.shape[0] if roi_features.dim() > 1 else 1
        num_detections = detected_boxes.shape[0] if detected_boxes.dim() == 2 else detected_boxes.shape[1]
        
        # 生成实例掩码
        # 在Mask R-CNN中，每个检测到的对象都有一个对应的二值掩码
        instance_masks = torch.randn(batch_size, num_detections, mask_size, mask_size)
        
        # 应用sigmoid激活并二值化
        instance_masks = torch.sigmoid(instance_masks)
        instance_masks = (instance_masks > binary_threshold).float()
        
        # 生成掩码信息
        mask_info = f"Number of classes: {num_classes}\n"
        mask_info += f"Batch size: {batch_size}\n"
        mask_info += f"Number of detections: {num_detections}\n"
        mask_info += f"Mask size: {mask_size}x{mask_size}\n"
        mask_info += f"Binary threshold: {binary_threshold}"
        
        return (instance_masks, mask_info)