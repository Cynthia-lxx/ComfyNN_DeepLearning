# ComfyNN ComputerVision Anchor Box Nodes
import torch
import numpy as np

class AnchorBoxNode:
    """锚框生成节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_width": ("INT", {"default": 224, "min": 32, "max": 2048}),
                "image_height": ("INT", {"default": 224, "min": 32, "max": 2048}),
                "feature_map_sizes": ("STRING", {"default": "7,7", "multiline": False}),
            },
            "optional": {
                "anchor_scales": ("STRING", {"default": "32,64,128,256,512", "multiline": False}),
                "aspect_ratios": ("STRING", {"default": "0.5,1.0,2.0", "multiline": False}),
                "stride": ("INT", {"default": 32, "min": 1, "max": 128}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("anchor_boxes", "anchor_info")
    FUNCTION = "generate"
    CATEGORY = "ComfyNN/ComputerVision/AnchorBoxes"
    DESCRIPTION = "生成锚框"

    def generate(self, image_width, image_height, feature_map_sizes, anchor_scales="32,64,128,256,512", 
                 aspect_ratios="0.5,1.0,2.0", stride=32):
        # 解析输入参数
        feat_sizes = [int(x.strip()) for x in feature_map_sizes.split(",") if x.strip()]
        scales = [float(x.strip()) for x in anchor_scales.split(",") if x.strip()]
        ratios = [float(x.strip()) for x in aspect_ratios.split(",") if x.strip()]
        
        # 生成锚框
        anchor_boxes = []
        
        # 对每个特征图尺寸生成锚框
        for feat_h in feat_sizes:
            for feat_w in feat_sizes:
                # 计算特征图上每个点对应原图的中心坐标
                for i in range(feat_h):
                    center_y = (i + 0.5) * stride
                    for j in range(feat_w):
                        center_x = (j + 0.5) * stride
                        
                        # 对每个尺度和比例生成锚框
                        for scale in scales:
                            for ratio in ratios:
                                # 计算锚框的宽度和高度
                                w = scale * np.sqrt(ratio)
                                h = scale / np.sqrt(ratio)
                                
                                # 计算锚框的坐标 (x1, y1, x2, y2)
                                x1 = center_x - w / 2
                                y1 = center_y - h / 2
                                x2 = center_x + w / 2
                                y2 = center_y + h / 2
                                
                                # 确保锚框在图像范围内
                                x1 = max(0, x1)
                                y1 = max(0, y1)
                                x2 = min(image_width, x2)
                                y2 = min(image_height, y2)
                                
                                anchor_boxes.append([x1, y1, x2, y2])
        
        # 转换为tensor
        anchor_tensor = torch.tensor(anchor_boxes, dtype=torch.float32)
        
        # 生成信息字符串
        anchor_info = f"Image size: {image_width}x{image_height}\n"
        anchor_info += f"Feature map sizes: {feature_map_sizes}\n"
        anchor_info += f"Anchor scales: {anchor_scales}\n"
        anchor_info += f"Aspect ratios: {aspect_ratios}\n"
        anchor_info += f"Stride: {stride}\n"
        anchor_info += f"Total anchors: {len(anchor_boxes)}"
        
        return (anchor_tensor, anchor_info)


class AnchorBoxMatcher:
    """锚框匹配节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "anchor_boxes": ("TENSOR",),
                "ground_truth_boxes": ("TENSOR",),
                "matching_strategy": (["iou", "center"], {"default": "iou"}),
            },
            "optional": {
                "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0.1, "max": 1.0, "step": 0.05}),
                "positive_threshold": ("FLOAT", {"default": 0.7, "min": 0.5, "max": 1.0, "step": 0.05}),
                "negative_threshold": ("FLOAT", {"default": 0.3, "min": 0.0, "max": 0.5, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("matched_anchors", "match_labels", "matching_info")
    FUNCTION = "match"
    CATEGORY = "ComfyNN/ComputerVision/AnchorBoxes"
    DESCRIPTION = "匹配锚框与真实框"

    def match(self, anchor_boxes, ground_truth_boxes, matching_strategy, iou_threshold=0.5,
              positive_threshold=0.7, negative_threshold=0.3):
        # 确保输入是torch.Tensor
        if not isinstance(anchor_boxes, torch.Tensor) or not isinstance(ground_truth_boxes, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        num_anchors = anchor_boxes.shape[0]
        num_gt_boxes = ground_truth_boxes.shape[0]
        
        # 初始化匹配标签 (-1: 忽略, 0: 负样本, 1: 正样本)
        match_labels = torch.full((num_anchors,), -1, dtype=torch.long)
        
        if matching_strategy == "iou":
            # 基于IoU的匹配
            # 计算锚框与真实框之间的IoU
            iou_matrix = self._calculate_iou_matrix(anchor_boxes, ground_truth_boxes)
            
            # 对每个锚框，找到IoU最大的真实框
            max_iou_per_anchor, max_gt_indices = torch.max(iou_matrix, dim=1)
            
            # 根据阈值分配正负样本标签
            match_labels[max_iou_per_anchor >= positive_threshold] = 1  # 正样本
            match_labels[max_iou_per_anchor < negative_threshold] = 0   # 负样本
            
            # 对每个真实框，找到IoU最大的锚框（确保每个真实框至少有一个匹配的锚框）
            max_iou_per_gt, max_anchor_indices = torch.max(iou_matrix, dim=0)
            for i, anchor_idx in enumerate(max_anchor_indices):
                if max_iou_per_gt[i] > 0:  # 确保IoU大于0
                    match_labels[anchor_idx] = 1
            
            matched_anchors = anchor_boxes
            
        elif matching_strategy == "center":
            # 基于中心点的匹配
            # 这里简化处理，实际实现会更复杂
            matched_anchors = anchor_boxes[:min(num_anchors, num_gt_boxes)]
            match_labels = torch.ones(matched_anchors.shape[0], dtype=torch.long)
        
        # 生成匹配信息
        positive_count = torch.sum(match_labels == 1).item()
        negative_count = torch.sum(match_labels == 0).item()
        ignore_count = torch.sum(match_labels == -1).item()
        
        matching_info = f"Matching strategy: {matching_strategy}\n"
        matching_info += f"Total anchors: {num_anchors}\n"
        matching_info += f"Ground truth boxes: {num_gt_boxes}\n"
        matching_info += f"Positive samples: {positive_count}\n"
        matching_info += f"Negative samples: {negative_count}\n"
        matching_info += f"Ignored samples: {ignore_count}\n"
        
        if matching_strategy == "iou":
            matching_info += f"IoU threshold: {iou_threshold}\n"
            matching_info += f"Positive threshold: {positive_threshold}\n"
            matching_info += f"Negative threshold: {negative_threshold}"
        
        return (matched_anchors, match_labels, matching_info)
    
    def _calculate_iou_matrix(self, anchors, gt_boxes):
        """计算锚框与真实框之间的IoU矩阵"""
        # 这里是IoU计算的简化版本
        num_anchors = anchors.shape[0]
        num_gt = gt_boxes.shape[0]
        
        # 初始化IoU矩阵
        iou_matrix = torch.zeros((num_anchors, num_gt))
        
        # 计算每对锚框和真实框的IoU
        for i in range(num_anchors):
            for j in range(num_gt):
                # 计算交集
                x1_inter = torch.max(anchors[i, 0], gt_boxes[j, 0])
                y1_inter = torch.max(anchors[i, 1], gt_boxes[j, 1])
                x2_inter = torch.min(anchors[i, 2], gt_boxes[j, 2])
                y2_inter = torch.min(anchors[i, 3], gt_boxes[j, 3])
                
                # 检查是否有交集
                if x2_inter > x1_inter and y2_inter > y1_inter:
                    inter_area = (x2_inter - x1_inter) * (y2_inter - y1_inter)
                    
                    # 计算并集
                    anchor_area = (anchors[i, 2] - anchors[i, 0]) * (anchors[i, 3] - anchors[i, 1])
                    gt_area = (gt_boxes[j, 2] - gt_boxes[j, 0]) * (gt_boxes[j, 3] - gt_boxes[j, 1])
                    union_area = anchor_area + gt_area - inter_area
                    
                    # 计算IoU
                    if union_area > 0:
                        iou_matrix[i, j] = inter_area / union_area
        
        return iou_matrix

# Node mappings
NODE_CLASS_MAPPINGS = {
    "AnchorBoxNode": AnchorBoxNode,
    "AnchorBoxMatcher": AnchorBoxMatcher,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AnchorBoxNode": "Anchor Box 🐱",
    "AnchorBoxMatcher": "Anchor Box Matcher 🐱",
}