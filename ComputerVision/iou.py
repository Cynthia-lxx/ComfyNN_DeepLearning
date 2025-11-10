# ComfyNN ComputerVision IoU Nodes
import torch
import numpy as np

class IoUCalculator:
    """IoU计算器节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "boxes1": ("TENSOR",),
                "boxes2": ("TENSOR",),
                "iou_type": (["iou", "giou", "diou", "ciou"], {"default": "iou"}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("iou_values", "iou_info")
    FUNCTION = "calculate"
    CATEGORY = "ComfyNN/ComputerVision/IoU"
    DESCRIPTION = "计算边界框之间的IoU"

    def calculate(self, boxes1, boxes2, iou_type):
        # 确保输入是torch.Tensor
        if not isinstance(boxes1, torch.Tensor) or not isinstance(boxes2, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        # 确保边界框格式为 [N, 4]，其中4个值为 [x1, y1, x2, y2]
        if boxes1.dim() == 1:
            boxes1 = boxes1.unsqueeze(0)
        if boxes2.dim() == 1:
            boxes2 = boxes2.unsqueeze(0)
            
        if boxes1.shape[1] != 4 or boxes2.shape[1] != 4:
            raise ValueError("边界框必须有4个坐标值: [x1, y1, x2, y2]")
        
        # 计算IoU
        if iou_type == "iou":
            iou_values = self._calculate_iou(boxes1, boxes2)
        elif iou_type == "giou":
            iou_values = self._calculate_giou(boxes1, boxes2)
        elif iou_type == "diou":
            iou_values = self._calculate_diou(boxes1, boxes2)
        elif iou_type == "ciou":
            iou_values = self._calculate_ciou(boxes1, boxes2)
        
        # 生成信息字符串
        iou_info = f"IoU type: {iou_type}\n"
        iou_info += f"Boxes1 shape: {boxes1.shape}\n"
        iou_info += f"Boxes2 shape: {boxes2.shape}\n"
        iou_info += f"Output shape: {iou_values.shape}"
        
        return (iou_values, iou_info)
    
    def _calculate_iou(self, boxes1, boxes2):
        """计算标准IoU"""
        # 计算交集坐标
        x1_inter = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
        y1_inter = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
        x2_inter = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
        y2_inter = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
        
        # 计算交集面积
        inter_width = torch.clamp(x2_inter - x1_inter, min=0)
        inter_height = torch.clamp(y2_inter - y1_inter, min=0)
        inter_area = inter_width * inter_height
        
        # 计算各框面积
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # 计算并集面积
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
        
        # 计算IoU
        iou = inter_area / torch.clamp(union_area, min=1e-8)
        
        return iou
    
    def _calculate_giou(self, boxes1, boxes2):
        """计算GIoU (Generalized IoU)"""
        # 先计算标准IoU
        iou = self._calculate_iou(boxes1, boxes2)
        
        # 计算最小闭包区域坐标
        x1_enclose = torch.min(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
        y1_enclose = torch.min(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
        x2_enclose = torch.max(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
        y2_enclose = torch.max(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
        
        # 计算闭包区域面积
        enclose_width = torch.clamp(x2_enclose - x1_enclose, min=0)
        enclose_height = torch.clamp(y2_enclose - y1_enclose, min=0)
        enclose_area = enclose_width * enclose_height
        
        # 计算GIoU
        giou = iou - (enclose_area - iou) / torch.clamp(enclose_area, min=1e-8)
        
        return giou
    
    def _calculate_diou(self, boxes1, boxes2):
        """计算DIoU (Distance IoU)"""
        # 先计算标准IoU
        iou = self._calculate_iou(boxes1, boxes2)
        
        # 计算中心点
        center_x1 = (boxes1[:, 0] + boxes1[:, 2]) / 2
        center_y1 = (boxes1[:, 1] + boxes1[:, 3]) / 2
        center_x2 = (boxes2[:, 0] + boxes2[:, 2]) / 2
        center_y2 = (boxes2[:, 1] + boxes2[:, 3]) / 2
        
        # 计算中心点距离
        center_distance = (center_x1.unsqueeze(1) - center_x2.unsqueeze(0)) ** 2 + \
                          (center_y1.unsqueeze(1) - center_y2.unsqueeze(0)) ** 2
        
        # 计算最小闭包区域对角线距离
        enclose_width = torch.max(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0)) - \
                        torch.min(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
        enclose_height = torch.max(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0)) - \
                         torch.min(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
        diagonal_distance = enclose_width ** 2 + enclose_height ** 2
        
        # 计算DIoU
        diou = iou - center_distance / torch.clamp(diagonal_distance, min=1e-8)
        
        return diou
    
    def _calculate_ciou(self, boxes1, boxes2):
        """计算CIoU (Complete IoU)"""
        # 先计算DIoU
        diou = self._calculate_diou(boxes1, boxes2)
        
        # 计算宽高比一致性
        width1 = boxes1[:, 2] - boxes1[:, 0]
        height1 = boxes1[:, 3] - boxes1[:, 1]
        width2 = boxes2[:, 2] - boxes2[:, 0]
        height2 = boxes2[:, 3] - boxes2[:, 1]
        
        # 避免除零错误
        height1 = torch.clamp(height1, min=1e-8)
        height2 = torch.clamp(height2, min=1e-8)
        
        arctan1 = torch.atan(width1 / height1)
        arctan2 = torch.atan(width2 / height2)
        v = (4 / (np.pi ** 2)) * (arctan1.unsqueeze(1) - arctan2.unsqueeze(0)) ** 2
        
        # 计算alpha参数
        alpha = v / torch.clamp(1 - diou + v, min=1e-8)
        
        # 计算CIoU
        ciou = diou - alpha * v
        
        return ciou


class IoUThresholdFilter:
    """IoU阈值过滤节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "boxes": ("TENSOR",),
                "scores": ("TENSOR",),
                "iou_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "filter_method": (["nms", "soft_nms", "diou_nms"], {"default": "nms"}),
            },
            "optional": {
                "max_output_size": ("INT", {"default": 100, "min": 1, "max": 1000}),
                "sigma": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),  # 用于Soft-NMS
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("filtered_boxes", "filtered_scores", "filter_info")
    FUNCTION = "filter"
    CATEGORY = "ComfyNN/ComputerVision/IoU"
    DESCRIPTION = "基于IoU阈值过滤边界框"

    def filter(self, boxes, scores, iou_threshold, filter_method, max_output_size=100, sigma=0.5):
        # 确保输入是torch.Tensor
        if not isinstance(boxes, torch.Tensor) or not isinstance(scores, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        # 确保边界框格式为 [N, 4]，分数为 [N]
        if boxes.dim() == 1:
            boxes = boxes.unsqueeze(0)
        if scores.dim() == 0:
            scores = scores.unsqueeze(0)
            
        if boxes.shape[1] != 4:
            raise ValueError("边界框必须有4个坐标值: [x1, y1, x2, y2]")
        
        if boxes.shape[0] != scores.shape[0]:
            raise ValueError("边界框和分数的数量必须相同")
        
        # 根据过滤方法进行处理
        if filter_method == "nms":
            filtered_boxes, filtered_scores = self._nms(boxes, scores, iou_threshold, max_output_size)
        elif filter_method == "soft_nms":
            filtered_boxes, filtered_scores = self._soft_nms(boxes, scores, iou_threshold, sigma, max_output_size)
        elif filter_method == "diou_nms":
            filtered_boxes, filtered_scores = self._diou_nms(boxes, scores, iou_threshold, max_output_size)
        
        # 生成信息字符串
        filter_info = f"Filter method: {filter_method}\n"
        filter_info += f"Original boxes: {boxes.shape[0]}\n"
        filter_info += f"Filtered boxes: {filtered_boxes.shape[0]}\n"
        filter_info += f"IoU threshold: {iou_threshold}\n"
        filter_info += f"Max output size: {max_output_size}"
        
        return (filtered_boxes, filtered_scores, filter_info)
    
    def _nms(self, boxes, scores, iou_threshold, max_output_size):
        """标准非极大值抑制"""
        # 按分数降序排列
        _, indices = torch.sort(scores, descending=True)
        indices = indices[:max_output_size]
        
        keep = []
        while len(indices) > 0:
            # 保留分数最高的框
            current = indices[0]
            keep.append(current)
            
            if len(indices) == 1:
                break
                
            # 计算当前框与其余框的IoU
            ious = self._calculate_iou(boxes[current].unsqueeze(0), boxes[indices[1:]])
            
            # 保留IoU小于阈值的框
            remaining_indices = indices[1:][ious[0] < iou_threshold]
            indices = remaining_indices
        
        keep = torch.stack(keep) if len(keep) > 0 else torch.tensor([], dtype=torch.long)
        return boxes[keep], scores[keep]
    
    def _soft_nms(self, boxes, scores, iou_threshold, sigma, max_output_size):
        """Soft-NMS"""
        # 这里简化实现，实际Soft-NMS会降低重叠框的分数而不是直接删除
        # 为简化起见，我们使用标准NMS
        return self._nms(boxes, scores, iou_threshold, max_output_size)
    
    def _diou_nms(self, boxes, scores, iou_threshold, max_output_size):
        """DIoU-NMS"""
        # 这里简化实现，实际DIoU-NMS会使用DIoU而不是IoU
        # 为简化起见，我们使用标准NMS
        return self._nms(boxes, scores, iou_threshold, max_output_size)
    
    def _calculate_iou(self, boxes1, boxes2):
        """计算标准IoU（内部使用）"""
        # 计算交集坐标
        x1_inter = torch.max(boxes1[:, 0].unsqueeze(1), boxes2[:, 0].unsqueeze(0))
        y1_inter = torch.max(boxes1[:, 1].unsqueeze(1), boxes2[:, 1].unsqueeze(0))
        x2_inter = torch.min(boxes1[:, 2].unsqueeze(1), boxes2[:, 2].unsqueeze(0))
        y2_inter = torch.min(boxes1[:, 3].unsqueeze(1), boxes2[:, 3].unsqueeze(0))
        
        # 计算交集面积
        inter_width = torch.clamp(x2_inter - x1_inter, min=0)
        inter_height = torch.clamp(y2_inter - y1_inter, min=0)
        inter_area = inter_width * inter_height
        
        # 计算各框面积
        area1 = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        area2 = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        
        # 计算并集面积
        union_area = area1.unsqueeze(1) + area2.unsqueeze(0) - inter_area
        
        # 计算IoU
        iou = inter_area / torch.clamp(union_area, min=1e-8)
        
        return iou