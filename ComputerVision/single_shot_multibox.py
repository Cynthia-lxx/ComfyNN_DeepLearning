# ComfyNN ComputerVision Single Shot Multibox Detector Nodes
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class SSDModel:
    """单发多框检测器节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_images": ("IMAGE",),
                "num_classes": ("INT", {"default": 21, "min": 2, "max": 1000}),
                "backbone": (["vgg16", "resnet50", "mobilenet"], {"default": "vgg16"}),
            },
            "optional": {
                "feature_map_sizes": ("STRING", {"default": "38,19,10,5,3,1", "multiline": False}),
                "anchor_boxes_per_location": ("INT", {"default": 4, "min": 1, "max": 10}),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("predicted_boxes", "predicted_scores", "detection_info")
    FUNCTION = "detect"
    CATEGORY = "ComfyNN/ComputerVision/SSD"
    DESCRIPTION = "单发多框检测器"

    def detect(self, input_images, num_classes, backbone, feature_map_sizes="38,19,10,5,3,1", 
               anchor_boxes_per_location=4, confidence_threshold=0.5):
        # 确保输入是torch.Tensor
        if not isinstance(input_images, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        batch_size = input_images.shape[0]
        
        # 解析特征图尺寸
        feat_sizes = [int(x.strip()) for x in feature_map_sizes.split(",") if x.strip()]
        
        # 计算总的锚框数量
        total_anchors = sum(size * size * anchor_boxes_per_location for size in feat_sizes)
        
        # 生成预测框（这里简化处理，实际SSD会输出真实的边界框预测）
        predicted_boxes = torch.randn(batch_size, total_anchors, 4)  # [x1, y1, x2, y2]
        
        # 生成预测分数（每个类别一个分数）
        predicted_scores = torch.randn(batch_size, total_anchors, num_classes)
        predicted_scores = F.softmax(predicted_scores, dim=-1)  # 应用softmax
        
        # 生成检测信息
        detection_info = f"SSD Model: {backbone}\n"
        detection_info += f"Input batch size: {batch_size}\n"
        detection_info += f"Number of classes: {num_classes}\n"
        detection_info += f"Feature map sizes: {feature_map_sizes}\n"
        detection_info += f"Anchor boxes per location: {anchor_boxes_per_location}\n"
        detection_info += f"Total anchors: {total_anchors}\n"
        detection_info += f"Confidence threshold: {confidence_threshold}"
        
        return (predicted_boxes, predicted_scores, detection_info)


class SSDAnchorGenerator:
    """SSD锚框生成器节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_size": ("INT", {"default": 300, "min": 32, "max": 1024}),
                "feature_map_sizes": ("STRING", {"default": "38,19,10,5,3,1", "multiline": False}),
            },
            "optional": {
                "min_sizes": ("STRING", {"default": "30,60,111,162,213,264", "multiline": False}),
                "max_sizes": ("STRING", {"default": "60,111,162,213,264,315", "multiline": False}),
                "aspect_ratios": ("STRING", {"default": "2,3,2,3,2,3", "multiline": False}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("ssd_anchors", "anchor_info")
    FUNCTION = "generate"
    CATEGORY = "ComfyNN/ComputerVision/SSD"
    DESCRIPTION = "SSD锚框生成器"

    def generate(self, image_size, feature_map_sizes, min_sizes="30,60,111,162,213,264", 
                 max_sizes="60,111,162,213,264,315", aspect_ratios="2,3,2,3,2,3"):
        # 解析输入参数
        feat_sizes = [int(x.strip()) for x in feature_map_sizes.split(",") if x.strip()]
        min_sz = [float(x.strip()) for x in min_sizes.split(",") if x.strip()]
        max_sz = [float(x.strip()) for x in max_sizes.split(",") if x.strip()]
        ratios = [float(x.strip()) for x in aspect_ratios.split(",") if x.strip()]
        
        # 确保参数长度一致
        if not (len(feat_sizes) == len(min_sz) == len(max_sz) == len(ratios)):
            raise ValueError("特征图尺寸、最小尺寸、最大尺寸和宽高比的数量必须一致")
        
        anchors = []
        
        # 为每个特征图生成锚框
        for i, (feat_size, min_size, max_size, ratio) in enumerate(zip(feat_sizes, min_sz, max_sz, ratios)):
            # 计算步长
            step = image_size // feat_size
            
            # 为特征图上的每个位置生成锚框
            for y in range(feat_size):
                for x in range(feat_size):
                    # 计算中心点坐标
                    center_x = (x + 0.5) * step
                    center_y = (y + 0.5) * step
                    
                    # 生成不同尺寸和比例的锚框
                    # 1. 基本锚框 (宽高比为1:1)
                    w1, h1 = min_size, min_size
                    anchors.append([center_x - w1/2, center_y - h1/2, center_x + w1/2, center_y + h1/2])
                    
                    w2, h2 = np.sqrt(min_size * max_size), np.sqrt(min_size * max_size)
                    anchors.append([center_x - w2/2, center_y - h2/2, center_x + w2/2, center_y + h2/2])
                    
                    # 2. 不同宽高比的锚框
                    w3, h3 = min_size * np.sqrt(ratio), min_size / np.sqrt(ratio)
                    anchors.append([center_x - w3/2, center_y - h3/2, center_x + w3/2, center_y + h3/2])
                    
                    w4, h4 = min_size / np.sqrt(ratio), min_size * np.sqrt(ratio)
                    anchors.append([center_x - w4/2, center_y - h4/2, center_x + w4/2, center_y + h4/2])
        
        # 转换为tensor
        anchor_tensor = torch.tensor(anchors, dtype=torch.float32)
        
        # 生成锚框信息
        anchor_info = f"Image size: {image_size}\n"
        anchor_info += f"Feature map sizes: {feature_map_sizes}\n"
        anchor_info += f"Min sizes: {min_sizes}\n"
        anchor_info += f"Max sizes: {max_sizes}\n"
        anchor_info += f"Aspect ratios: {aspect_ratios}\n"
        anchor_info += f"Total anchors: {len(anchors)}"
        
        return (anchor_tensor, anchor_info)


class SSDDetectionPostProcessor:
    """SSD检测后处理节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "predicted_boxes": ("TENSOR",),
                "predicted_scores": ("TENSOR",),
                "confidence_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "nms_threshold": ("FLOAT", {"default": 0.45, "min": 0.0, "max": 1.0, "step": 0.01}),
            },
            "optional": {
                "top_k": ("INT", {"default": 200, "min": 1, "max": 1000}),
                "keep_top_k": ("INT", {"default": 100, "min": 1, "max": 500}),
                "background_label_id": ("INT", {"default": 0, "min": 0, "max": 100}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "TENSOR", "STRING")
    RETURN_NAMES = ("final_boxes", "final_scores", "final_labels", "postprocess_info")
    FUNCTION = "postprocess"
    CATEGORY = "ComfyNN/ComputerVision/SSD"
    DESCRIPTION = "SSD检测后处理"

    def postprocess(self, predicted_boxes, predicted_scores, confidence_threshold, nms_threshold, 
                    top_k=200, keep_top_k=100, background_label_id=0):
        # 确保输入是torch.Tensor
        if not isinstance(predicted_boxes, torch.Tensor) or not isinstance(predicted_scores, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        batch_size = predicted_boxes.shape[0]
        num_anchors = predicted_boxes.shape[1]
        
        # 获取前景类别的分数（排除背景类）
        if predicted_scores.shape[-1] > background_label_id:
            # 选择除背景外分数最高的类别
            fg_scores = torch.cat([
                predicted_scores[:, :, :background_label_id],
                predicted_scores[:, :, background_label_id + 1:]
            ], dim=-1) if background_label_id > 0 else predicted_scores[:, :, 1:]
            
            # 获取最高分数和对应的类别索引
            max_scores, max_indices = torch.max(fg_scores, dim=-1)
            max_indices += (1 if background_label_id == 0 else 0)  # 调整类别索引
        else:
            max_scores, max_indices = torch.max(predicted_scores, dim=-1)
        
        # 应用置信度阈值
        score_mask = max_scores > confidence_threshold
        filtered_scores = []
        filtered_boxes = []
        filtered_labels = []
        
        for i in range(batch_size):
            # 获取该图像中满足阈值的预测
            valid_indices = torch.nonzero(score_mask[i], as_tuple=False).squeeze(-1)
            
            if len(valid_indices) > 0:
                # 获取对应的框、分数和标签
                valid_boxes = predicted_boxes[i][valid_indices]
                valid_scores = max_scores[i][valid_indices]
                valid_labels = max_indices[i][valid_indices]
                
                # 如果预测数量超过top_k，只保留分数最高的top_k个
                if len(valid_scores) > top_k:
                    _, top_indices = torch.topk(valid_scores, top_k)
                    valid_boxes = valid_boxes[top_indices]
                    valid_scores = valid_scores[top_indices]
                    valid_labels = valid_labels[top_indices]
                
                # 应用NMS
                keep_indices = self._nms(valid_boxes, valid_scores, nms_threshold)
                
                # 保留NMS后的预测
                final_boxes = valid_boxes[keep_indices]
                final_scores = valid_scores[keep_indices]
                final_labels = valid_labels[keep_indices]
                
                # 如果预测数量超过keep_top_k，只保留分数最高的keep_top_k个
                if len(final_scores) > keep_top_k:
                    _, top_indices = torch.topk(final_scores, keep_top_k)
                    final_boxes = final_boxes[top_indices]
                    final_scores = final_scores[top_indices]
                    final_labels = final_labels[top_indices]
            else:
                # 没有满足阈值的预测，返回空的结果
                final_boxes = torch.empty((0, 4))
                final_scores = torch.empty(0)
                final_labels = torch.empty(0, dtype=torch.long)
            
            filtered_boxes.append(final_boxes)
            filtered_scores.append(final_scores)
            filtered_labels.append(final_labels)
        
        # 为了保持输出格式一致，我们需要填充或截断到相同长度
        # 这里简化处理，只返回第一个批次的结果
        final_boxes = filtered_boxes[0] if len(filtered_boxes) > 0 else torch.empty((0, 4))
        final_scores = filtered_scores[0] if len(filtered_scores) > 0 else torch.empty(0)
        final_labels = filtered_labels[0] if len(filtered_labels) > 0 else torch.empty(0, dtype=torch.long)
        
        # 生成后处理信息
        postprocess_info = f"Batch size: {batch_size}\n"
        postprocess_info += f"Number of anchors: {num_anchors}\n"
        postprocess_info += f"Confidence threshold: {confidence_threshold}\n"
        postprocess_info += f"NMS threshold: {nms_threshold}\n"
        postprocess_info += f"Top K: {top_k}\n"
        postprocess_info += f"Keep Top K: {keep_top_k}\n"
        postprocess_info += f"Background label ID: {background_label_id}\n"
        postprocess_info += f"Final detections: {final_boxes.shape[0]}"
        
        return (final_boxes, final_scores, final_labels, postprocess_info)
    
    def _nms(self, boxes, scores, threshold):
        """非极大值抑制"""
        if boxes.shape[0] == 0:
            return torch.empty(0, dtype=torch.long)
        
        # 按分数降序排列
        _, indices = torch.sort(scores, descending=True)
        
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
            remaining_indices = indices[1:][ious[0] < threshold]
            indices = remaining_indices
        
        return torch.stack(keep) if len(keep) > 0 else torch.tensor([], dtype=torch.long)
    
    def _calculate_iou(self, boxes1, boxes2):
        """计算IoU"""
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