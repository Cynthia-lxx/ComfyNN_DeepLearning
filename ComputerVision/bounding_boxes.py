# ComfyNN ComputerVision Bounding Box Nodes
import torch
import numpy as np

class BoundingBoxNode:
    """边界框生成节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "bbox_format": (["xyxy", "xywh", "cxcywh"], {"default": "xyxy"}),
            },
            "optional": {
                "normalize_coords": ("BOOLEAN", {"default": False}),
                "add_padding": ("BOOLEAN", {"default": False}),
                "padding_ratio": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("bounding_boxes", "bbox_info")
    FUNCTION = "generate"
    CATEGORY = "ComfyNN/ComputerVision/BoundingBox"
    DESCRIPTION = "生成边界框"

    def generate(self, image_batch, bbox_format, normalize_coords=False, add_padding=False, padding_ratio=0.1):
        # 获取图像批次信息
        batch_size, height, width = image_batch.shape[:3]
        
        # 生成示例边界框（在实际应用中，这些应该来自目标检测算法）
        # 这里我们生成一些随机边界框作为示例
        bboxes = []
        for i in range(batch_size):
            # 生成一个随机边界框
            x1 = np.random.randint(0, width // 2)
            y1 = np.random.randint(0, height // 2)
            x2 = np.random.randint(width // 2, width)
            y2 = np.random.randint(height // 2, height)
            
            if bbox_format == "xyxy":
                bbox = [x1, y1, x2, y2]
            elif bbox_format == "xywh":
                bbox = [x1, y1, x2 - x1, y2 - y1]
            elif bbox_format == "cxcywh":
                cx = (x1 + x2) / 2
                cy = (y1 + y2) / 2
                w = x2 - x1
                h = y2 - y1
                bbox = [cx, cy, w, h]
            
            bboxes.append(bbox)
        
        # 转换为tensor
        bbox_tensor = torch.tensor(bboxes, dtype=torch.float32)
        
        # 如果需要归一化坐标
        if normalize_coords:
            if bbox_format == "xyxy":
                bbox_tensor[:, [0, 2]] /= width
                bbox_tensor[:, [1, 3]] /= height
            elif bbox_format == "xywh":
                bbox_tensor[:, [0, 2]] /= width
                bbox_tensor[:, [1, 3]] /= height
            elif bbox_format == "cxcywh":
                bbox_tensor[:, [0, 2]] /= width
                bbox_tensor[:, [1, 3]] /= height
        
        # 生成边界框信息字符串
        bbox_info = f"Batch size: {batch_size}\n"
        bbox_info += f"Image size: {width}x{height}\n"
        bbox_info += f"Bounding box format: {bbox_format}\n"
        bbox_info += f"Normalized coordinates: {normalize_coords}\n"
        bbox_info += f"Sample bbox: {bboxes[0] if bboxes else 'None'}"
        
        return (bbox_tensor, bbox_info)


class BoundingBoxMatchingNode:
    """边界框处理节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "bounding_boxes": ("TENSOR",),
                "operation": (["scale", "shift", "clip", "filter"], {"default": "scale"}),
                "image_width": ("INT", {"default": 224, "min": 32, "max": 2048}),
                "image_height": ("INT", {"default": 224, "min": 32, "max": 2048}),
            },
            "optional": {
                "scale_factor": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 3.0, "step": 0.1}),
                "shift_x": ("INT", {"default": 0, "min": -100, "max": 100}),
                "shift_y": ("INT", {"default": 0, "min": -100, "max": 100}),
                "min_area": ("INT", {"default": 100, "min": 1, "max": 10000}),
                "max_area": ("INT", {"default": 10000, "min": 100, "max": 100000}),
            }
        }

    RETURN_TYPES = ("TENSOR", "STRING")
    RETURN_NAMES = ("processed_bboxes", "processing_info")
    FUNCTION = "process"
    CATEGORY = "ComfyNN/ComputerVision/BoundingBox"
    DESCRIPTION = "处理边界框"

    def process(self, bounding_boxes, operation, image_width, image_height, scale_factor=1.0,
                shift_x=0, shift_y=0, min_area=100, max_area=10000):
        # 确保输入是torch.Tensor
        if not isinstance(bounding_boxes, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        # 复制边界框以避免修改原始数据
        processed_bboxes = bounding_boxes.clone()
        
        # 获取边界框格式（假设是xyxy格式）
        if processed_bboxes.shape[1] < 4:
            raise ValueError("边界框必须至少包含4个坐标值")
        
        info = f"Operation: {operation}\n"
        
        if operation == "scale":
            # 缩放边界框
            processed_bboxes[:, [0, 2]] *= scale_factor  # x coordinates
            processed_bboxes[:, [1, 3]] *= scale_factor  # y coordinates
            info += f"Scaled by factor: {scale_factor}\n"
            
        elif operation == "shift":
            # 平移边界框
            processed_bboxes[:, [0, 2]] += shift_x  # x coordinates
            processed_bboxes[:, [1, 3]] += shift_y  # y coordinates
            info += f"Shifted by ({shift_x}, {shift_y})\n"
            
        elif operation == "clip":
            # 裁剪边界框到图像边界
            processed_bboxes[:, [0, 2]] = torch.clamp(processed_bboxes[:, [0, 2]], 0, image_width)
            processed_bboxes[:, [1, 3]] = torch.clamp(processed_bboxes[:, [1, 3]], 0, image_height)
            info += f"Clipped to image size: {image_width}x{image_height}\n"
            
        elif operation == "filter":
            # 根据面积过滤边界框
            widths = processed_bboxes[:, 2] - processed_bboxes[:, 0]
            heights = processed_bboxes[:, 3] - processed_bboxes[:, 1]
            areas = widths * heights
            
            # 保留面积在指定范围内的边界框
            valid_indices = (areas >= min_area) & (areas <= max_area)
            processed_bboxes = processed_bboxes[valid_indices]
            info += f"Filtered by area [{min_area}, {max_area}]\n"
            info += f"Remaining boxes: {processed_bboxes.shape[0]}/{bounding_boxes.shape[0]}\n"
        
        info += f"Processed boxes shape: {processed_bboxes.shape}"
        
        return (processed_bboxes, info)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "BoundingBoxNode": BoundingBoxNode,
    "BoundingBoxMatchingNode": BoundingBoxMatchingNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "BoundingBoxNode": "Bounding Box 🐱",
    "BoundingBoxMatchingNode": "Bounding Box Matching 🐱",
}