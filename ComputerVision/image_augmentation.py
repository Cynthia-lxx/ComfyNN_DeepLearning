# ComfyNN ComputerVision Image Augmentation Nodes
import torch
import torch.nn.functional as F
import numpy as np
import random
import math
from PIL import Image, ImageEnhance, ImageOps
from torchvision import transforms
from ..DLBasic.nodes import TensorDataType

class ImageAugmentation:
    """图像增广节点"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image_batch": ("IMAGE",),
                "augmentation_type": ([
                    "horizontal_flip", 
                    "vertical_flip", 
                    "random_rotation", 
                    "color_jitter", 
                    "random_crop",
                    "gaussian_noise",
                    "brightness",
                    "contrast",
                    "saturation",
                    "hue"
                ], {"default": "horizontal_flip"}),
            },
            "optional": {
                "probability": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01}),
                "rotation_range": ("INT", {"default": 30, "min": 0, "max": 180}),
                "crop_size": ("INT", {"default": 224, "min": 32, "max": 512}),
                "noise_intensity": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
                "brightness_factor": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "contrast_factor": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "saturation_factor": ("FLOAT", {"default": 0.2, "min": 0.0, "max": 1.0, "step": 0.01}),
                "hue_factor": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 0.5, "step": 0.01}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("augmented_images",)
    FUNCTION = "augment"
    CATEGORY = "ComfyNN/ComputerVision/ImageAugmentation"
    DESCRIPTION = "对图像批次进行增广处理"

    def augment(self, image_batch, augmentation_type, probability=0.5, rotation_range=30, 
                crop_size=224, noise_intensity=0.1, brightness_factor=0.2, contrast_factor=0.2,
                saturation_factor=0.2, hue_factor=0.1):
        # 确保输入是torch.Tensor
        if not isinstance(image_batch, torch.Tensor):
            raise TypeError("输入必须是torch.Tensor类型")
        
        # 获取批次大小和图像尺寸
        batch_size = image_batch.shape[0]
        height, width = image_batch.shape[1:3]
        
        # 创建输出张量
        augmented_images = image_batch.clone()
        
        # 对批次中的每张图像进行增广
        for i in range(batch_size):
            # 根据概率决定是否进行增广
            if random.random() > probability:
                continue
                
            # 获取单张图像
            image = augmented_images[i]  # [H, W, C]
            
            # 转换为PIL图像进行处理
            # 注意：ComfyUI的图像格式是 [0, 1] 范围的浮点数
            pil_image = Image.fromarray((image.cpu().numpy() * 255).astype(np.uint8))
            
            # 根据增广类型进行处理
            if augmentation_type == "horizontal_flip":
                if random.random() > 0.5:
                    pil_image = ImageOps.mirror(pil_image)
                    
            elif augmentation_type == "vertical_flip":
                if random.random() > 0.5:
                    pil_image = ImageOps.flip(pil_image)
                    
            elif augmentation_type == "random_rotation":
                angle = random.uniform(-rotation_range, rotation_range)
                pil_image = pil_image.rotate(angle, expand=False, fillcolor=(0, 0, 0))
                
            elif augmentation_type == "color_jitter":
                # 色彩抖动
                brightness = random.uniform(max(0, 1 - brightness_factor), 1 + brightness_factor)
                contrast = random.uniform(max(0, 1 - contrast_factor), 1 + contrast_factor)
                saturation = random.uniform(max(0, 1 - saturation_factor), 1 + saturation_factor)
                hue = random.uniform(-hue_factor, hue_factor)
                
                # 应用色彩变换
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(brightness)
                
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast)
                
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(saturation)
                
                # Hue调整需要特殊处理
                if abs(hue) > 1e-6:
                    hsv_image = pil_image.convert('HSV')
                    hsv_array = np.array(hsv_image)
                    hsv_array[:, :, 0] = (hsv_array[:, :, 0].astype(np.float32) + hue * 255) % 255
                    hsv_image = Image.fromarray(hsv_array, 'HSV')
                    pil_image = hsv_image.convert('RGB')
                    
            elif augmentation_type == "random_crop":
                # 随机裁剪
                if width > crop_size and height > crop_size:
                    max_x_offset = width - crop_size
                    max_y_offset = height - crop_size
                    x_offset = random.randint(0, max_x_offset)
                    y_offset = random.randint(0, max_y_offset)
                    pil_image = pil_image.crop((x_offset, y_offset, x_offset + crop_size, y_offset + crop_size))
                    # 调整回原始尺寸
                    pil_image = pil_image.resize((width, height), Image.BILINEAR)
                    
            elif augmentation_type == "gaussian_noise":
                # 添加高斯噪声
                image_array = np.array(pil_image).astype(np.float32)
                noise = np.random.normal(0, noise_intensity * 255, image_array.shape)
                noisy_image = image_array + noise
                noisy_image = np.clip(noisy_image, 0, 255).astype(np.uint8)
                pil_image = Image.fromarray(noisy_image)
                
            elif augmentation_type == "brightness":
                brightness = random.uniform(max(0, 1 - brightness_factor), 1 + brightness_factor)
                enhancer = ImageEnhance.Brightness(pil_image)
                pil_image = enhancer.enhance(brightness)
                
            elif augmentation_type == "contrast":
                contrast = random.uniform(max(0, 1 - contrast_factor), 1 + contrast_factor)
                enhancer = ImageEnhance.Contrast(pil_image)
                pil_image = enhancer.enhance(contrast)
                
            elif augmentation_type == "saturation":
                saturation = random.uniform(max(0, 1 - saturation_factor), 1 + saturation_factor)
                enhancer = ImageEnhance.Color(pil_image)
                pil_image = enhancer.enhance(saturation)
                
            elif augmentation_type == "hue":
                if abs(hue_factor) > 1e-6:
                    hue = random.uniform(-hue_factor, hue_factor)
                    hsv_image = pil_image.convert('HSV')
                    hsv_array = np.array(hsv_image)
                    hsv_array[:, :, 0] = (hsv_array[:, :, 0].astype(np.float32) + hue * 255) % 255
                    hsv_image = Image.fromarray(hsv_array, 'HSV')
                    pil_image = hsv_image.convert('RGB')
            
            # 转换回tensor并保存
            augmented_image = torch.from_numpy(np.array(pil_image)).float() / 255.0
            augmented_images[i] = augmented_image
            
        return (augmented_images,)