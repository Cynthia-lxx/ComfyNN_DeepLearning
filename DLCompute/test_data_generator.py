# ComfyNN DLCompute Test Data Generator
import torch
import random
import numpy as np
from ..DLBasic.nodes import TensorDataType

class DLComputeTestDataGenerator:
    """生成用于DLCompute模块测试的随机数据"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data_type": (["image_classification", "nlp_sequence", "tabular_data"], {"default": "image_classification"}),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 256}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
            },
            "optional": {
                "channels": ("INT", {"default": 3, "min": 1, "max": 3}),
                "height": ("INT", {"default": 32, "min": 8, "max": 256}),
                "width": ("INT", {"default": 32, "min": 8, "max": 256}),
                "num_classes": ("INT", {"default": 10, "min": 2, "max": 1000}),
                "sequence_length": ("INT", {"default": 128, "min": 10, "max": 512}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR, TensorDataType.TENSOR, "INT", "INT")
    RETURN_NAMES = ("test_data", "test_labels", "num_classes", "sequence_length")
    FUNCTION = "generate"
    CATEGORY = "ComfyNN/DLCompute/Testing"
    DESCRIPTION = "生成用于DLCompute模块测试的随机数据"

    def generate(self, data_type, batch_size, seed, channels=3, height=32, width=32, 
                 num_classes=10, sequence_length=128):
        # 设置随机种子
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        
        if data_type == "image_classification":
            # 生成图像分类数据 [B, C, H, W]
            test_data = torch.randn(batch_size, channels, height, width)
            test_labels = torch.randint(0, num_classes, (batch_size,))
            return (test_data, test_labels, num_classes, sequence_length)
            
        elif data_type == "nlp_sequence":
            # 生成NLP序列数据 [B, S, H]
            test_data = torch.randn(batch_size, sequence_length, height)
            test_labels = torch.randint(0, num_classes, (batch_size,))
            return (test_data, test_labels, num_classes, sequence_length)
            
        elif data_type == "tabular_data":
            # 生成表格数据 [B, F]
            features = channels * height  # 使用channels和height作为特征数
            test_data = torch.randn(batch_size, features)
            test_labels = torch.randint(0, num_classes, (batch_size,))
            return (test_data, test_labels, num_classes, sequence_length)