# ComfyNN 张量创建节点
import torch
import numpy as np

class TensorDataType:
    TENSOR = "TENSOR"

class TensorCreator:
    """创建各种类型的张量"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape": ("STRING", {"default": "1, 2, 3", "multiline": False}),
                "dtype": (["float32", "float64", "int32", "int64", "bool"], {"default": "float32"}),
            },
            "optional": {
                "fill_value": ("FLOAT", {"default": 0.0}),
                "requires_grad": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "create"
    CATEGORY = "ComfyNN/DataTypes/Creation"
    DESCRIPTION = "创建指定形状和数据类型的张量"

    def create(self, shape, dtype, fill_value=0.0, requires_grad=False):
        # 解析形状
        shape_list = [int(x.strip()) for x in shape.split(",") if x.strip()]
        
        # 确定数据类型
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        torch_dtype = dtype_map[dtype]
        
        # 创建张量
        tensor = torch.full(shape_list, fill_value, dtype=torch_dtype, requires_grad=requires_grad)
        
        return (tensor,)


class RandomTensorCreator:
    """创建随机张量"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "shape": ("STRING", {"default": "3, 4", "multiline": False}),
                "random_type": (["uniform", "normal", "randint"], {"default": "uniform"}),
                "dtype": (["float32", "float64", "int32", "int64"], {"default": "float32"}),
            },
            "optional": {
                "min_val": ("FLOAT", {"default": 0.0}),
                "max_val": ("FLOAT", {"default": 1.0}),
                "mean": ("FLOAT", {"default": 0.0}),
                "std": ("FLOAT", {"default": 1.0}),
                "requires_grad": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "create"
    CATEGORY = "ComfyNN/DataTypes/Creation"
    DESCRIPTION = "创建随机张量"

    def create(self, shape, random_type, dtype, min_val=0.0, max_val=1.0, mean=0.0, std=1.0, requires_grad=False):
        # 解析形状
        shape_list = [int(x.strip()) for x in shape.split(",") if x.strip()]
        
        # 确定数据类型
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        torch_dtype = dtype_map[dtype]
        
        # 创建随机张量
        if random_type == "uniform":
            tensor = torch.empty(shape_list, dtype=torch_dtype, requires_grad=requires_grad).uniform_(min_val, max_val)
        elif random_type == "normal":
            tensor = torch.empty(shape_list, dtype=torch_dtype, requires_grad=requires_grad).normal_(mean, std)
        elif random_type == "randint":
            tensor = torch.randint(int(min_val), int(max_val), shape_list, dtype=torch_dtype, requires_grad=requires_grad)
        
        return (tensor,)


class SpecialTensorCreator:
    """创建特殊类型的张量（如单位矩阵、对角矩阵等）"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor_type": (["zeros", "ones", "eye", "identity", "arange"], {"default": "zeros"}),
                "dtype": (["float32", "float64", "int32", "int64"], {"default": "float32"}),
            },
            "optional": {
                "size": ("INT", {"default": 5, "min": 1, "max": 1000}),
                "shape": ("STRING", {"default": "3, 4", "multiline": False}),
                "start": ("INT", {"default": 0}),
                "end": ("INT", {"default": 10}),
                "step": ("INT", {"default": 1}),
                "requires_grad": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "create"
    CATEGORY = "ComfyNN/DataTypes/Creation"
    DESCRIPTION = "创建特殊类型的张量"

    def create(self, tensor_type, dtype, size=5, shape="3, 4", start=0, end=10, step=1, requires_grad=False):
        # 确定数据类型
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
        }
        torch_dtype = dtype_map[dtype]
        
        # 解析形状
        shape_list = [int(x.strip()) for x in shape.split(",") if x.strip()]
        
        # 创建特殊张量
        if tensor_type == "zeros":
            if len(shape_list) == 1:
                tensor = torch.zeros(shape_list[0], dtype=torch_dtype, requires_grad=requires_grad)
            else:
                tensor = torch.zeros(shape_list, dtype=torch_dtype, requires_grad=requires_grad)
        elif tensor_type == "ones":
            if len(shape_list) == 1:
                tensor = torch.ones(shape_list[0], dtype=torch_dtype, requires_grad=requires_grad)
            else:
                tensor = torch.ones(shape_list, dtype=torch_dtype, requires_grad=requires_grad)
        elif tensor_type == "eye" or tensor_type == "identity":
            tensor = torch.eye(size, dtype=torch_dtype, requires_grad=requires_grad)
        elif tensor_type == "arange":
            tensor = torch.arange(start, end, step, dtype=torch_dtype, requires_grad=requires_grad)
        
        return (tensor,)


class TensorReshaper:
    """重塑张量形状"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "new_shape": ("STRING", {"default": "2, 6", "multiline": False}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("reshaped_tensor",)
    FUNCTION = "reshape"
    CATEGORY = "ComfyNN/DataTypes/Manipulation"
    DESCRIPTION = "重塑张量形状"

    def reshape(self, tensor, new_shape):
        # 解析新形状
        shape_list = [int(x.strip()) for x in new_shape.split(",") if x.strip()]
        
        # 重塑张量
        reshaped_tensor = tensor.reshape(shape_list)
        
        return (reshaped_tensor,)


class TensorConverter:
    """转换张量数据类型"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "dtype": (["float32", "float64", "int32", "int64", "bool"], {"default": "float32"}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("converted_tensor",)
    FUNCTION = "convert"
    CATEGORY = "ComfyNN/DataTypes/Manipulation"
    DESCRIPTION = "转换张量数据类型"

    def convert(self, tensor, dtype):
        # 确定数据类型
        dtype_map = {
            "float32": torch.float32,
            "float64": torch.float64,
            "int32": torch.int32,
            "int64": torch.int64,
            "bool": torch.bool,
        }
        torch_dtype = dtype_map[dtype]
        
        # 转换张量数据类型
        converted_tensor = tensor.to(torch_dtype)
        
        return (converted_tensor,)