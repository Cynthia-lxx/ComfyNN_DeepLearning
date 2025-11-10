# ComfyNN 基本张量操作节点
import torch
import math
from comfy.comfy_types.node_typing import IO

# 定义之前创建的TENSOR类型
class TensorDataType:
    TENSOR = "TENSOR"

class TensorAdd:
    """张量加法"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor_a": (TensorDataType.TENSOR,),
                "tensor_b": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "对两个张量执行加法操作"

    def execute(self, tensor_a, tensor_b):
        result = tensor_a + tensor_b
        return (result,)


class TensorSubtract:
    """张量减法"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor_a": (TensorDataType.TENSOR,),
                "tensor_b": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "对两个张量执行减法操作"

    def execute(self, tensor_a, tensor_b):
        result = tensor_a - tensor_b
        return (result,)


class TensorMultiply:
    """张量乘法"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor_a": (TensorDataType.TENSOR,),
                "tensor_b": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "对两个张量执行乘法操作"

    def execute(self, tensor_a, tensor_b):
        result = tensor_a * tensor_b
        return (result,)


class TensorDivide:
    """张量除法"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor_a": (TensorDataType.TENSOR,),
                "tensor_b": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "对两个张量执行除法操作"

    def execute(self, tensor_a, tensor_b):
        # 避免除零错误
        epsilon = 1e-8
        result = tensor_a / (tensor_b + epsilon)
        return (result,)


class TensorPower:
    """张量幂运算"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "exponent": ("FLOAT", {"default": 2.0, "min": -10.0, "max": 10.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "对张量执行幂运算"

    def execute(self, tensor, exponent):
        result = torch.pow(tensor, exponent)
        return (result,)


class TensorSqrt:
    """张量平方根"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "计算张量的平方根"

    def execute(self, tensor):
        # 确保输入值非负
        tensor = torch.abs(tensor)
        result = torch.sqrt(tensor)
        return (result,)


class TensorTranspose:
    """张量转置"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "dim0": ("INT", {"default": -1, "min": -4, "max": 3, "step": 1}),
                "dim1": ("INT", {"default": -2, "min": -4, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "交换张量的两个维度"

    def execute(self, tensor, dim0, dim1):
        result = torch.transpose(tensor, dim0, dim1)
        return (result,)


class TensorReshape:
    """张量重塑"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "shape": ("STRING", {"default": "1, -1", "multiline": False}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "重塑张量形状"

    def execute(self, tensor, shape):
        # 解析形状字符串
        shape_list = [int(x.strip()) for x in shape.split(",")]
        result = tensor.reshape(shape_list)
        return (result,)


class TensorSqueeze:
    """张量降维"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "dim": ("INT", {"default": 0, "min": -4, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "移除张量指定维度中大小为1的维度"

    def execute(self, tensor, dim):
        try:
            result = torch.squeeze(tensor, dim)
        except:
            # 如果指定维度大小不为1，则不进行操作
            result = tensor
        return (result,)


class TensorUnsqueeze:
    """张量升维"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "dim": ("INT", {"default": 0, "min": -4, "max": 4, "step": 1}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "在指定位置添加大小为1的维度"

    def execute(self, tensor, dim):
        result = torch.unsqueeze(tensor, dim)
        return (result,)


class TensorConcatenate:
    """张量拼接"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor_a": (TensorDataType.TENSOR,),
                "tensor_b": (TensorDataType.TENSOR,),
                "dim": ("INT", {"default": 0, "min": -4, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "沿指定维度拼接两个张量"

    def execute(self, tensor_a, tensor_b, dim):
        try:
            result = torch.cat([tensor_a, tensor_b], dim=dim)
        except Exception as e:
            # 如果无法拼接，返回第一个张量
            result = tensor_a
        return (result,)


class TensorSum:
    """张量求和"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            },
            "optional": {
                "dim": ("INT", {"default": -1, "min": -4, "max": 3, "step": 1}),
                "keepdim": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "计算张量元素的总和"

    def execute(self, tensor, dim=-1, keepdim=False):
        result = torch.sum(tensor, dim=dim, keepdim=keepdim)
        return (result,)


class TensorMean:
    """张量平均值"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            },
            "optional": {
                "dim": ("INT", {"default": -1, "min": -4, "max": 3, "step": 1}),
                "keepdim": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "计算张量元素的平均值"

    def execute(self, tensor, dim=-1, keepdim=False):
        result = torch.mean(tensor, dim=dim, keepdim=keepdim)
        return (result,)


class TensorMax:
    """张量最大值"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            },
            "optional": {
                "dim": ("INT", {"default": -1, "min": -4, "max": 3, "step": 1}),
                "keepdim": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR, TensorDataType.TENSOR,)
    RETURN_NAMES = ("values", "indices")
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "计算张量元素的最大值"

    def execute(self, tensor, dim=-1, keepdim=False):
        values, indices = torch.max(tensor, dim=dim, keepdim=keepdim)
        return (values, indices)


class TensorMin:
    """张量最小值"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            },
            "optional": {
                "dim": ("INT", {"default": -1, "min": -4, "max": 3, "step": 1}),
                "keepdim": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR, TensorDataType.TENSOR,)
    RETURN_NAMES = ("values", "indices")
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "计算张量元素的最小值"

    def execute(self, tensor, dim=-1, keepdim=False):
        values, indices = torch.min(tensor, dim=dim, keepdim=keepdim)
        return (values, indices)


class TensorAbs:
    """张量绝对值"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "计算张量元素的绝对值"

    def execute(self, tensor):
        result = torch.abs(tensor)
        return (result,)


class TensorSin:
    """张量正弦值"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "计算张量元素的正弦值"

    def execute(self, tensor):
        result = torch.sin(tensor)
        return (result,)


class TensorCos:
    """张量余弦值"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "计算张量元素的余弦值"

    def execute(self, tensor):
        result = torch.cos(tensor)
        return (result,)


class TensorExp:
    """张量指数"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "计算张量元素的指数值 (e^x)"

    def execute(self, tensor):
        result = torch.exp(tensor)
        return (result,)


class TensorLog:
    """张量对数"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic"
    DESCRIPTION = "计算张量元素的自然对数 (ln(x))"

    def execute(self, tensor):
        # 确保输入值为正数
        tensor = torch.abs(tensor) + 1e-8
        result = torch.log(tensor)
        return (result,)


# ========== 激活函数节点 ==========
class TensorReLU:
    """ReLU激活函数"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic/Activation"
    DESCRIPTION = "ReLU (Rectified Linear Unit)激活函数: max(0, x)"

    def execute(self, tensor):
        result = torch.relu(tensor)
        return (result,)


class TensorLeakyReLU:
    """LeakyReLU激活函数"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "negative_slope": ("FLOAT", {"default": 0.01, "min": 0.0, "max": 1.0, "step": 0.001}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic/Activation"
    DESCRIPTION = "LeakyReLU激活函数: max(0, x) + negative_slope * min(0, x)"

    def execute(self, tensor, negative_slope):
        result = torch.nn.functional.leaky_relu(tensor, negative_slope=negative_slope)
        return (result,)


class TensorSigmoid:
    """Sigmoid激活函数"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic/Activation"
    DESCRIPTION = "Sigmoid激活函数: 1 / (1 + exp(-x))"

    def execute(self, tensor):
        result = torch.sigmoid(tensor)
        return (result,)


class TensorTanh:
    """Tanh激活函数"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic/Activation"
    DESCRIPTION = "Tanh激活函数: (exp(x) - exp(-x)) / (exp(x) + exp(-x))"

    def execute(self, tensor):
        result = torch.tanh(tensor)
        return (result,)


class TensorSoftmax:
    """Softmax激活函数"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "dim": ("INT", {"default": -1, "min": -4, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic/Activation"
    DESCRIPTION = "Softmax激活函数: exp(x) / sum(exp(x))"

    def execute(self, tensor, dim):
        result = torch.softmax(tensor, dim=dim)
        return (result,)


class TensorELU:
    """ELU激活函数"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "alpha": ("FLOAT", {"default": 1.0, "min": 0.1, "max": 5.0, "step": 0.1}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "execute"
    CATEGORY = "ComfyNN/DLBasic/Activation"
    DESCRIPTION = "ELU (Exponential Linear Unit)激活函数"

    def execute(self, tensor, alpha):
        result = torch.nn.functional.elu(tensor, alpha=alpha)
        return (result,)