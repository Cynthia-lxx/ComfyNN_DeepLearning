# ComfyNN 深度学习计算节点
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
import json
from comfy.comfy_types.node_typing import IO
import folder_paths

# 定义之前创建的TENSOR类型
class TensorDataType:
    TENSOR = "TENSOR"

# ========== 数据处理节点 ==========

class TensorDataLoader:
    """简单的数据加载器"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "data": (TensorDataType.TENSOR,),
                "labels": (TensorDataType.TENSOR,),
                "batch_size": ("INT", {"default": 32, "min": 1, "max": 1024, "step": 1}),
                "shuffle": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR", "INT")
    RETURN_NAMES = ("batch_data", "batch_labels", "num_batches")
    FUNCTION = "load_data"
    CATEGORY = "ComfyNN/DLCompute/Data"
    DESCRIPTION = "将数据分批处理"

    def load_data(self, data, labels, batch_size, shuffle):
        # 获取数据大小
        data_size = data.shape[0]
        num_batches = (data_size + batch_size - 1) // batch_size
        
        # 创建索引
        indices = list(range(data_size))
        if shuffle:
            random.shuffle(indices)
        
        # 简化处理：这里我们只返回第一个批次
        batch_indices = indices[:batch_size]
        batch_data = data[batch_indices]
        batch_labels = labels[batch_indices]
        
        return (batch_data, batch_labels, num_batches)


class TensorNormalize:
    """归一化张量数据"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "method": (["minmax", "zscore", "unit"], {"default": "zscore"}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("normalized_tensor",)
    FUNCTION = "normalize"
    CATEGORY = "ComfyNN/DLCompute/Data"
    DESCRIPTION = "归一化张量数据"

    def normalize(self, tensor, method):
        if method == "minmax":
            # Min-Max归一化到[0,1]
            tensor_min = tensor.min()
            tensor_max = tensor.max()
            if tensor_max > tensor_min:
                normalized = (tensor - tensor_min) / (tensor_max - tensor_min)
            else:
                normalized = torch.zeros_like(tensor)
        elif method == "zscore":
            # Z-Score标准化
            mean = tensor.mean()
            std = tensor.std()
            if std > 0:
                normalized = (tensor - mean) / std
            else:
                normalized = torch.zeros_like(tensor)
        elif method == "unit":
            # 单位向量归一化
            norm = tensor.norm(p=2, dim=-1, keepdim=True)
            normalized = tensor / (norm + 1e-8)
        
        return (normalized,)


class TensorAugment:
    """简单的数据增强"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "augmentation_type": (["noise", "flip", "rotate"], {"default": "noise"}),
                "intensity": ("FLOAT", {"default": 0.1, "min": 0.0, "max": 1.0, "step": 0.01}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("augmented_tensor",)
    FUNCTION = "augment"
    CATEGORY = "ComfyNN/DLCompute/Data"
    DESCRIPTION = "对张量数据进行简单增强"

    def augment(self, tensor, augmentation_type, intensity):
        if augmentation_type == "noise":
            # 添加高斯噪声
            noise = torch.randn_like(tensor) * intensity
            augmented = tensor + noise
        elif augmentation_type == "flip":
            # 随机翻转
            if random.random() > 0.5:
                augmented = torch.flip(tensor, dims=[-1])  # 翻转最后一个维度
            else:
                augmented = tensor
        elif augmentation_type == "rotate":
            # 简单旋转（仅适用于方形图像）
            if tensor.dim() >= 2 and tensor.shape[-1] == tensor.shape[-2]:
                k = random.randint(0, 3)  # 0, 1, 2, 3 表示 0°, 90°, 180°, 270°
                augmented = torch.rot90(tensor, k=k, dims=[-2, -1])
            else:
                augmented = tensor
        
        return (augmented,)

# ========== 模型层节点 ==========

class TensorLinearLayer:
    """线性层（全连接层）"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_tensor": (TensorDataType.TENSOR,),
                "in_features": ("INT", {"default": 784, "min": 1, "max": 10000, "step": 1}),
                "out_features": ("INT", {"default": 128, "min": 1, "max": 10000, "step": 1}),
                "bias": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("output_tensor",)
    FUNCTION = "forward"
    CATEGORY = "ComfyNN/DLCompute/Layers"
    DESCRIPTION = "线性变换层"

    def forward(self, input_tensor, in_features, out_features, bias):
        # 创建线性层权重和偏置
        weight = torch.randn(out_features, in_features) * np.sqrt(2.0 / in_features)
        if bias:
            bias_tensor = torch.zeros(out_features)
        else:
            bias_tensor = None
        
        # 确保输入张量的最后一个维度与in_features匹配
        if input_tensor.shape[-1] != in_features:
            # 重塑输入张量
            input_tensor = input_tensor.view(-1, in_features)
        
        # 执行线性变换
        if bias_tensor is not None:
            output = torch.matmul(input_tensor, weight.t()) + bias_tensor
        else:
            output = torch.matmul(input_tensor, weight.t())
        
        return (output,)


class TensorConv2DLayer:
    """2D卷积层"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_tensor": (TensorDataType.TENSOR,),
                "in_channels": ("INT", {"default": 3, "min": 1, "max": 512, "step": 1}),
                "out_channels": ("INT", {"default": 32, "min": 1, "max": 512, "step": 1}),
                "kernel_size": ("INT", {"default": 3, "min": 1, "max": 7, "step": 1}),
                "stride": ("INT", {"default": 1, "min": 1, "max": 4, "step": 1}),
                "padding": ("INT", {"default": 1, "min": 0, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("output_tensor",)
    FUNCTION = "forward"
    CATEGORY = "ComfyNN/DLCompute/Layers"
    DESCRIPTION = "2D卷积层"

    def forward(self, input_tensor, in_channels, out_channels, kernel_size, stride, padding):
        # 确保输入张量是4D [B, C, H, W]
        if input_tensor.dim() == 3:
            input_tensor = input_tensor.unsqueeze(0)
        elif input_tensor.dim() == 2:
            # 假设是展平的图像，尝试重塑
            h = w = int(np.sqrt(input_tensor.shape[-1] / in_channels))
            input_tensor = input_tensor.view(-1, in_channels, h, w)
        
        # 创建卷积核权重和偏置
        weight = torch.randn(out_channels, in_channels, kernel_size, kernel_size) * np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        bias = torch.zeros(out_channels)
        
        # 执行卷积操作
        output = F.conv2d(input_tensor, weight, bias, stride=stride, padding=padding)
        
        return (output,)


class TensorActivation:
    """激活函数层"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_tensor": (TensorDataType.TENSOR,),
                "activation_type": (["relu", "sigmoid", "tanh", "softmax"], {"default": "relu"}),
                "dim": ("INT", {"default": -1, "min": -4, "max": 3, "step": 1}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("output_tensor",)
    FUNCTION = "activate"
    CATEGORY = "ComfyNN/DLCompute/Layers"
    DESCRIPTION = "激活函数层"

    def activate(self, input_tensor, activation_type, dim):
        if activation_type == "relu":
            output = F.relu(input_tensor)
        elif activation_type == "sigmoid":
            output = torch.sigmoid(input_tensor)
        elif activation_type == "tanh":
            output = torch.tanh(input_tensor)
        elif activation_type == "softmax":
            output = F.softmax(input_tensor, dim=dim)
        else:
            output = input_tensor  # 默认不改变
        
        return (output,)

# ========== 损失函数节点 ==========

class TensorMSELoss:
    """均方误差损失"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "predictions": (TensorDataType.TENSOR,),
                "targets": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("loss",)
    FUNCTION = "compute"
    CATEGORY = "ComfyNN/DLCompute/Loss"
    DESCRIPTION = "均方误差损失函数"

    def compute(self, predictions, targets):
        # 计算MSE损失
        loss = F.mse_loss(predictions, targets)
        return (loss,)


class TensorCrossEntropyLoss:
    """交叉熵损失"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "predictions": (TensorDataType.TENSOR,),
                "targets": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("loss",)
    FUNCTION = "compute"
    CATEGORY = "ComfyNN/DLCompute/Loss"
    DESCRIPTION = "交叉熵损失函数"

    def compute(self, predictions, targets):
        # 确保目标是类别索引而不是one-hot编码
        if targets.dim() > 1 and targets.shape[-1] != 1:
            # 假设targets是one-hot编码，转换为类别索引
            targets = torch.argmax(targets, dim=-1)
        
        # 如果targets是多维的，将其展平
        if targets.dim() > 1:
            targets = targets.squeeze()
        
        # 计算交叉熵损失
        loss = F.cross_entropy(predictions, targets)
        return (loss,)

# ========== 优化器节点 ==========

class TensorSGDOptimizer:
    """随机梯度下降优化器"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parameters": (TensorDataType.TENSOR,),  # 这里简化处理，实际应该是一组参数
                "learning_rate": ("FLOAT", {"default": 0.01, "min": 0.0001, "max": 1.0, "step": 0.0001}),
                "momentum": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 0.99, "step": 0.01}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("updated_parameters",)
    FUNCTION = "optimize"
    CATEGORY = "ComfyNN/DLCompute/Optimizer"
    DESCRIPTION = "随机梯度下降优化器"

    def optimize(self, parameters, learning_rate, momentum):
        # 这里简化处理，实际优化器需要维护状态
        # 我们只是简单地返回参数，表示优化器已创建
        return (parameters,)


class TensorAdamOptimizer:
    """Adam优化器"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parameters": (TensorDataType.TENSOR,),
                "learning_rate": ("FLOAT", {"default": 0.001, "min": 0.00001, "max": 1.0, "step": 0.00001}),
                "beta1": ("FLOAT", {"default": 0.9, "min": 0.0, "max": 1.0, "step": 0.01}),
                "beta2": ("FLOAT", {"default": 0.999, "min": 0.0, "max": 1.0, "step": 0.001}),
                "epsilon": ("FLOAT", {"default": 1e-8, "min": 1e-10, "max": 1e-5, "step": 1e-9}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("updated_parameters",)
    FUNCTION = "optimize"
    CATEGORY = "ComfyNN/DLCompute/Optimizer"
    DESCRIPTION = "Adam优化器"

    def optimize(self, parameters, learning_rate, beta1, beta2, epsilon):
        # 这里简化处理，实际优化器需要维护状态
        # 我们只是简单地返回参数，表示优化器已创建
        return (parameters,)

# ========== 训练节点 ==========

class TensorForwardPass:
    """前向传播"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_tensor": (TensorDataType.TENSOR,),
                "model_weights": (TensorDataType.TENSOR,),  # 简化处理
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("output_tensor",)
    FUNCTION = "forward"
    CATEGORY = "ComfyNN/DLCompute/Training"
    DESCRIPTION = "执行前向传播"

    def forward(self, input_tensor, model_weights):
        # 简化处理：这里只是返回输入，实际应该执行模型的前向传播
        # 在完整的实现中，这里会应用模型权重到输入数据
        return (input_tensor,)


class TensorBackwardPass:
    """反向传播"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "loss": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("gradients",)
    FUNCTION = "backward"
    CATEGORY = "ComfyNN/DLCompute/Training"
    DESCRIPTION = "执行反向传播"

    def backward(self, loss):
        # 简化处理：这里只是返回一个与loss形状相同的张量作为梯度
        # 实际应用中，这里会计算损失相对于模型参数的梯度
        gradients = torch.ones_like(loss)  # 占位符
        return (gradients,)


class TensorUpdateWeights:
    """更新权重"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "parameters": (TensorDataType.TENSOR,),
                "gradients": (TensorDataType.TENSOR,),
                "learning_rate": ("FLOAT", {"default": 0.001, "min": 0.0001, "max": 1.0, "step": 0.0001}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("updated_parameters",)
    FUNCTION = "update"
    CATEGORY = "ComfyNN/DLCompute/Training"
    DESCRIPTION = "更新模型权重"

    def update(self, parameters, gradients, learning_rate):
        # 简化的权重更新
        # 实际中，这会根据优化器类型和参数来更新权重
        updated_params = parameters - learning_rate * gradients
        return (updated_params,)