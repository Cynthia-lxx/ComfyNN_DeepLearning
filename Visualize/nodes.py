# ComfyNN 可视化节点
import torch
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
from PIL import Image
import base64
import json
from comfy.comfy_types.node_typing import IO
import folder_paths

# 定义之前创建的TENSOR类型
class TensorDataType:
    TENSOR = "TENSOR"

class TensorToImageVisualizer:
    """将张量可视化为图像"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "colormap": (["viridis", "plasma", "inferno", "magma", "cividis", "gray", "hot", "cool"], 
                            {"default": "viridis"}),
                "normalize": (["none", "minmax", "standard"], {"default": "minmax"}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("image",)
    FUNCTION = "visualize"
    CATEGORY = "ComfyNN/Visualize"
    OUTPUT_NODE = True
    DESCRIPTION = "将张量数据可视化为图像"

    def visualize(self, tensor, colormap, normalize):
        # 处理不同维度的张量
        if tensor.dim() == 4:
            # 如果是4D张量 [B, H, W, C]，取第一个批次
            tensor = tensor[0]
        if tensor.dim() == 3:
            # 如果是3D张量 [H, W, C]
            if tensor.shape[2] == 1:
                # 单通道
                data = tensor.squeeze(2)
            elif tensor.shape[2] <= 3:
                # RGB或RGBA，直接返回
                # 转换为0-1范围
                data = torch.clamp(tensor, 0, 1)
            else:
                # TODO: 支持多通道可视化选择和比较
                # 多通道，取第一个通道
                data = tensor[:, :, 0]
        elif tensor.dim() == 2:
            # 2D张量 [H, W]
            data = tensor
        else:
            # 1D或其他维度，重塑为方形
            size = int(np.sqrt(tensor.numel()))
            data = tensor[:size*size].view(size, size)
        
        # 标准化处理
        if normalize == "minmax":
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
            else:
                data = torch.zeros_like(data)
        elif normalize == "standard":
            data_mean = data.mean()
            data_std = data.std()
            if data_std > 0:
                data = (data - data_mean) / data_std
                # 限制在一定范围内以避免极值
                data = torch.clamp(data, -3, 3)
                # 转换到0-1范围
                data = (data + 3) / 6
            else:
                data = torch.zeros_like(data)
        
        # 确保数据在0-1范围内
        data = torch.clamp(data, 0, 1)
        
        # 转换为numpy数组
        data_np = data.cpu().numpy()
        
        # 应用颜色映射
        cmap = plt.get_cmap(colormap)
        colored_data = cmap(data_np)
        
        # 转换为RGB格式（去掉alpha通道）
        if colored_data.shape[2] == 4:
            colored_data = colored_data[:, :, :3]
        
        # 转换为torch tensor格式 [H, W, C]
        image_tensor = torch.from_numpy(colored_data).float()
        
        # 添加批次维度，使其成为 [1, H, W, C] 格式
        image_tensor = image_tensor.unsqueeze(0)
        
        return (image_tensor,)


class TensorHeatmapVisualizer:
    """将张量可视化为热力图，支持详细参数调节"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "colormap": ([
                    "viridis", "plasma", "inferno", "magma", "cividis", 
                    "gray", "hot", "cool", "coolwarm", "seismic",
                    "autumn", "spring", "winter", "summer"
                ], {"default": "viridis"}),
                "normalize": (["none", "minmax", "standard"], {"default": "minmax"}),
                "interpolation": (["nearest", "bilinear", "bicubic"], {"default": "bilinear"}),
                "max_size": ("INT", {"default": 512, "min": 64, "max": 2048, "step": 64}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("heatmap_image",)
    FUNCTION = "visualize"
    CATEGORY = "ComfyNN/Visualize"
    OUTPUT_NODE = True
    DESCRIPTION = "将张量数据可视化为热力图，支持详细参数调节"

    def visualize(self, tensor, colormap, normalize, interpolation, max_size):
        # 处理不同维度的张量
        if tensor.dim() == 4:
            # 如果是4D张量 [B, H, W, C]，取第一个批次的第一个通道
            tensor = tensor[0, :, :, 0] if tensor.shape[3] >= 1 else tensor[0]
        if tensor.dim() == 3:
            # 如果是3D张量 [H, W, C]，取第一个通道
            tensor = tensor[:, :, 0] if tensor.shape[2] >= 1 else tensor
        elif tensor.dim() == 2:
            # 2D张量 [H, W]
            pass
        else:
            # 1D或其他维度，重塑为方形
            size = int(np.sqrt(tensor.numel()))
            tensor = tensor[:size*size].view(size, size)
        
        # 性能优化：如果张量太大，进行下采样
        h, w = tensor.shape
        if h > max_size or w > max_size:
            # 计算缩放因子
            scale_factor = min(max_size / h, max_size / w)
            new_h, new_w = int(h * scale_factor), int(w * scale_factor)
            
            # 使用插值进行下采样
            tensor = torch.nn.functional.interpolate(
                tensor.unsqueeze(0).unsqueeze(0), 
                size=(new_h, new_w), 
                mode=interpolation, 
                align_corners=False if interpolation != 'nearest' else None
            ).squeeze(0).squeeze(0)
        
        # 标准化处理
        data = tensor.clone()
        if normalize == "minmax":
            data_min = data.min()
            data_max = data.max()
            if data_max > data_min:
                data = (data - data_min) / (data_max - data_min)
            else:
                data = torch.zeros_like(data)
        elif normalize == "standard":
            data_mean = data.mean()
            data_std = data.std()
            if data_std > 0:
                data = (data - data_mean) / data_std
                # 限制在一定范围内以避免极值
                data = torch.clamp(data, -3, 3)
                # 转换到0-1范围
                data = (data + 3) / 6
            else:
                data = torch.zeros_like(data)
        
        # 确保数据在0-1范围内
        data = torch.clamp(data, 0, 1)
        
        # 转换为numpy数组
        data_np = data.cpu().numpy()
        
        # 创建热力图
        plt.figure(figsize=(8, 6))
        plt.imshow(data_np, cmap=colormap, interpolation=interpolation)
        plt.colorbar()
        plt.tight_layout()
        
        # 保存为图像
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # 转换为PIL图像
        pil_image = Image.open(buf)
        
        # 转换为torch tensor格式 [H, W, C]
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)
        
        # 添加批次维度，使其成为 [1, H, W, C] 格式
        image_tensor = image_tensor.unsqueeze(0)
        
        return (image_tensor,)


class TensorShapeVisualizer:
    """显示张量形状信息的图形化表示"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "display_mode": (["diagram", "text"], {"default": "diagram"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("shape_image", "shape_info")
    FUNCTION = "visualize"
    CATEGORY = "ComfyNN/Visualize"
    OUTPUT_NODE = True
    DESCRIPTION = "显示张量的形状信息"

    def visualize(self, tensor, display_mode):
        shape_info = f"Tensor Shape: {list(tensor.shape)}\n"
        shape_info += f"Dimensions: {tensor.dim()}\n"
        shape_info += f"Total Elements: {tensor.numel()}\n"
        shape_info += f"Data Type: {tensor.dtype}\n"
        shape_info += f"Device: {tensor.device}"
        
        if display_mode == "text":
            # 创建文本图像
            fig, ax = plt.subplots(figsize=(10, 4))
            ax.text(0.1, 0.5, shape_info, fontsize=12, verticalalignment='center', fontfamily='monospace')
            ax.axis('off')
            plt.tight_layout()
            
            # 保存为图像
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            # 转换为PIL图像
            pil_image = Image.open(buf)
            
            # 转换为torch tensor格式 [H, W, C]
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)
            
            # 添加批次维度，使其成为 [1, H, W, C] 格式
            image_tensor = image_tensor.unsqueeze(0)
            
            return (image_tensor, shape_info)
        else:  # diagram模式
            # 创建形状图解
            fig, ax = plt.subplots(figsize=(10, 6))
            
            # 绘制维度框图
            dims = list(tensor.shape)
            dim_names = [f"Dim {i}" for i in range(len(dims))]
            
            # 绘制柱状图表示各维度大小
            bars = ax.bar(range(len(dims)), dims, color=['#FF6B35', '#4A90E2', '#50C878', '#FFD700', '#9370DB'])
            
            # 添加数值标签
            for i, (bar, dim) in enumerate(zip(bars, dims)):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + max(dims)*0.01,
                       f'{dim}', ha='center', va='bottom')
            
            # 设置标签
            ax.set_xlabel('Dimensions')
            ax.set_ylabel('Size')
            ax.set_title(f'Tensor Shape Diagram\nTotal Elements: {tensor.numel()}')
            ax.set_xticks(range(len(dims)))
            ax.set_xticklabels(dim_names)
            ax.grid(axis='y', alpha=0.3)
            
            plt.tight_layout()
            
            # 保存为图像
            buf = BytesIO()
            plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
            buf.seek(0)
            plt.close()
            
            # 转换为PIL图像
            pil_image = Image.open(buf)
            
            # 转换为torch tensor格式 [H, W, C]
            image_np = np.array(pil_image).astype(np.float32) / 255.0
            image_tensor = torch.from_numpy(image_np)
            
            # 添加批次维度，使其成为 [1, H, W, C] 格式
            image_tensor = image_tensor.unsqueeze(0)
            
            return (image_tensor, shape_info)


class TensorLineChartVisualizer:
    """将张量的特定维度可视化为折线图"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "max_samples": ("INT", {"default": 1000, "min": 10, "max": 10000, "step": 10}),
                "line_color": (["blue", "red", "green", "orange", "purple", "brown"], {"default": "blue"}),
                "line_width": ("FLOAT", {"default": 1.0, "min": 0.5, "max": 5.0, "step": 0.5}),
            },
            "optional": {
                "dimension": ("INT", {"default": 0, "min": 0, "max": 10, "step": 1}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    RETURN_NAMES = ("linechart_image",)
    FUNCTION = "visualize"
    CATEGORY = "ComfyNN/Visualize"
    OUTPUT_NODE = True
    DESCRIPTION = "将张量的特定维度可视化为折线图"

    def visualize(self, tensor, max_samples, line_color, line_width, dimension=0):
        # 展平张量
        flat_tensor = tensor.flatten()
        
        # 性能优化：如果张量太大，进行下采样
        total_elements = flat_tensor.numel()
        if total_elements > max_samples:
            # 选择均匀分布的样本点
            indices = torch.linspace(0, total_elements-1, max_samples).long()
            data = flat_tensor[indices]
        else:
            data = flat_tensor
        
        # 转换为numpy数组
        data_np = data.cpu().numpy()
        x_values = np.arange(len(data_np))
        
        # 创建折线图
        plt.figure(figsize=(10, 6))
        plt.plot(x_values, data_np, color=line_color, linewidth=line_width)
        plt.xlabel('Index')
        plt.ylabel('Value')
        plt.title(f'Tensor Line Chart (Dimension {dimension})\nTotal Points: {len(data_np)}')
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        
        # 保存为图像
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # 转换为PIL图像
        pil_image = Image.open(buf)
        
        # 转换为torch tensor格式 [H, W, C]
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)
        
        # 添加批次维度，使其成为 [1, H, W, C] 格式
        image_tensor = image_tensor.unsqueeze(0)
        
        return (image_tensor,)


class TensorStatisticsVisualizer:
    """显示张量统计信息的图形化表示"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "chart_type": (["bar", "pie"], {"default": "bar"}),
            }
        }

    RETURN_TYPES = ("IMAGE", "STRING")
    RETURN_NAMES = ("stats_image", "statistics")
    FUNCTION = "visualize"
    CATEGORY = "ComfyNN/Visualize"
    OUTPUT_NODE = True
    DESCRIPTION = "显示张量的统计信息"

    def visualize(self, tensor, chart_type):
        # 计算统计信息
        stats = {
            "Min": float(tensor.min()),
            "Max": float(tensor.max()),
            "Mean": float(tensor.mean()),
            "Std": float(tensor.std()),
            "Sum": float(tensor.sum()),
        }
        
        # 添加额外统计信息
        stats["Median"] = float(torch.median(tensor))
        stats["Variance"] = float(torch.var(tensor))
        
        # 格式化为字符串
        stats_str = "\n".join([f"{k}: {v:.6f}" if isinstance(v, float) else f"{k}: {v}" for k, v in stats.items()])
        
        # 创建统计图表
        stat_names = list(stats.keys())
        stat_values = list(stats.values())
        
        plt.figure(figsize=(10, 6))
        
        if chart_type == "bar":
            # 创建柱状图
            bars = plt.bar(range(len(stat_names)), stat_values, 
                          color=['#FF6B35', '#4A90E2', '#50C878', '#FFD700', '#9370DB', '#FF69B4', '#8B4513'])
            
            # 添加数值标签
            for i, (bar, value) in enumerate(zip(bars, stat_values)):
                height = bar.get_height()
                plt.text(bar.get_x() + bar.get_width()/2., height + max(abs(np.array(stat_values)))*0.01,
                        f'{value:.3f}', ha='center', va='bottom')
            
            plt.xlabel('Statistics')
            plt.ylabel('Value')
            plt.title('Tensor Statistics')
            plt.xticks(range(len(stat_names)), stat_names, rotation=45)
            plt.grid(axis='y', alpha=0.3)
            
        else:  # pie chart
            # 创建饼图（仅适用于正值）
            abs_values = [abs(v) for v in stat_values]
            plt.pie(abs_values, labels=stat_names, autopct='%1.1f%%', startangle=90)
            plt.title('Tensor Statistics Distribution')
        
        plt.tight_layout()
        
        # 保存为图像
        buf = BytesIO()
        plt.savefig(buf, format='png', dpi=100, bbox_inches='tight')
        buf.seek(0)
        plt.close()
        
        # 转换为PIL图像
        pil_image = Image.open(buf)
        
        # 转换为torch tensor格式 [H, W, C]
        image_np = np.array(pil_image).astype(np.float32) / 255.0
        image_tensor = torch.from_numpy(image_np)
        
        # 添加批次维度，使其成为 [1, H, W, C] 格式
        image_tensor = image_tensor.unsqueeze(0)
        
        return (image_tensor, stats_str)