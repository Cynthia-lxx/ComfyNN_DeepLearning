# ComfyNN 数据类型转换节点
import torch
import json
import csv
import os
from io import StringIO
import folder_paths
import comfy.utils
from comfy.comfy_types.node_typing import IO

# 定义自定义数据类型
class TensorDataType:
    """自定义Tensor数据类型"""
    TENSOR = "TENSOR"

class ImageToTensor:
    """将IMAGE转换为TENSOR"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "image": (IO.IMAGE,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "convert"
    CATEGORY = "ComfyNN/DataTypes"
    DESCRIPTION = "将IMAGE转换为TENSOR"

    def convert(self, image):
        # IMAGE已经是torch.Tensor格式 [B, H, W, C]
        # 直接返回作为TENSOR类型
        return (image,)


class TensorToImage:
    """将TENSOR转换为IMAGE"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
            }
        }

    RETURN_TYPES = (IO.IMAGE,)
    RETURN_NAMES = ("image",)
    FUNCTION = "convert"
    CATEGORY = "ComfyNN/DataTypes"
    DESCRIPTION = "将TENSOR转换为IMAGE"

    def convert(self, tensor):
        # 确保tensor是正确的IMAGE格式 [B, H, W, C]
        return (tensor,)


class ModelToTensor:
    """将MODEL转换为TENSOR"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "model": (IO.MODEL,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "convert"
    CATEGORY = "ComfyNN/DataTypes"
    DESCRIPTION = "将MODEL转换为TENSOR"

    def convert(self, model):
        # MODEL对象不能直接转换为Tensor，这里我们返回一个占位符
        # 实际应用中，可能需要访问模型的特定权重
        return (torch.tensor([0.0]),)  # 占位符


class TensorToModel:
    """将TENSOR转换为MODEL"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "reference_model": (IO.MODEL,),
            }
        }

    RETURN_TYPES = (IO.MODEL,)
    RETURN_NAMES = ("model",)
    FUNCTION = "convert"
    CATEGORY = "ComfyNN/DataTypes"
    DESCRIPTION = "将TENSOR转换为MODEL"

    def convert(self, tensor, reference_model):
        # TENSOR不能直接转换为MODEL，这里我们返回参考模型
        # 实际应用中，可能需要用tensor更新模型权重
        return (reference_model,)


class ClipToTensor:
    """将CLIP转换为TENSOR"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "clip": (IO.CLIP,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "convert"
    CATEGORY = "ComfyNN/DataTypes"
    DESCRIPTION = "将CLIP转换为TENSOR"

    def convert(self, clip):
        # CLIP对象不能直接转换为Tensor，这里我们返回一个占位符
        return (torch.tensor([0.0]),)  # 占位符


class TensorToClip:
    """将TENSOR转换为CLIP"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "reference_clip": (IO.CLIP,),
            }
        }

    RETURN_TYPES = (IO.CLIP,)
    RETURN_NAMES = ("clip",)
    FUNCTION = "convert"
    CATEGORY = "ComfyNN/DataTypes"
    DESCRIPTION = "将TENSOR转换为CLIP"

    def convert(self, tensor, reference_clip):
        # TENSOR不能直接转换为CLIP，这里我们返回参考CLIP
        return (reference_clip,)


class VaeToTensor:
    """将VAE转换为TENSOR"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "vae": (IO.VAE,),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "convert"
    CATEGORY = "ComfyNN/DataTypes"
    DESCRIPTION = "将VAE转换为TENSOR"

    def convert(self, vae):
        # VAE对象不能直接转换为Tensor，这里我们返回一个占位符
        return (torch.tensor([0.0]),)  # 占位符


class TensorToVae:
    """将TENSOR转换为VAE"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "reference_vae": (IO.VAE,),
            }
        }

    RETURN_TYPES = (IO.VAE,)
    RETURN_NAMES = ("vae",)
    FUNCTION = "convert"
    CATEGORY = "ComfyNN/DataTypes"
    DESCRIPTION = "将TENSOR转换为VAE"

    def convert(self, tensor, reference_vae):
        # TENSOR不能直接转换为VAE，这里我们返回参考VAE
        return (reference_vae,)


class LoadTensor:
    """从文件加载Tensor"""
    
    @classmethod
    def INPUT_TYPES(s):
        # 支持的文件类型
        input_dir = folder_paths.get_input_directory()
        files = [
            f for f in os.listdir(input_dir) 
            if os.path.isfile(os.path.join(input_dir, f)) and 
            f.endswith(('.pt', '.pth', '.safetensors', '.json', '.txt', '.csv'))
        ]
        return {
            "required": {
                "file_name": (sorted(files), {"tooltip": "要加载的Tensor文件"}),
                "file_type": (["auto", "pickle", "safetensors", "json", "txt", "csv"], 
                             {"default": "auto", "tooltip": "文件类型，auto会自动检测"}),
            }
        }

    RETURN_TYPES = (TensorDataType.TENSOR,)
    RETURN_NAMES = ("tensor",)
    FUNCTION = "load_tensor"
    CATEGORY = "ComfyNN/DataTypes"
    DESCRIPTION = "从多种文件格式加载Tensor数据"

    def load_tensor(self, file_name, file_type):
        file_path = folder_paths.get_annotated_filepath(file_name)
        
        try:
            # 根据文件扩展名或指定类型加载
            if file_type == "auto":
                if file_name.endswith(('.pt', '.pth')):
                    file_type = "pickle"
                elif file_name.endswith('.safetensors'):
                    file_type = "safetensors"
                elif file_name.endswith('.json'):
                    file_type = "json"
                elif file_name.endswith('.txt'):
                    file_type = "txt"
                elif file_name.endswith('.csv'):
                    file_type = "csv"
                else:
                    # 默认使用pickle
                    file_type = "pickle"
            
            # 根据类型加载数据
            if file_type in ["pickle", "safetensors"]:
                # 使用ComfyUI的工具函数加载
                tensor_data = comfy.utils.load_torch_file(file_path, safe_load=True)
                # 如果是字典，提取第一个张量
                if isinstance(tensor_data, dict):
                    # 获取第一个键值对
                    first_key = next(iter(tensor_data))
                    tensor = tensor_data[first_key]
                else:
                    tensor = tensor_data
                    
            elif file_type == "json":
                with open(file_path, 'r') as f:
                    data = json.load(f)
                # 尝试从JSON数据创建张量
                tensor = self._json_to_tensor(data)
                
            elif file_type == "txt":
                with open(file_path, 'r') as f:
                    content = f.read()
                # 尝试解析文本内容为张量
                tensor = self._text_to_tensor(content)
                
            elif file_type == "csv":
                with open(file_path, 'r') as f:
                    reader = csv.reader(f)
                    data = list(reader)
                # 尝试从CSV数据创建张量
                tensor = self._csv_to_tensor(data)
                
            # 确保返回的是张量
            if not isinstance(tensor, torch.Tensor):
                tensor = torch.tensor(tensor)
                
            return (tensor,)
            
        except Exception as e:
            # 出现错误时返回默认张量
            print(f"加载张量文件时出错: {e}")
            return (torch.zeros(1),)

    def _json_to_tensor(self, data):
        """将JSON数据转换为张量"""
        try:
            # 如果是数值列表，直接转换
            if isinstance(data, list) and all(isinstance(x, (int, float)) for x in data):
                return torch.tensor(data)
            # 如果是嵌套列表，转换为多维张量
            elif isinstance(data, list):
                return torch.tensor(data)
            # 如果是字典，尝试提取数值
            elif isinstance(data, dict):
                # 简单处理，提取所有数值
                values = []
                for v in data.values():
                    if isinstance(v, (int, float)):
                        values.append(v)
                    elif isinstance(v, list) and all(isinstance(x, (int, float)) for x in v):
                        values.extend(v)
                return torch.tensor(values) if values else torch.zeros(1)
            else:
                return torch.zeros(1)
        except:
            return torch.zeros(1)

    def _text_to_tensor(self, content):
        """将文本内容转换为张量"""
        try:
            # 尝试解析为数值列表
            lines = content.strip().split('\n')
            values = []
            for line in lines:
                # 分割并转换每行的数值
                nums = line.strip().split()
                for num in nums:
                    try:
                        values.append(float(num))
                    except ValueError:
                        pass
            return torch.tensor(values) if values else torch.zeros(1)
        except:
            return torch.zeros(1)

    def _csv_to_tensor(self, data):
        """将CSV数据转换为张量"""
        try:
            # 转换为数值矩阵
            matrix = []
            for row in data:
                row_values = []
                for cell in row:
                    try:
                        row_values.append(float(cell))
                    except ValueError:
                        pass
                if row_values:
                    matrix.append(row_values)
            return torch.tensor(matrix, dtype=torch.float32) if matrix else torch.zeros(1)
        except:
            return torch.zeros(1)

    @classmethod
    def IS_CHANGED(s, file_name, file_type):
        # 文件改变时重新加载
        file_path = folder_paths.get_annotated_filepath(file_name)
        if os.path.exists(file_path):
            mtime = os.path.getmtime(file_path)
            return mtime
        return 0

    @classmethod
    def VALIDATE_INPUTS(s, file_name, file_type):
        # 验证文件是否存在
        if not folder_paths.exists_annotated_filepath(file_name):
            return f"找不到文件: {file_name}"
        return True



class SaveTensor:
    """保存Tensor到文件"""
    
    def __init__(self):
        self.output_dir = folder_paths.get_output_directory()
        self.type = "output"

    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "tensor": (TensorDataType.TENSOR,),
                "filename_prefix": ("STRING", {"default": "ComfyUI", "tooltip": "文件名前缀"}),
                "file_type": (["pickle", "safetensors", "json", "txt", "csv"], 
                             {"default": "pickle", "tooltip": "保存的文件类型"}),
            },
            "optional": {
                "save_prompt": ("BOOLEAN", {"default": False, "label_on": "enabled", "label_off": "disabled", 
                                           "tooltip": "是否保存提示信息"}),
            }
        }

    RETURN_TYPES = ()
    FUNCTION = "save_tensor"
    CATEGORY = "ComfyNN/DataTypes"
    DESCRIPTION = "将Tensor数据保存到文件"
    OUTPUT_NODE = True

    def save_tensor(self, tensor, filename_prefix="ComfyUI", file_type="pickle", save_prompt=False):
        try:
            # 获取保存路径和文件名信息
            full_output_folder, filename, counter, subfolder, filename_prefix = folder_paths.get_save_image_path(
                filename_prefix, self.output_dir, tensor.shape[-1] if tensor.dim() > 0 else 1, 
                tensor.shape[-2] if tensor.dim() > 1 else 1)
            
            # 创建文件名
            file = f"{filename}_{counter:05}_.{file_type}"
            file_path = os.path.join(full_output_folder, file)
            
            # 根据类型保存数据
            if file_type in ["pickle", "pth"]:
                # 保存为pickle格式
                torch.save(tensor, file_path)
                
            elif file_type == "safetensors":
                # 保存为safetensors格式
                from safetensors.torch import save_file
                if isinstance(tensor, dict):
                    save_file(tensor, file_path)
                else:
                    save_file({"tensor": tensor}, file_path)
                    
            elif file_type == "json":
                # 保存为JSON格式
                tensor_data = tensor.detach().cpu().numpy().tolist()
                with open(file_path, 'w') as f:
                    json.dump(tensor_data, f, indent=2)
                    
            elif file_type == "txt":
                # 保存为文本格式
                tensor_data = tensor.detach().cpu().numpy()
                with open(file_path, 'w') as f:
                    if tensor_data.ndim == 1:
                        f.write(' '.join(map(str, tensor_data)))
                    else:
                        # 多维数组按行保存
                        for row in tensor_data:
                            f.write(' '.join(map(str, row.flatten())) + '\n')
                            
            elif file_type == "csv":
                # 保存为CSV格式
                tensor_data = tensor.detach().cpu().numpy()
                with open(file_path, 'w', newline='') as f:
                    writer = csv.writer(f)
                    if tensor_data.ndim == 1:
                        writer.writerow(tensor_data)
                    elif tensor_data.ndim == 2:
                        writer.writerows(tensor_data)
                    else:
                        # 对于更高维度的张量，展平后保存
                        flattened = tensor_data.reshape(-1, tensor_data.shape[-1])
                        writer.writerows(flattened)
            
            print(f"成功保存张量到: {file_path}")
            
        except Exception as e:
            print(f"保存张量文件时出错: {e}")
            raise e
            
        return {}

    @classmethod
    def IS_CHANGED(s, tensor, filename_prefix, file_type, save_prompt=False):
        # 张量改变时重新保存
        return float("NaN")