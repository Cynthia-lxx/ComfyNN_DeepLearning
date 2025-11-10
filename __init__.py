# ComfyNN_DeepLearning 插件初始化文件
# 遵循UNIX哲学，将不同功能模块化到独立的子文件夹中
#
# 开发规范：
# 1. 添加新功能时，请创建新的功能子目录并在其中放置节点代码
# 2. 或者在现有子目录中修改或新建代码
# 3. 所有节点必须在此文件中集中引用和注册
# 4. 遵循"做一件事并做好"的UNIX哲学，确保模块高内聚、低耦合

import os
import sys

# 获取当前目录路径
current_dir = os.path.dirname(__file__)

# 定义模块路径
modules = [
    "DataTypes",
    "DLBasic", 
    "DLCompute",
    "Visualize"
]

# 添加模块路径到sys.path
for module in modules:
    module_path = os.path.join(current_dir, module)
    if os.path.exists(module_path) and module_path not in sys.path:
        sys.path.append(module_path)

# 节点映射
NODE_CLASS_MAPPINGS = {}
NODE_DISPLAY_NAME_MAPPINGS = {}

# 导入各个模块的节点
try:
    # 导入DataTypes模块
    from .DataTypes.nodes_Creation import (
        TensorCreator,
        RandomTensorCreator,
        SpecialTensorCreator,
        TensorReshaper,
        TensorConverter
    )
    
    from .DataTypes.nodes_DataTypes import (
        ImageToTensor,
        TensorToImage,
        ModelToTensor,
        TensorToModel,
        ClipToTensor,
        TensorToClip,
        VaeToTensor,
        TensorToVae,
        LoadTensor,
        SaveTensor
    )
    
    # 更新节点映射
    NODE_CLASS_MAPPINGS.update({
        # DataTypes Creation节点
        "TensorCreator": TensorCreator,
        "RandomTensorCreator": RandomTensorCreator,
        "SpecialTensorCreator": SpecialTensorCreator,
        "TensorReshaper": TensorReshaper,
        "TensorConverter": TensorConverter,
        
        # DataTypes节点
        "ImageToTensor": ImageToTensor,
        "TensorToImage": TensorToImage,
        "ModelToTensor": ModelToTensor,
        "TensorToModel": TensorToModel,
        "ClipToTensor": ClipToTensor,
        "TensorToClip": TensorToClip,
        "VaeToTensor": VaeToTensor,
        "TensorToVae": TensorToVae,
        "LoadTensor": LoadTensor,
        "SaveTensor": SaveTensor,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        # DataTypes Creation节点
        "TensorCreator": "Tensor Creator",
        "RandomTensorCreator": "Random Tensor Creator",
        "SpecialTensorCreator": "Special Tensor Creator",
        "TensorReshaper": "Tensor Reshaper",
        "TensorConverter": "Tensor Converter",
        
        # DataTypes节点
        "ImageToTensor": "Image to Tensor",
        "TensorToImage": "Tensor to Image",
        "ModelToTensor": "Model to Tensor",
        "TensorToModel": "Tensor to Model",
        "ClipToTensor": "CLIP to Tensor",
        "TensorToClip": "Tensor to CLIP",
        "VaeToTensor": "VAE to Tensor",
        "TensorToVae": "Tensor to VAE",
        "LoadTensor": "Load Tensor",
        "SaveTensor": "Save Tensor",
    })
    
except ImportError as e:
    print(f"Error importing DataTypes nodes: {e}")

try:
    # 导入DLBasic模块
    from .DLBasic.nodes import (
        TensorAdd,
        TensorSubtract,
        TensorMultiply,
        TensorDivide,
        TensorPower,
        TensorSqrt,
        TensorTranspose,
        TensorReshape,
        TensorSqueeze,
        TensorUnsqueeze,
        TensorConcatenate,
        TensorSum,
        TensorMean,
        TensorMax,
        TensorMin,
        TensorAbs,
        TensorSin,
        TensorCos,
        TensorExp,
        TensorLog,
        TensorReLU,
        TensorLeakyReLU,
        TensorSigmoid,
        TensorTanh,
        TensorSoftmax,
        TensorELU
    )
    
    # 更新节点映射
    NODE_CLASS_MAPPINGS.update({
        "TensorAdd": TensorAdd,
        "TensorSubtract": TensorSubtract,
        "TensorMultiply": TensorMultiply,
        "TensorDivide": TensorDivide,
        "TensorPower": TensorPower,
        "TensorSqrt": TensorSqrt,
        "TensorTranspose": TensorTranspose,
        "TensorReshape_DLBasic": TensorReshape,
        "TensorSqueeze": TensorSqueeze,
        "TensorUnsqueeze": TensorUnsqueeze,
        "TensorConcatenate": TensorConcatenate,
        "TensorSum": TensorSum,
        "TensorMean": TensorMean,
        "TensorMax": TensorMax,
        "TensorMin": TensorMin,
        "TensorAbs": TensorAbs,
        "TensorSin": TensorSin,
        "TensorCos": TensorCos,
        "TensorExp": TensorExp,
        "TensorLog": TensorLog,
        "TensorReLU": TensorReLU,
        "TensorLeakyReLU": TensorLeakyReLU,
        "TensorSigmoid": TensorSigmoid,
        "TensorTanh": TensorTanh,
        "TensorSoftmax": TensorSoftmax,
        "TensorELU": TensorELU,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "TensorAdd": "Tensor Add",
        "TensorSubtract": "Tensor Subtract",
        "TensorMultiply": "Tensor Multiply",
        "TensorDivide": "Tensor Divide",
        "TensorPower": "Tensor Power",
        "TensorSqrt": "Tensor Sqrt",
        "TensorTranspose": "Tensor Transpose",
        "TensorReshape_DLBasic": "Tensor Reshape",
        "TensorSqueeze": "Tensor Squeeze",
        "TensorUnsqueeze": "Tensor Unsqueeze",
        "TensorConcatenate": "Tensor Concatenate",
        "TensorSum": "Tensor Sum",
        "TensorMean": "Tensor Mean",
        "TensorMax": "Tensor Max",
        "TensorMin": "Tensor Min",
        "TensorAbs": "Tensor Abs",
        "TensorSin": "Tensor Sin",
        "TensorCos": "Tensor Cos",
        "TensorExp": "Tensor Exp",
        "TensorLog": "Tensor Log",
        "TensorReLU": "Tensor ReLU",
        "TensorLeakyReLU": "Tensor LeakyReLU",
        "TensorSigmoid": "Tensor Sigmoid",
        "TensorTanh": "Tensor Tanh",
        "TensorSoftmax": "Tensor Softmax",
        "TensorELU": "Tensor ELU",
    })
    
except ImportError as e:
    print(f"Error importing DLBasic nodes: {e}")

try:
    # 导入DLCompute模块
    from .DLCompute.nodes import (
        TensorDataLoader,
        TensorNormalize,
        TensorAugment,
        TensorLinearLayer,
        TensorConv2DLayer,
        TensorActivation,
        TensorMSELoss,
        TensorCrossEntropyLoss,
        TensorSGDOptimizer,
        TensorAdamOptimizer,
        TensorForwardPass,
        TensorBackwardPass,
        TensorUpdateWeights
    )
    
    # 更新节点映射
    NODE_CLASS_MAPPINGS.update({
        "TensorDataLoader": TensorDataLoader,
        "TensorNormalize": TensorNormalize,
        "TensorAugment": TensorAugment,
        "TensorLinearLayer": TensorLinearLayer,
        "TensorConv2DLayer": TensorConv2DLayer,
        "TensorActivation": TensorActivation,
        "TensorMSELoss": TensorMSELoss,
        "TensorCrossEntropyLoss": TensorCrossEntropyLoss,
        "TensorSGDOptimizer": TensorSGDOptimizer,
        "TensorAdamOptimizer": TensorAdamOptimizer,
        "TensorForwardPass": TensorForwardPass,
        "TensorBackwardPass": TensorBackwardPass,
        "TensorUpdateWeights": TensorUpdateWeights,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "TensorDataLoader": "Tensor Data Loader",
        "TensorNormalize": "Tensor Normalize",
        "TensorAugment": "Tensor Augment",
        "TensorLinearLayer": "Tensor Linear Layer",
        "TensorConv2DLayer": "Tensor Conv2D Layer",
        "TensorActivation": "Tensor Activation",
        "TensorMSELoss": "Tensor MSE Loss",
        "TensorCrossEntropyLoss": "Tensor Cross Entropy Loss",
        "TensorSGDOptimizer": "Tensor SGD Optimizer",
        "TensorAdamOptimizer": "Tensor Adam Optimizer",
        "TensorForwardPass": "Tensor Forward Pass",
        "TensorBackwardPass": "Tensor Backward Pass",
        "TensorUpdateWeights": "Tensor Update Weights",
    })
    
except ImportError as e:
    print(f"Error importing DLCompute nodes: {e}")

try:
    # 导入Visualize模块
    from .Visualize.nodes import (
        TensorToImageVisualizer,
        TensorHeatmapVisualizer,
        TensorShapeVisualizer,
        TensorLineChartVisualizer,
        TensorStatisticsVisualizer
    )
    
    # 更新节点映射
    NODE_CLASS_MAPPINGS.update({
        "TensorToImageVisualizer": TensorToImageVisualizer,
        "TensorHeatmapVisualizer": TensorHeatmapVisualizer,
        "TensorShapeVisualizer": TensorShapeVisualizer,
        "TensorLineChartVisualizer": TensorLineChartVisualizer,
        "TensorStatisticsVisualizer": TensorStatisticsVisualizer,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "TensorToImageVisualizer": "Tensor to Image Visualizer",
        "TensorHeatmapVisualizer": "Tensor Heatmap Visualizer",
        "TensorShapeVisualizer": "Tensor Shape Visualizer",
        "TensorLineChartVisualizer": "Tensor Line Chart Visualizer",
        "TensorStatisticsVisualizer": "Tensor Statistics Visualizer",
    })
    
except ImportError as e:
    print(f"Error importing Visualize nodes: {e}")

# 定义要导出的类
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
