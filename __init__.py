# ComfyNN_DeepLearning 插件初始化文件
# 遵循UNIX哲学，将不同功能模块化到独立的子文件夹中
#
# 开发规范（重要，务必阅读每一条！）：
# 1. 添加新功能时，请创建新的功能子目录并在其中放置节点代码
# 2. 或者在现有子目录中修改或新建代码
# 3. 所有节点必须在此文件中集中引用和注册
# 4. 遵循"做一件事并做好"的UNIX哲学，确保模块高内聚、低耦合
# 5. 所有的功能性子分类都需要测试数据生成节点和example_workflow
# 6. 所有节点的名字后面都需要有🐱
# 7. 编写任何插件代码，都应该先阅读ComfyUI的源代码以及已经测试稳定的插件代码作为参考
# 8. 每写一个功能，都在/READMEs/编写相应的详细说明，并更新主目录下的README.md和README_zh.md
# 9. 当引用来自别处的代码时，在引用的开头和结尾都应该用注释声明引用来源并简短表达致谢

import os
import sys

# 获取当前目录路径
current_dir = os.path.dirname(__file__)

# 定义模块路径
modules = [
    "DataTypes",
    "DLBasic", 
    "DLCompute",
    "Visualize",
    "NLP_Pretrain",
    "ComputerVision"  # 新增计算机视觉模块
]

# 添加当前目录到sys.path，确保可以正确导入模块
if current_dir not in sys.path:
    sys.path.append(current_dir)

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
        "TensorCreator": "Tensor Creator 🐱",
        "RandomTensorCreator": "Random Tensor Creator 🐱",
        "SpecialTensorCreator": "Special Tensor Creator 🐱",
        "TensorReshaper": "Tensor Reshaper 🐱",
        "TensorConverter": "Tensor Converter 🐱",
        
        # DataTypes节点
        "ImageToTensor": "Image to Tensor 🐱",
        "TensorToImage": "Tensor to Image 🐱",
        "ModelToTensor": "Model to Tensor 🐱",
        "TensorToModel": "Tensor to Model 🐱",
        "ClipToTensor": "CLIP to Tensor 🐱",
        "TensorToClip": "Tensor to CLIP 🐱",
        "VaeToTensor": "VAE to Tensor 🐱",
        "TensorToVae": "Tensor to VAE 🐱",
        "LoadTensor": "Load Tensor 🐱",
        "SaveTensor": "Save Tensor 🐱",
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
        "TensorAdd": "Tensor Add 🐱",
        "TensorSubtract": "Tensor Subtract 🐱",
        "TensorMultiply": "Tensor Multiply 🐱",
        "TensorDivide": "Tensor Divide 🐱",
        "TensorPower": "Tensor Power 🐱",
        "TensorSqrt": "Tensor Sqrt 🐱",
        "TensorTranspose": "Tensor Transpose 🐱",
        "TensorReshape_DLBasic": "Tensor Reshape 🐱",
        "TensorSqueeze": "Tensor Squeeze 🐱",
        "TensorUnsqueeze": "Tensor Unsqueeze 🐱",
        "TensorConcatenate": "Tensor Concatenate 🐱",
        "TensorSum": "Tensor Sum 🐱",
        "TensorMean": "Tensor Mean 🐱",
        "TensorMax": "Tensor Max 🐱",
        "TensorMin": "Tensor Min 🐱",
        "TensorAbs": "Tensor Abs 🐱",
        "TensorSin": "Tensor Sin 🐱",
        "TensorCos": "Tensor Cos 🐱",
        "TensorExp": "Tensor Exp 🐱",
        "TensorLog": "Tensor Log 🐱",
        "TensorReLU": "Tensor ReLU 🐱",
        "TensorLeakyReLU": "Tensor LeakyReLU 🐱",
        "TensorSigmoid": "Tensor Sigmoid 🐱",
        "TensorTanh": "Tensor Tanh 🐱",
        "TensorSoftmax": "Tensor Softmax 🐱",
        "TensorELU": "Tensor ELU 🐱",
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
    
    # 导入DLCompute测试数据生成器
    from .DLCompute.test_data_generator import (
        DLComputeTestDataGenerator
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
        "DLComputeTestDataGenerator": DLComputeTestDataGenerator,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        "TensorDataLoader": "Tensor Data Loader 🐱",
        "TensorNormalize": "Tensor Normalize 🐱",
        "TensorAugment": "Tensor Augment 🐱",
        "TensorLinearLayer": "Tensor Linear Layer 🐱",
        "TensorConv2DLayer": "Tensor Conv2D Layer 🐱",
        "TensorActivation": "Tensor Activation 🐱",
        "TensorMSELoss": "Tensor MSE Loss 🐱",
        "TensorCrossEntropyLoss": "Tensor Cross Entropy Loss 🐱",
        "TensorSGDOptimizer": "Tensor SGD Optimizer 🐱",
        "TensorAdamOptimizer": "Tensor Adam Optimizer 🐱",
        "TensorForwardPass": "Tensor Forward Pass 🐱",
        "TensorBackwardPass": "Tensor Backward Pass 🐱",
        "TensorUpdateWeights": "Tensor Update Weights 🐱",
        "DLComputeTestDataGenerator": "DLCompute Test Data Generator 🐱",
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
        "TensorToImageVisualizer": "Tensor to Image Visualizer 🐱",
        "TensorHeatmapVisualizer": "Tensor Heatmap Visualizer 🐱",
        "TensorShapeVisualizer": "Tensor Shape Visualizer 🐱",
        "TensorLineChartVisualizer": "Tensor Line Chart Visualizer 🐱",
        "TensorStatisticsVisualizer": "Tensor Statistics Visualizer 🐱",
    })
    
except ImportError as e:
    print(f"Error importing Visualize nodes: {e}")

# 新增NLP预训练模块
try:
    # 导入NLP预训练模块 - 词嵌入相关节点
    from .NLP_Pretrain.word_embeddings import (
        Word2VecSelfSupervised,
        SkipGramModel,
        CBOWModel,
        SubsamplingNLP
    )
    
    # 导入NLP预训练模块 - 近似训练相关节点
    from .NLP_Pretrain.approximate_training import (
        NegativeSamplingNode,
        HierarchicalSoftmaxNode
    )
    
    # 导入NLP预训练模块 - GloVe相关节点
    from .NLP_Pretrain.glove import (
        GloVeNode
    )
    
    # 导入NLP预训练模块 - FastText相关节点
    from .NLP_Pretrain.fasttext import (
        FastTextModel
    )
    
    # 导入NLP预训练模块 - BERT相关节点
    from .NLP_Pretrain.bert import (
        BERTModel,
        BERTMaskedLanguageModel
    )
    
    # 导入NLP预训练模块 - 测试数据生成器
    from .NLP_Pretrain.test_data_generator import (
        NLPTestDataGenerator
    )
    
    # 更新节点映射
    NODE_CLASS_MAPPINGS.update({
        # 词嵌入相关节点
        "Word2VecSelfSupervised": Word2VecSelfSupervised,
        "SkipGramModel": SkipGramModel,
        "CBOWModel": CBOWModel,
        "SubsamplingNLP": SubsamplingNLP,
        
        # 近似训练相关节点
        "NegativeSamplingNode": NegativeSamplingNode,
        "HierarchicalSoftmaxNode": HierarchicalSoftmaxNode,
        
        # GloVe相关节点
        "GloVeNode": GloVeNode,
        
        # FastText相关节点
        "FastTextModel": FastTextModel,
        
        # BERT相关节点
        "BERTModel": BERTModel,
        "BERTMaskedLanguageModel": BERTMaskedLanguageModel,
        
        # 测试数据生成器
        "NLPTestDataGenerator": NLPTestDataGenerator,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        # 词嵌入相关节点
        "Word2VecSelfSupervised": "Word2Vec Self-Supervised 🐱",
        "SkipGramModel": "Skip-Gram Model 🐱",
        "CBOWModel": "CBOW Model 🐱",
        "SubsamplingNLP": "Subsampling NLP 🐱",
        
        # 近似训练相关节点
        "NegativeSamplingNode": "Negative Sampling 🐱",
        "HierarchicalSoftmaxNode": "Hierarchical Softmax 🐱",
        
        # GloVe相关节点
        "GloVeNode": "GloVe Embeddings 🐱",
        
        # FastText相关节点
        "FastTextModel": "FastText Model 🐱",
        
        # BERT相关节点
        "BERTModel": "BERT Model 🐱",
        "BERTMaskedLanguageModel": "BERT Masked Language Model 🐱",
        
        # 测试数据生成器
        "NLPTestDataGenerator": "NLP Test Data Generator 🐱",
    })
    
except ImportError as e:
    print(f"Error importing NLP_Pretrain nodes: {e}")

# 新增计算机视觉模块
try:
    # 导入计算机视觉模块 - 图像增广相关节点
    from .ComputerVision.image_augmentation import (
        ImageAugmentationNode,
        BatchImageAugmentationNode
    )
    
    # 导入计算机视觉模块 - 微调相关节点
    from .ComputerVision.finetuning import (
        FinetuningNode,
        TransferLearningNode
    )
    
    # 导入计算机视觉模块 - 边界框相关节点
    from .ComputerVision.bounding_boxes import (
        BoundingBoxNode,
        BoundingBoxMatchingNode
    )
    
    # 导入计算机视觉模块 - 锚框相关节点
    from .ComputerVision.anchor_boxes import (
        AnchorBoxNode,
        AnchorBoxMatcher
    )
    
    # 导入计算机视觉模块 - IoU相关节点
    from .ComputerVision.iou import (
        IoUNode,
        IoUThresholdFilter
    )
    
    # 导入计算机视觉模块 - 单发多框检测相关节点
    from .ComputerVision.single_shot_multibox import (
        SingleShotMultiboxNode,
        SSDAnchorGenerator,
        SSDDetectionPostProcessor
    )
    
    # 导入计算机视觉模块 - R-CNN系列相关节点
    from .ComputerVision.rcnn_series import (
        RCNNModelNode,
        RegionProposalNetwork,
        ROIPooling,
        MaskHead
    )
    
    # 导入计算机视觉模块 - 语义分割相关节点
    from .ComputerVision.semantic_segmentation import (
        SemanticSegmentationNode,
        InstanceSegmentationNode
    )
    
    # 导入计算机视觉模块 - 转置卷积相关节点
    from .ComputerVision.transposed_convolution import (
        TransposedConv2DNode,
        BilinearUpsampleNode
    )
    
    # 导入计算机视觉模块 - 全卷积网络相关节点
    from .ComputerVision.fully_convolutional_network import (
        FCNNode,
        SegmentationHeadNode
    )
    
    # 导入计算机视觉模块 - 风格迁移相关节点
    from .ComputerVision.style_transfer import (
        StyleTransferNode,
        FastStyleTransferNode
    )
    
    # 更新节点映射
    NODE_CLASS_MAPPINGS.update({
        # 图像增广相关节点
        "ImageAugmentationNode": ImageAugmentationNode,
        "BatchImageAugmentationNode": BatchImageAugmentationNode,
        
        # 微调相关节点
        "FinetuningNode": FinetuningNode,
        "TransferLearningNode": TransferLearningNode,
        
        # 边界框相关节点
        "BoundingBoxNode": BoundingBoxNode,
        "BoundingBoxMatchingNode": BoundingBoxMatchingNode,
        
        # 锚框相关节点
        "AnchorBoxNode": AnchorBoxNode,
        "AnchorBoxMatcher": AnchorBoxMatcher,
        
        # IoU相关节点
        "IoUNode": IoUNode,
        "IoUThresholdFilter": IoUThresholdFilter,
        
        # 单发多框检测相关节点
        "SingleShotMultiboxNode": SingleShotMultiboxNode,
        "SSDAnchorGenerator": SSDAnchorGenerator,
        "SSDDetectionPostProcessor": SSDDetectionPostProcessor,
        
        # R-CNN系列相关节点
        "RCNNModelNode": RCNNModelNode,
        "RegionProposalNetwork": RegionProposalNetwork,
        "ROIPooling": ROIPooling,
        "MaskHead": MaskHead,
        
        # 语义分割相关节点
        "SemanticSegmentationNode": SemanticSegmentationNode,
        "InstanceSegmentationNode": InstanceSegmentationNode,
        
        # 转置卷积相关节点
        "TransposedConv2DNode": TransposedConv2DNode,
        "BilinearUpsampleNode": BilinearUpsampleNode,
        
        # 全卷积网络相关节点
        "FCNNode": FCNNode,
        "SegmentationHeadNode": SegmentationHeadNode,
        
        # 风格迁移相关节点
        "StyleTransferNode": StyleTransferNode,
        "FastStyleTransferNode": FastStyleTransferNode,
    })
    
    NODE_DISPLAY_NAME_MAPPINGS.update({
        # 图像增广相关节点
        "ImageAugmentationNode": "Image Augmentation 🐱",
        "BatchImageAugmentationNode": "Batch Image Augmentation 🐱",
        
        # 微调相关节点
        "FinetuningNode": "Finetuning 🐱",
        "TransferLearningNode": "Transfer Learning 🐱",
        
        # 边界框相关节点
        "BoundingBoxNode": "Bounding Box 🐱",
        "BoundingBoxMatchingNode": "Bounding Box Matching 🐱",
        
        # 锚框相关节点
        "AnchorBoxNode": "Anchor Box 🐱",
        "AnchorBoxMatcher": "Anchor Box Matcher 🐱",
        
        # IoU相关节点
        "IoUNode": "IoU 🐱",
        "IoUThresholdFilter": "IoU Threshold Filter 🐱",
        
        # 单发多框检测相关节点
        "SingleShotMultiboxNode": "Single Shot Multibox 🐱",
        "SSDAnchorGenerator": "SSD Anchor Generator 🐱",
        "SSDDetectionPostProcessor": "SSD Detection Post Processor 🐱",
        
        # R-CNN系列相关节点
        "RCNNModelNode": "R-CNN Model 🐱",
        "RegionProposalNetwork": "Region Proposal Network 🐱",
        "ROIPooling": "ROI Pooling 🐱",
        "MaskHead": "Mask Head 🐱",
        
        # 语义分割相关节点
        "SemanticSegmentationNode": "Semantic Segmentation 🐱",
        "InstanceSegmentationNode": "Instance Segmentation 🐱",
        
        # 转置卷积相关节点
        "TransposedConv2DNode": "Transposed Conv2D 🐱",
        "BilinearUpsampleNode": "Bilinear Upsample 🐱",
        
        # 全卷积网络相关节点
        "FCNNode": "Fully Convolutional Network 🐱",
        "SegmentationHeadNode": "Segmentation Head 🐱",
        
        # 风格迁移相关节点
        "StyleTransferNode": "Neural Style Transfer 🐱",
        "FastStyleTransferNode": "Fast Style Transfer 🐱",
    })
    
except ImportError as e:
    print(f"Error importing ComputerVision nodes: {e}")

# 定义要导出的类
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']