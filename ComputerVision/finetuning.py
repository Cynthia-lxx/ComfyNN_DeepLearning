# ComfyNN ComputerVision Fine-tuning Nodes
# Based on d2l-zh implementation (https://github.com/d2l-ai/d2l-zh)
# Thank you d2l-ai team for the excellent educational resource

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

class FinetuningNode:
    """模型微调节点，基于d2l-zh实现"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "pretrained_model_name": (["resnet18", "resnet34", "resnet50", "vgg16", "vgg19"], 
                                        {"default": "resnet18"}),
                "num_classes": ("INT", {"default": 10, "min": 2, "max": 1000}),
                "freeze_backbone": ("BOOLEAN", {"default": False}),
            },
            "optional": {
                "learning_rate_multiplier": ("FLOAT", {"default": 0.1, "min": 0.001, "max": 1.0, "step": 0.001}),
                "dropout_rate": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.9, "step": 0.05}),
            }
        }

    RETURN_TYPES = ("CUSTOM", "STRING")
    RETURN_NAMES = ("fine_tuned_model", "finetuning_info")
    FUNCTION = "fine_tune"
    CATEGORY = "ComfyNN/ComputerVision/FineTuning"
    DESCRIPTION = "对预训练模型进行微调，基于d2l-zh实现"

    def fine_tune(self, pretrained_model_name, num_classes, freeze_backbone=False, 
                  learning_rate_multiplier=0.1, dropout_rate=0.5):
        # 加载预训练模型
        if pretrained_model_name == "resnet18":
            model = models.resnet18(pretrained=True)
        elif pretrained_model_name == "resnet34":
            model = models.resnet34(pretrained=True)
        elif pretrained_model_name == "resnet50":
            model = models.resnet50(pretrained=True)
        elif pretrained_model_name == "vgg16":
            model = models.vgg16(pretrained=True)
        elif pretrained_model_name == "vgg19":
            model = models.vgg19(pretrained=True)
        else:
            raise ValueError(f"Unsupported model: {pretrained_model_name}")
        
        # 冻结骨干网络参数（如果需要）
        if freeze_backbone:
            for param in model.parameters():
                param.requires_grad = False
        
        # 修改分类头以适应新的类别数
        if "resnet" in pretrained_model_name:
            # ResNet系列模型
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
            # 如果不冻结骨干网络，确保分类层参数可训练
            for param in model.fc.parameters():
                param.requires_grad = True
                
        elif "vgg" in pretrained_model_name:
            # VGG系列模型
            in_features = model.classifier[6].in_features
            model.classifier[6] = nn.Sequential(
                nn.Dropout(dropout_rate),
                nn.Linear(in_features, num_classes)
            )
            
            # 如果不冻结骨干网络，确保分类层参数可训练
            for param in model.classifier[6].parameters():
                param.requires_grad = True
        
        # 生成微调信息
        finetuning_info = f"Model Fine-tuning Completed\n"
        finetuning_info += f"Base model: {pretrained_model_name}\n"
        finetuning_info += f"Number of classes: {num_classes}\n"
        finetuning_info += f"Backbone frozen: {freeze_backbone}\n"
        finetuning_info += f"Learning rate multiplier: {learning_rate_multiplier}\n"
        finetuning_info += f"Dropout rate: {dropout_rate}\n"
        
        # 计算可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        finetuning_info += f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)"
        
        return (model, finetuning_info)


class TransferLearningNode:
    """迁移学习节点，基于d2l-zh实现"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "source_model": ("CUSTOM",),
                "target_dataset_info": ("STRING", {"default": "CIFAR-10", "multiline": False}),
                "transfer_strategy": (["feature_extractor", "fine_tuning", "adapter"], 
                                    {"default": "feature_extractor"}),
            },
            "optional": {
                "freeze_layers": ("BOOLEAN", {"default": True}),
                "new_layers_dropout": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 0.9, "step": 0.05}),
                "num_target_classes": ("INT", {"default": 10, "min": 2, "max": 1000}),
            }
        }

    RETURN_TYPES = ("CUSTOM", "STRING")
    RETURN_NAMES = ("adapted_model", "transfer_info")
    FUNCTION = "transfer"
    CATEGORY = "ComfyNN/ComputerVision/FineTuning"
    DESCRIPTION = "执行迁移学习，基于d2l-zh实现"

    def transfer(self, source_model, target_dataset_info, transfer_strategy, freeze_layers=True, 
                 new_layers_dropout=0.5, num_target_classes=10):
        # 根据迁移策略调整模型
        model = source_model
        
        if transfer_strategy == "feature_extractor":
            # 使用源模型作为特征提取器
            if freeze_layers:
                # 冻结特征提取层
                for param in model.parameters():
                    param.requires_grad = False
                    
        elif transfer_strategy == "fine_tuning":
            # 微调模型（部分或全部层）
            # 这里我们简单地确保所有参数都可训练
            for param in model.parameters():
                param.requires_grad = True
                
        elif transfer_strategy == "adapter":
            # 使用适配器模式
            # 这里简化处理，实际实现会更复杂
            pass
            
        # 修改分类头以适应目标任务
        if hasattr(model, 'fc'):
            # ResNet系列
            in_features = model.fc.in_features
            model.fc = nn.Sequential(
                nn.Dropout(new_layers_dropout),
                nn.Linear(in_features, num_target_classes)
            )
        elif hasattr(model, 'classifier'):
            # VGG系列
            if isinstance(model.classifier, nn.Sequential):
                in_features = model.classifier[-1].in_features
                model.classifier[-1] = nn.Sequential(
                    nn.Dropout(new_layers_dropout),
                    nn.Linear(in_features, num_target_classes)
                )
            else:
                in_features = model.classifier.in_features
                model.classifier = nn.Sequential(
                    nn.Dropout(new_layers_dropout),
                    nn.Linear(in_features, num_target_classes)
                )
        
        # 确保新添加的层参数可训练
        if transfer_strategy in ["feature_extractor", "adapter"]:
            if hasattr(model, 'fc'):
                for param in model.fc.parameters():
                    param.requires_grad = True
            elif hasattr(model, 'classifier'):
                if isinstance(model.classifier, nn.Sequential):
                    for param in model.classifier[-1].parameters():
                        param.requires_grad = True
                else:
                    for param in model.classifier.parameters():
                        param.requires_grad = True
        
        # 生成迁移学习信息
        transfer_info = f"Transfer Learning Completed\n"
        transfer_info += f"Source model adapted for: {target_dataset_info}\n"
        transfer_info += f"Transfer strategy: {transfer_strategy}\n"
        transfer_info += f"Layers frozen: {freeze_layers}\n"
        transfer_info += f"New layers dropout: {new_layers_dropout}\n"
        transfer_info += f"Target classes: {num_target_classes}\n"
        
        # 计算可训练参数数量
        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        total_params = sum(p.numel() for p in model.parameters())
        transfer_info += f"Trainable parameters: {trainable_params}/{total_params} ({trainable_params/total_params*100:.2f}%)"
        
        return (model, transfer_info)

# Node mappings
NODE_CLASS_MAPPINGS = {
    "FinetuningNode": FinetuningNode,
    "TransferLearningNode": TransferLearningNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FinetuningNode": "Finetuning 🐱",
    "TransferLearningNode": "Transfer Learning 🐱",
}