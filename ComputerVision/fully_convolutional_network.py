import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

class FCNNode:
    """Fully Convolutional Network node based on d2l-zh implementation"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "input_tensor": ("TENSOR",),
                "num_classes": ("INT", {"default": 10, "min": 2, "max": 1000}),
                "backbone": (["resnet18", "resnet34", "resnet50"], {"default": "resnet18"}),
            },
            "optional": {
                "upscale_factor": ("INT", {"default": 32, "min": 1, "max": 64}),
                "use_pretrained": ("BOOLEAN", {"default": True}),
            }
        }

    RETURN_TYPES = ("TENSOR", "TENSOR")
    RETURN_NAMES = ("segmentation_output", "feature_output")
    FUNCTION = "fcn_forward"
    CATEGORY = "ComfyNN/ComputerVision/Segmentation"
    DESCRIPTION = "Fully Convolutional Network for semantic segmentation"

    def fcn_forward(self, input_tensor, num_classes, backbone="resnet18", 
                   upscale_factor=32, use_pretrained=True):
        # Ensure input is a tensor
        if not isinstance(input_tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        # Create backbone network
        if backbone == "resnet18":
            backbone_net = self._create_resnet_backbone(18, use_pretrained)
            backbone_channels = 512
        elif backbone == "resnet34":
            backbone_net = self._create_resnet_backbone(34, use_pretrained)
            backbone_channels = 512
        elif backbone == "resnet50":
            backbone_net = self._create_resnet_backbone(50, use_pretrained)
            backbone_channels = 2048
        else:
            raise ValueError(f"Unsupported backbone: {backbone}")
        
        # Create 1x1 convolution to reduce channels to num_classes
        conv_1x1 = nn.Conv2d(backbone_channels, num_classes, kernel_size=1)
        
        # Create transposed convolution for upsampling
        transposed_conv = nn.ConvTranspose2d(
            num_classes, num_classes, 
            kernel_size=upscale_factor*2, 
            stride=upscale_factor, 
            padding=upscale_factor//2,
            groups=num_classes,  # Each class processed separately
            bias=False
        )
        
        # Initialize transposed convolution with bilinear kernel
        bilinear_kernel = self._bilinear_kernel(num_classes, num_classes, upscale_factor*2)
        transposed_conv.weight.data.copy_(bilinear_kernel)
        
        # Forward pass through backbone
        features = backbone_net(input_tensor)
        
        # Apply 1x1 convolution
        reduced_features = conv_1x1(features)
        
        # Apply transposed convolution for upsampling
        output = transposed_conv(reduced_features)
        
        # Crop output to match input size if needed
        if output.shape[2:] != input_tensor.shape[2:]:
            # Center crop to match input size
            diff_h = output.shape[2] - input_tensor.shape[2]
            diff_w = output.shape[3] - input_tensor.shape[3]
            top = diff_h // 2
            bottom = top + input_tensor.shape[2]
            left = diff_w // 2
            right = left + input_tensor.shape[3]
            output = output[:, :, top:bottom, left:right]
        
        return (output, features)

    def _create_resnet_backbone(self, resnet_type, use_pretrained):
        """Create ResNet backbone without final layers"""
        if resnet_type == 18:
            resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet18', pretrained=use_pretrained)
            layers = list(resnet.children())[:-2]  # Remove avgpool and fc
        elif resnet_type == 34:
            resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet34', pretrained=use_pretrained)
            layers = list(resnet.children())[:-2]  # Remove avgpool and fc
        elif resnet_type == 50:
            resnet = torch.hub.load('pytorch/vision:v0.10.0', 'resnet50', pretrained=use_pretrained)
            layers = list(resnet.children())[:-2]  # Remove avgpool and fc
        else:
            raise ValueError(f"Unsupported ResNet type: {resnet_type}")
            
        return nn.Sequential(*layers)

    def _bilinear_kernel(self, in_channels, out_channels, kernel_size):
        """Generate bilinear interpolation kernel"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
            
        og = torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1)
        filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
        
        weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
        for i in range(in_channels):
            for j in range(out_channels):
                if i == j:
                    weight[i, j, :, :] = filt
                    
        return weight

class SegmentationHeadNode:
    """Segmentation Head node for FCN"""
    
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "feature_tensor": ("TENSOR",),
                "num_classes": ("INT", {"default": 10, "min": 2, "max": 1000}),
            },
            "optional": {
                "upscale_factor": ("INT", {"default": 32, "min": 1, "max": 64}),
            }
        }

    RETURN_TYPES = ("TENSOR",)
    RETURN_NAMES = ("segmentation_output",)
    FUNCTION = "segmentation_head"
    CATEGORY = "ComfyNN/ComputerVision/Segmentation"
    DESCRIPTION = "Segmentation head that upsamples feature maps to original image size"

    def segmentation_head(self, feature_tensor, num_classes, upscale_factor=32):
        # Ensure input is a tensor
        if not isinstance(feature_tensor, torch.Tensor):
            raise TypeError("Input must be a torch.Tensor")
        
        # Get number of input channels
        in_channels = feature_tensor.shape[1]
        
        # Create 1x1 convolution to reduce channels to num_classes
        conv_1x1 = nn.Conv2d(in_channels, num_classes, kernel_size=1)
        
        # Create transposed convolution for upsampling
        transposed_conv = nn.ConvTranspose2d(
            num_classes, num_classes, 
            kernel_size=upscale_factor*2, 
            stride=upscale_factor, 
            padding=upscale_factor//2,
            groups=num_classes,  # Each class processed separately
            bias=False
        )
        
        # Initialize transposed convolution with bilinear kernel
        bilinear_kernel = self._bilinear_kernel(num_classes, num_classes, upscale_factor*2)
        transposed_conv.weight.data.copy_(bilinear_kernel)
        
        # Apply 1x1 convolution
        reduced_features = conv_1x1(feature_tensor)
        
        # Apply transposed convolution for upsampling
        output = transposed_conv(reduced_features)
        
        return (output,)

    def _bilinear_kernel(self, in_channels, out_channels, kernel_size):
        """Generate bilinear interpolation kernel"""
        factor = (kernel_size + 1) // 2
        if kernel_size % 2 == 1:
            center = factor - 1
        else:
            center = factor - 0.5
            
        og = torch.arange(kernel_size).reshape(-1, 1), torch.arange(kernel_size).reshape(1, -1)
        filt = (1 - torch.abs(og[0] - center) / factor) * (1 - torch.abs(og[1] - center) / factor)
        
        weight = torch.zeros((in_channels, out_channels, kernel_size, kernel_size))
        for i in range(in_channels):
            for j in range(out_channels):
                if i == j:
                    weight[i, j, :, :] = filt
                    
        return weight

# Node mappings
NODE_CLASS_MAPPINGS = {
    "FCNNode": FCNNode,
    "SegmentationHeadNode": SegmentationHeadNode,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "FCNNode": "Fully Convolutional Network 🐱",
    "SegmentationHeadNode": "Segmentation Head 🐱",
}
