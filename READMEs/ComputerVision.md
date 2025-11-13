# ComputerVision Module

The ComputerVision module provides nodes for computer vision tasks, including image augmentation, object detection, segmentation, and style transfer.

## Features

### Image Augmentation
- ImageAugmentationNode: Apply various augmentation techniques to images
- BatchImageAugmentationNode: Apply multiple augmentations to image batches

### Model Fine-tuning
- FinetuningNode: Fine-tune pre-trained models for specific tasks
- TransferLearningNode: Apply transfer learning techniques

### Object Detection Utilities
- BoundingBoxNode: Generate and manipulate bounding boxes
- BoundingBoxMatchingNode: Match bounding boxes with ground truth
- AnchorBoxNode: Generate anchor boxes for object detection
- AnchorBoxMatcher: Match anchor boxes with ground truth
- IoUNode: Calculate Intersection over Union metrics
- IoUThresholdFilter: Filter detections based on IoU thresholds

### Object Detection Models
- SingleShotMultiboxNode: Single Shot Multibox Detector implementation
- SSDAnchorGenerator: Generate anchors for SSD models
- SSDDetectionPostProcessor: Post-process SSD detections
- RCNNModelNode: R-CNN series models (R-CNN, Fast R-CNN, Faster R-CNN, Mask R-CNN)
- RegionProposalNetwork: Region Proposal Network for object detection
- ROIPooling: Region of Interest pooling
- MaskHead: Mask head for instance segmentation

### Segmentation
- SemanticSegmentationNode: Perform semantic segmentation
- InstanceSegmentationNode: Perform instance segmentation
- FCNNode: Fully Convolutional Network for segmentation
- EncoderDecoderNode: Encoder-decoder architecture for segmentation
- TransposedConv2DNode: Transposed convolution for upsampling
- MultiScaleTransposedConvNode: Multi-scale transposed convolution

### Style Transfer
- StyleTransferNode: Neural style transfer
- FastStyleTransferNode: Fast style transfer using pre-trained networks

## Example Workflow

See `example_workflow.json` in the ComputerVision directory for a demonstration of how to use these nodes.