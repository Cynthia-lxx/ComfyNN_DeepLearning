[中文](README_zh.md)

# ComfyNN_DeepLearning Plugin

A deep learning plugin for ComfyUI based on [d2l-zh (Dive into Deep Learning)](https://zh.d2l.ai/) implementations

## Module Overview

### DLBasic - Deep Learning Basics
Basic tensor operations and activation functions.

### ComputerVision - Computer Vision
Implementations based on d2l-zh computer vision chapters, including image augmentation, fine-tuning, and more.

### DataTypes - Data Types
Data type definitions for deep learning.

### NLP_Pretrain - Natural Language Processing Pretraining
Implementations based on d2l-zh NLP pretraining chapters, including word embeddings, approximate training, and more.

### Visualize - Visualization
Tools for visualizing data and models.

### RNNs - Recurrent Neural Networks
Implementations based on d2l-zh recurrent neural network chapters, including basic RNNs, GRUs, and LSTMs.

## Development Guidelines

1. All node class names must start with "ComfyNN"
2. All node display names must end with " 🐱"
3. Each module must be registered in the root __init__.py
4. Each module needs to provide NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS
5. Important: Each module's __init__.py file must export NODE_CLASS_MAPPINGS and NODE_DISPLAY_NAME_MAPPINGS

## Usage Instructions

Place the plugin folder in ComfyUI's custom_nodes directory to use it.

See example workflows for each module to understand how to use the nodes:
- [DataTypes/example_workflow.json](DataTypes/example_workflow.json)
- [DLBasic/example_workflow.json](DLBasic/example_workflow.json)
- [DLCompute/example_workflow.json](DLCompute/example_workflow.json)
- [ComputerVision/example_workflow.json](ComputerVision/example_workflow.json)
- [NLP_Pretrain/example_workflow.json](NLP_Pretrain/example_workflow.json)
- [RNNs/example_workflow.json](RNNs/example_workflow.json)

## License and Credits

This project is developed and maintained by Cynthia-lxx and maomaowjz_.
It is inspired by the excellent educational resource ["Dive into Deep Learning" (《动手学深度学习》)](https://zh.d2l.ai/).
Some code has been adapted or modified from the [d2l repository](https://github.com/d2l-ai/d2l-zh).

The project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.
