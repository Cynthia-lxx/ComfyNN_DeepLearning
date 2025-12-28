<div align="center">
  <a href="README_zh.md">中文</a>
  <br><br>
  <img src="icon.png" alt="Icon" style="width: max(10vw, 200px);">
  <h1>ComfyNN_DeepLearning</h1>
  <p>A set of custom nodes that integrates deeplearning to ComfyUI!</p>
</div>

## To Install

Directly copy the source code folder to `ComfyUI/custom_nodes/ComfyNN_DeepLearning` (make sure that the __init__.py and this README file can be found in this directory).  
Or, if you are using Git, you can clone this repository.

## Workflow Examples

In each sub-directory there is a `example_workflow.json` which shows the basic usage of the nodes.

## Function overview
**Data Types**
- Defines the `ComfyNN_DeepLearning.TensorDatatype.TENSOR` data type for tensors.
- Reads tensors from files; save or export tensors.
- Basic tensor operations(reshape, calculation, etc).  

**Deep Learning Basic**
- Basic tensor operations and activation functions.

**Deep Learning Computation**
- Layers: Conv2d, Linear, etc.
- Optimizers: SGD, Adam, etc.
- Loss functions: CrossEntropyLoss, MSELoss, etc.

**Computer Vision**
- Implementations based on d2l-zh computer vision chapters, including image augmentation, fine-tuning, and more.

**NLP Pretrain**
- NLP Models: GloVe, BERT, etc.

**Visualize**
- Tensor visualization as images
- Heatmap generation for tensor data
- Shape information display
- Statistical visualization of tensor data

**RNNs**
- Implementations based on d2l-zh recurrent neural network chapters, including basic RNns, GRUs, and LSTMs.

## License and Credits

This project is developed and maintained by Cynthia-lxx and maomaowjz_.
It is inspired by the excellent educational resource ["Dive into Deep Learning" (《动手学深度学习》)](https://zh.d2l.ai/).
Some code has been adapted or modified from the [d2l repository](https://github.com/d2l-ai/d2l-zh).

The project is licensed under the MIT License. See the [LICENSE](LICENSE) file for details.

## Send Feedback

If you have encountered any problems or have suggestions, please [open an issue](https://github.com/Cynthia-lxx/ComfyNN_DeepLearning/issues) or [start a pull request](https://github.com/Cynthia-lxx/ComfyNN_DeepLearning/pulls).