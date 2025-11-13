[中文](README_zh.md)

# ComfyNN_DeepLearning

ComfyNN_DeepLearning is a modular deep learning plugin suite designed for ComfyUI. Following the UNIX philosophy, this plugin divides different functionalities into independent submodules for easier maintenance and extension.

## Features

This plugin does not depend on any other plugins and is not included in ComfyUI Manager. Currently, it does not require any models.

## Module Structure

- **DataTypes**: Data type conversion and tensor creation functionality
  - See [DataTypes README](READMEs/DataTypes.md) for details

- **DLBasic**: Basic tensor operations
  - See [DLBasic README](READMEs/DLBasic.md) for details

- **DLCompute**: Deep learning computation related nodes
  - See [DLCompute README](READMEs/DLCompute.md) for details

- **Visualize**: Data visualization functionality
  - See [Visualize README](READMEs/Visualize.md) for details

- **NLP_Pretrain**: Natural language processing pre-training related nodes
  - See [NLP_Pretrain README](READMEs/NLP_Pretrain.md) for details

- **ComputerVision**: Computer vision related nodes
  - See [ComputerVision README](READMEs/ComputerVision.md) for details

## Examples

Each module contains example workflows that demonstrate the usage of the nodes in that category. Look for the `example_workflow.json` files in each subdirectory.

## Installation

Clone this repository to your ComfyUI `custom_nodes` directory:

```
cd ComfyUI/custom_nodes
git clone https://github.com/Cynthia-lxx/ComfyNN_DeepLearning.git
```

After installation, restart ComfyUI to load the plugin.

## Usage

After installation and restarting ComfyUI, all nodes will automatically appear in the node list, organized by module category.

## Development

Each module is independent and can be developed and modified in its corresponding subdirectory without affecting other modules.

Each module follows the principle of "Do One Thing and Do It Well" and maintains high cohesion and low coupling.

## GitHub Repository

For issues, suggestions, or contributions, please visit our GitHub repository: [https://github.com/Cynthia-lxx/ComfyNN_DeepLearning](https://github.com/Cynthia-lxx/ComfyNN_DeepLearning)

## License

[To be determined]