[English](README.md)

# ComfyNN_DeepLearning

ComfyNN_DeepLearning 是一个模块化的深度学习插件集合，专为 ComfyUI 设计。该插件遵循 UNIX 哲学，将不同功能划分为独立的子模块，便于维护和扩展。

## 功能特点

此插件不依赖任何其他插件，也不被 ComfyUI Manager 收录。目前不需要任何模型。

## 模块结构

- **DataTypes**: 数据类型转换和张量创建功能
  - 详见 [DataTypes README](READMEs/DataTypes.md)

- **DLBasic**: 基础张量操作
  - 详见 [DLBasic README](READMEs/DLBasic.md)

- **DLCompute**: 深度学习计算相关节点
  - 详见 [DLCompute README](READMEs/DLCompute.md)

- **Visualize**: 数据可视化功能
  - 详见 [Visualize README](READMEs/Visualize.md)

- **NLP_Pretrain**: 自然语言处理预训练相关节点
  - 详见 [NLP_Pretrain README](READMEs/NLP_Pretrain.md)

- **ComputerVision**: 计算机视觉相关节点
  - 详见 [ComputerVision README](READMEs/ComputerVision.md)

## 示例

每个模块都包含示例工作流，演示该类别节点的用法。请在每个子目录中查找 `example_workflow.json` 文件。

## 安装

将此仓库克隆到您的 ComfyUI `custom_nodes` 目录中：

```
cd ComfyUI/custom_nodes
git clone https://github.com/Cynthia-lxx/ComfyNN_DeepLearning.git
```

安装完成后，重启 ComfyUI 以加载插件。

## 使用

安装并重启 ComfyUI 后，所有节点将自动出现在节点列表中，按模块分类组织。

## 开发

每个模块都是独立的，可以在对应的子目录中进行开发和修改，不会影响其他模块。

每个模块都遵循"做一件事并做好"的原则，保持高内聚、低耦合。

## GitHub 仓库

如有问题、建议或贡献，请访问我们的 GitHub 仓库：[https://github.com/Cynthia-lxx/ComfyNN_DeepLearning](https://github.com/Cynthia-lxx/ComfyNN_DeepLearning)

## 许可证

[待定]