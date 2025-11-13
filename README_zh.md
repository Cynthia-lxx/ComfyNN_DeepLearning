# ComfyNN_DeepLearning 插件

基于 [d2l-zh (动手学深度学习)](https://zh.d2l.ai/) 实现的 ComfyUI 深度学习插件

## 模块介绍

### DLBasic - 深度学习基础
基础的张量操作和激活函数节点。

### ComputerVision - 计算机视觉
基于d2l-zh计算机视觉章节的实现，包括图像增强、微调等技术。

### DataTypes - 数据类型
用于深度学习的数据类型定义。

### NLP_Pretrain - 自然语言处理预训练
基于d2l-zh自然语言处理预训练章节的实现，包括词嵌入、近似训练等技术。

### Visualize - 可视化
数据和模型的可视化工具。

### RNNs - 循环神经网络
基于d2l-zh循环神经网络章节的实现，包括基础RNN、GRU和LSTM。

## 开发规范

1. 所有节点类名必须以"ComfyNN"开头
2. 所有节点显示名称必须以" 🐱"结尾
3. 每个模块必须在根目录的__init__.py中注册
4. 每个模块需要提供NODE_CLASS_MAPPINGS和NODE_DISPLAY_NAME_MAPPINGS
5. 重要：所有模块的__init__.py文件必须导出NODE_CLASS_MAPPINGS和NODE_DISPLAY_NAME_MAPPINGS

## 使用说明

将插件文件夹放入ComfyUI的custom_nodes目录中即可使用。

查看各模块的示例工作流以了解如何使用节点：
- [DataTypes/example_workflow.json](DataTypes/example_workflow.json)
- [DLBasic/example_workflow.json](DLBasic/example_workflow.json)
- [DLCompute/example_workflow.json](DLCompute/example_workflow.json)
- [ComputerVision/example_workflow.json](ComputerVision/example_workflow.json)
- [NLP_Pretrain/example_workflow.json](NLP_Pretrain/example_workflow.json)
- [RNNs/example_workflow.json](RNNs/example_workflow.json)

## 许可证和致谢

本项目由 Cynthia-lxx 和 maomaowjz_ 策划和开发。
灵感来源于优秀的教育资源[《动手学深度学习》](https://zh.d2l.ai/)。
部分代码复用或改装自[d2l仓库](https://github.com/d2l-ai/d2l-zh)。

项目采用 MIT 许可证，详见 [LICENSE](LICENSE) 文件。