<div align="center">
  <a href="README.md">English</a>
  <br><br>
  <img src="icon.png" alt="Icon" style="width: max(10vw, 200px);">
  <h1>ComfyNN_DeepLearning</h1>
  <p>一套将深度学习集成到 ComfyUI 中的自定义节点！</p>
</div>

## 安装方式

将源代码文件夹直接复制到 `ComfyUI/custom_nodes/ComfyNN_DeepLearning` 目录下（确保此目录中可以找到[__init__.py](file://E:\Dev\ComfyNN_v0\app\__init__.py)和本 README 文件）。
或者，如果您使用 Git，可以克隆此存储库。

## 工作流示例

在每个子目录中都有一个 `example_workflow.json` 文件，展示了节点的基本用法。

## 功能概述
**数据类型**
- 定义 `ComfyNN_DeepLearning.TensorDatatype.TENSOR` 张量数据类型
- 从文件读取张量；保存或导出张量
- 基本张量操作（重塑、计算等）

**深度学习基础**
- 基本张量操作和激活函数

**深度学习计算**
- 层：Conv2d、Linear 等
- 优化器：SGD、Adam 等
- 损失函数：CrossEntropyLoss、MSELoss 等

**计算机视觉**
- 基于 d2l-zh 计算机视觉章节的实现，包括图像增强、微调等

**自然语言处理预训练**
- NLP 模型：GloVe、BERT 等

**可视化**
- 将张量可视化为图像
- 为张量数据生成热力图
- 显示形状信息
- 张量数据的统计可视化

**循环神经网络**
- 基于 d2l-zh 循环神经网络章节的实现，包括基础 RNN、GRU 和 LSTM

## 许可证和致谢

本项目由 Cynthia-lxx 和 maomaowjz_ 开发和维护。
它受到优秀教育资源[《动手学深度学习》](https://zh.d2l.ai/)的启发。
部分代码改编或修改自 [d2l 仓库](https://github.com/d2l-ai/d2l-zh)。

该项目根据 MIT 许可证授权。有关详细信息，请参阅 [LICENSE](LICENSE) 文件。

## 反馈

如果您遇到任何问题或有建议，请[提交问题](https://github.com/Cynthia-lxx/ComfyNN_DeepLearning/issues)或[发起拉取请求](https://github.com/Cynthia-lxx/ComfyNN_DeepLearning/pulls)。