# EDFA 软阈值去噪网络代码

[English](README.md)

关联论文：Hongbo Chen、Guo-Wei Lu、Kunping Luo，*Accurate EDFA Gain Modelling using Convolutional Neural Network with Denoising Layers enabled by Soft-thresholding*，ECOC 2023，282–285 页。[论文 DOI](https://doi.org/10.1049/icp.2023.2060)。

## 本次整理的范围

从原 `EDFA1` 工作目录中提取三种单台 EDFA 模型，保留模型文件原样，另加统一的训练和评估入口。原桌面目录没有修改。

仓库包含采用自适应软阈值去噪的宽通道 CNN、残差注意力 CNN 变体，以及全连接基线。具体结构见 [模型说明](docs/MODELS.md)。

| 模型选项 | 内容 |
| --- | --- |
| `wide-cnn` | 宽通道去噪 CNN，保留 128 通道输入卷积及 128/256/512 通道处理层 |
| `residual-attention-cnn` | 32 通道输入卷积，加 64/128/256 通道残差、软阈值及额外注意力模块 |
| `mlp` | 85 → 256 → 128 → 83 的全连接网络 |

三者输入均为 85 维，输出为 83 维。[原文件与整理版的对应关系](docs/SOURCE_MAP.md)。

## 怎么运行

安装环境和完整命令见 [英文 README](README.md)。统一使用 `python run.py`，支持三个子命令：

- `train`：在指定数据上训练并保存最终模型。
- `cross-validate`：默认五折，每一折重新初始化网络和优化器。
- `evaluate`：加载同一模型评估其他设备，不进行再训练。

运行时指定 CSV 路径、模型、轮数和输出目录；不再依赖本机桌面路径，也不再强制 CUDA。脚本默认不会覆盖已有输出目录。

统一入口使用 Adam、RMSE 形式的训练损失及可配置的固定学习率。训练轮数、批大小和随机种子由命令行设置；模型与运行配置一并保存。数据预处理在训练前完成。

## 数据说明

论文使用的公开数据来自 DTU，见 [原始数据 DOI](https://doi.org/10.11583/DTU.13135754.v1)。仓库不打包大体积原始数据、处理后 CSV、旧权重或论文 PDF，也不更改这些第三方材料的许可。

新入口接受 **169 列处理后 CSV**。原始数据库需要先按统一的频率通道、归一化和单位约定整理为该格式；有表头时添加 `--skip-header`。[数据列说明](data/README.md)。

实验分组中的 15/16/17/18 **dBm 表示输出功率设置**。评估结果使用所提供目标数据的单位。

## 开源范围

使用 MIT 代码许可，沿用作者此前选择。保留论文引用信息和 `CITATION.cff`。级联 EDFA、语义嵌入、迁移学习、重复画图脚本等扩展没有混入主发布目录，原文件仍留在本机。
