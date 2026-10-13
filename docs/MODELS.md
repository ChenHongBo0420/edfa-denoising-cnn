# Models and Workflow

All three implementations accept 85 input features and produce 83 spectral targets. Select a model using the `--model` option.

| Option | Active architecture |
| --- | --- |
| `wide-cnn` | 1-to-128 input convolution; three convolution/BatchNorm/soft-threshold stages with 128, 256 and 512 channels; flatten 5632 features; Linear output to 83 targets |
| `residual-attention-cnn` | 1-to-32 input convolution; 64/128/256-channel residual stages with soft thresholding, additional attention and GELU; flatten 2816 features; Linear output to 83 targets |
| `mlp` | Linear 85-to-256, BatchNorm and ReLU; Linear 256-to-128, BatchNorm and ReLU; Linear 128-to-83 |

The CNN implementations provide wide-channel and residual-attention variants of adaptive soft-threshold denoising. Their model files are preserved from the source paths documented in [SOURCE_MAP.md](SOURCE_MAP.md).

## Adaptive Soft Thresholding

For feature tensor `x`, the denoising layer computes:

```text
output = sign(x) * max(abs(x) - threshold, 0)
```

The threshold is learned from pooled absolute feature values. A fully connected branch with a sigmoid produces channel coefficients, which are multiplied by a global magnitude statistic. The resulting threshold is broadcast along the spectral dimension.

## Training and Evaluation

`run.py train` trains the selected model on the supplied prepared CSV. `run.py cross-validate` initializes a separate model and optimizer for each fold and saves the row indices. `run.py evaluate` applies a saved model to another prepared dataset.

The runner uses Adam, a configurable constant learning rate and `sqrt(MSE + 1e-6)` for training. Final-epoch checkpoints include the model name and configuration. Evaluation aggregates RMSE and MAE over all samples and output channels.

## 中文说明

三个模型均使用 85 维输入和 83 维输出。`wide-cnn` 为宽通道软阈值去噪网络，`residual-attention-cnn` 增加残差及注意力模块，`mlp` 为全连接基线。

软阈值层由特征绝对值、全局池化、全连接分支和 Sigmoid 学习阈值，再对特征进行收缩。统一入口支持单次训练、独立五折评估，以及同一模型在不同 EDFA 数据上的预测。
