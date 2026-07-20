# Version Notice

- 近期几乎所有 cu12 版本的 PyTorch 容器都是基于 Ubuntu 22.04 构建的，带有 Python 3.11 环境
- Flash-Attn 对 CPython 3.11 上限支持到 PyTorch 2.8
- imgaug 在 6 年前就停止维护了，所以改用 TorchVision 重新实现
- spconv 只支持到 CUDA 126

所以综上所述环境配置：

- Python 3.11
- CUDA 12.6
- PyTorch 2.8.0
- TorchVision 0.23.0
- spconv-cu126
