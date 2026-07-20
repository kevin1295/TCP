"""
v3: TCP + 双路 PointNet + 门控融合 (TODO)
- 点云做标定分解：
  1. 与图像重合的点 → PointNet → 门控融合与 ResNet34 特征
  2. 不重合的点 → 另一路 PointNet → 拼接

具体实现待第三代模型代码到位后补充。
"""

import torch
from torch import nn

from TCP.config import GlobalConfig as _GlobalConfig

# 临时 PIDController 占位（与 v1/v2 相同）
from TCP.models.tcp_simplepointnet import PIDController


class TCP(nn.Module):
    """
    第三代模型占位符。
    接收与 v2 相同的 forward 签名 (img, state, target_point, pcl)，
    待代码到位后替换实现。
    """

    def __init__(self, config, lidar_dim=256):
        super().__init__()
        raise NotImplementedError(
            "TCP_MODEL=tcp_v3 尚未实现。"
            "请将第三代模型代码放入 TCP/models/tcp_v3.py 后重试。"
        )

    def forward(self, img, state, target_point, pcl):
        raise NotImplementedError("tcp_v3 未实现")
