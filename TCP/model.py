"""
TCP 模型 Dispatcher

通过环境变量 TCP_MODEL 切换模型变体：
  TCP_MODEL=tcp_original        v1: 原始 TCP（无 LiDAR）
  TCP_MODEL=tcp_simplepointnet  v2: TCP + SimplePointNet（默认）
  TCP_MODEL=tcp_v3              v3: 双路 PointNet + 视觉门控融合（2026-10 补实现，未训练/未验证）

用法：
  export TCP_MODEL=tcp_original
  python TCP/train.py --gpus 1

  # 或单次覆盖：
  TCP_MODEL=tcp_original python scripts/export_openvino.py --ckpt ...
"""

import os
import importlib

# 模型注册表：名称 → 模块路径
_MODEL_REGISTRY = {
    "tcp_original": "TCP.models.tcp_original",
    "tcp_simplepointnet": "TCP.models.tcp_simplepointnet",
    "tcp_v3": "TCP.models.tcp_v3",
}

_DEFAULT_MODEL = "tcp_simplepointnet"

_model_name = os.environ.get("TCP_MODEL", _DEFAULT_MODEL)

if _model_name not in _MODEL_REGISTRY:
    raise ValueError(
        f"未知的 TCP_MODEL={_model_name!r}。"
        f"可选: {list(_MODEL_REGISTRY.keys())}"
    )

_mod = importlib.import_module(_MODEL_REGISTRY[_model_name])
TCP = _mod.TCP
PIDController = _mod.PIDController

# 可选：暴露模块级 __all__
__all__ = ["TCP", "PIDController"]
