#!/usr/bin/env python3
"""
③ TCP 直接 PyTorch 推理测试

在 CARLA 仿真器中加载训练好的 TCP checkpoint 进行闭环测试。
（不依赖 ROS2，直接通过 CARLA Python API 运行）

用法:
    python scripts/test_torch.py --ckpt /path/to/checkpoint.ckpt [--town Town01]
"""

import argparse
import numpy as np
import torch
import carla

from TCP.model import TCP
from TCP.config import GlobalConfig


def main():
    parser = argparse.ArgumentParser(description="TCP PyTorch 推理测试")
    parser.add_argument("--ckpt", type=str, required=True, help="模型 checkpoint 路径")
    parser.add_argument("--town", type=str, default="Town01", help="CARLA 城镇")
    parser.add_argument("--host", type=str, default="localhost", help="CARLA 主机")
    parser.add_argument("--port", type=int, default=2000, help="CARLA 端口")
    args = parser.parse_args()

    # 加载配置和模型
    config = GlobalConfig()
    model = TCP(config)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    # 过滤掉不匹配的键（如 pretrained ResNet 权重）
    model.load_state_dict(state_dict, strict=False)
    model.eval().cuda()
    print(f"[OK] 模型已加载: {args.ckpt}")

    # 连接 CARLA
    client = carla.Client(args.host, args.port)
    client.set_timeout(30.0)
    world = client.load_world(args.town)
    print(f"[OK] 已连接到 CARLA: {args.town}")

    # 此处应实现完整的 run_step 循环
    # 参考: leaderboard/leaderboard/team_code/otcp_agent.py
    print("[INFO] Torch 推理框架已就绪，需在 CARLA 仿真环境中运行")


if __name__ == "__main__":
    main()
