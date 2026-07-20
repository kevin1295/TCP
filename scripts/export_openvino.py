#!/usr/bin/env python3
"""
④ OpenVINO 量化导出脚本

将 PyTorch 模型导出为 ONNX，再通过 OpenVINO 转换为 IR 格式，
支持 FP32 / FP16 / INT8 量化。

用法:
    python scripts/export_openvino.py --ckpt /path/to/checkpoint.ckpt --output ./weights
"""

import argparse
import os
import numpy as np
import torch

from TCP.model import TCP
from TCP.config import GlobalConfig


def export_onnx(model, output_path):
    """导出为 ONNX 格式。"""
    dummy_rgb = torch.randn(1, 3, 256, 900)
    dummy_state = torch.randn(1, 1 + 2 + 6)
    dummy_target = torch.randn(1, 2)
    dummy_pcl = torch.randn(1, 16384, 4)

    torch.onnx.export(
        model,
        (dummy_rgb, dummy_state, dummy_target, dummy_pcl),
        output_path,
        input_names=["rgb", "state", "target_point", "pcl"],
        output_names=[
            "pred_wp", "mu_branches", "sigma_branches",
            "pred_speed", "pred_value_traj", "pred_value_ctrl",
        ],
        dynamic_axes={
            "rgb": {0: "batch"},
            "state": {0: "batch"},
            "target_point": {0: "batch"},
            "pcl": {0: "batch"},
        },
        opset_version=17,
    )
    print(f"[OK] ONNX 已导出: {output_path}")


def convert_to_openvino(onnx_path, output_dir, fp16=False, int8=False):
    """
    将 ONNX 转换为 OpenVINO IR。
    需要安装 openvino-dev: pip install openvino-dev
    """
    try:
        import openvino as ov
        from openvino.tools import mo
    except ImportError:
        print("[ERROR] 需要安装 openvino-dev: pip install openvino-dev")
        return

    # FP32 / FP16 转换
    model = mo.convert_model(onnx_path, compress_to_fp16=fp16)
    precision = "fp16" if fp16 else "fp32"
    ir_path = os.path.join(output_dir, f"tcp_deploy_{precision}")
    ov.save_model(model, ir_path + ".xml")
    print(f"[OK] OpenVINO IR ({precision}) 已导出: {ir_path}.xml")

    # INT8 量化（需要 NNCF + 代表数据集）
    if int8:
        print("[WARN] INT8 量化需要代表数据集和 NNCF，请参考:")
        print("  https://docs.openvino.ai/2024/openvino-workflow/model-optimization-guide/quantizing-models-post-training.html")


def main():
    parser = argparse.ArgumentParser(description="OpenVINO 量化导出")
    parser.add_argument("--ckpt", type=str, required=True, help="PyTorch checkpoint 路径")
    parser.add_argument("--output", type=str, default="./weights", help="输出目录")
    parser.add_argument("--fp16", action="store_true", help="导出 FP16")
    parser.add_argument("--int8", action="store_true", help="启用 INT8 量化（实验性）")
    parser.add_argument("--onnx-only", action="store_true", help="仅导出 ONNX，不做 OpenVINO 转换")
    args = parser.parse_args()

    os.makedirs(args.output, exist_ok=True)

    # 加载模型
    config = GlobalConfig()
    model = TCP(config)
    state_dict = torch.load(args.ckpt, map_location="cpu")
    if "state_dict" in state_dict:
        state_dict = state_dict["state_dict"]
    model.load_state_dict(state_dict, strict=False)
    model.eval()
    print(f"[OK] 模型已加载: {args.ckpt}")

    # 导出 ONNX
    onnx_path = os.path.join(args.output, "model.onnx")
    export_onnx(model, onnx_path)

    if not args.onnx_only:
        # FP32
        convert_to_openvino(onnx_path, args.output, fp16=False)
        # FP16
        if args.fp16:
            convert_to_openvino(onnx_path, args.output, fp16=True)
        # INT8
        if args.int8:
            convert_to_openvino(onnx_path, args.output, int8=True)

    print("[Done] 导出完成")


if __name__ == "__main__":
    main()
