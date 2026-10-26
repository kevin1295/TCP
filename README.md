# FishDrive — 基于 TCP 的第 1/2 代自动驾驶模型

> 本目录承载第 **1/2 代**模型（TCP 底座 + LiDAR）的采集 / 训练 / 评测 / 部署框架；
> 第 3 代模型已迁移到独立框架 [`new_LEAD`](../new_LEAD)（LEAD / TFv6）。
> 三代模型与各仓库的关系见上一级 [`CherryDrive/README.md`](../README.md)。
> 子仓库活动分支：`dev-lidar_data`。

## 简介

本项目源自 2026 年「Intel 杯」大学生电子设计竞赛 AI 专题赛赛题，团队考查开发平台性能后尝试构建自动驾驶模型。

以经典自动驾驶模型 [TCP（NeurIPS 2022）](https://arxiv.org/abs/2206.08129) 为基础进行创新设计：

- **v1**：原始 TCP 基线（相机输入，无 LiDAR）；
- **v2**：集成 **LiDAR 点云模块**（SimplePointNet）、设计特征融合方式，并完成 OpenVINO 量化与 ROS2 部署；
- 采集端使用 **Roach** RL 专家生成含点云的自采数据集。

## 模型代际（`TCP_MODEL` 切换）

模型入口为 `TCP/model.py` 的注册表，用环境变量 `TCP_MODEL` 选择变体：

| `TCP_MODEL` | 模块 | 说明 |
|---|---|---|
| `tcp_original` | `TCP/models/tcp_original.py` | v1：原始 TCP（无 LiDAR，`forward(img, state, target_point)`） |
| `tcp_simplepointnet` | `TCP/models/tcp_simplepointnet.py` | **v2（默认）**：TCP + SimplePointNet LiDAR，`forward(img, state, target_point, pcl)` |
| `tcp_v3` | `TCP/models/tcp_v3.py` | **占位（未实现）**：双路 PointNet + 门控融合；实际 v3 在 `../new_LEAD` 实现 |

```bash
export TCP_MODEL=tcp_original        # v1
export TCP_MODEL=tcp_simplepointnet  # v2（默认）
TCP_MODEL=tcp_original python TCP/train.py --gpus 1   # 单次覆盖
```

## 任务流程

```
① Roach 采集 ──→ ② TCP 训练 ──→ ③ Torch 推理测试
                      │
                      ▼
                ④ OpenVINO 导出
                      │
              ┌───────┴───────┐
              ▼               ▼
       ⑤ 容器外推理       ⑥ ROS2 容器推理
       (OpenVINO)          (OpenVINO)
```

| # | 任务 | 入口 / 命令 |
|---|------|------|
| ① | Roach 采集训练集（含 LiDAR） | `bash scripts/collect_data.sh`（后处理 `tools/filter_data.py` + `tools/gen_data.py`） |
| ② | TCP 模型训练 | `bash scripts/train.sh [gpus] [batch] [epochs] [lr] [id]` |
| ③ | PyTorch 直接推理 | `python scripts/test_torch.py --ckpt /path/to/ckpt`（半成品，闭环见 ⑤/⑥） |
| ④ | OpenVINO 导出 | `python scripts/export_openvino.py --ckpt /path/to/ckpt`（FP32 / FP16 / ONNX-only） |
| ⑤ | **容器外**推理（宿主机，无 PyTorch/ROS2） | Leaderboard 内使用 `leaderboard/team_code/ov_agent.py` 直接加载 IR |
| ⑥ | **容器内**推理（ROS2 + OpenVINO） | `docker/ros_client` 容器内运行 `ros_ws/src/vino_infer` 推理节点 |

- ⑤ 的已部署 OpenVINO 权重实际位于 `ros_ws/src/vino_infer/weights/`，**目前只剩 `model_fp32.xml` + `.bin`**。
  原先还有 `tcp_deploy_fp16` / `tcp_deploy_int8` 两套，因导出时输入宽度写成 928（源码与导出脚本都是 900）
  且 fp16 那份内部实为 FP32，已删除；需要时用 `scripts/export_openvino.py` 重导。
- ⑥ 的详细构建与启动方式见 `.claude/CLAUDE.md`；测试/评测入口集中在 `leaderboard/team_code/`。

## 已知缺口与使用提醒

- **文档与代码不一致**（三代交接期遗留，归档时如实标注）：
  - `scripts/infer_openvino.py` 在 README/CLAUDE 中被引用，但仓库内**不存在**；
  - `docker/ros_client/` 下没有 `docker-compose.yml`，需自行 `docker build`/`docker run`（基于其 `Dockerfile`）；
  - `leaderboard/team_code/weights/` 为空目录，真实权重见 `ros_ws/src/vino_infer/weights/`；
  - `scripts/test_torch.py` 为占位实现，实际闭环请走 Leaderboard 的 torch agent（`team_code/tcp_agent.py`）或 `ov_agent.py`。
- `docker/ros_client/` 依赖 `ros-bridge` 子模块（嵌套 `carla_msgs`）。克隆后请执行：

  ```bash
  git submodule update --init --recursive
  ```

- **数据集**：本项目使用**自采数据集**（含 LiDAR 点云），须经 CARLA + Roach 采集；原 TCP 数据集不含点云，不适用（下载脚本 `scripts/download_tcp_data.sh` 仅供 v1 基线参考）。

## 目录结构

```
├── TCP/                # 模型训练核心：model.py（注册表）/ models/（v1/v2/v3）、config.py、train.py
├── roach/              # Roach RL 专家模型（数据采集 teacher）
├── leaderboard/        # CARLA Leaderboard 评估框架 + Agent（team_code/：torch、ov、roach_ap、ros2_tcp）
├── scenario_runner/    # CARLA Scenario Runner
├── ros_ws/             # ROS2 工作空间：src/vino_infer（OpenVINO 推理包 + 权重）
├── docker/             # 容器编排：py311 / py37（训练）、ros_client（ROS2 Humble 推理）
├── scripts/            # 流水线入口（train / collect_data / export_openvino / start_carla …）
├── tools/              # 数据处理（filter_data.py、gen_data.py、statistics.py …）
├── results_TCP.json    # 一次 Leaderboard 评测断点（sensors 含 carla_lidar，为 v2 agent 的运行）
├── .claude/CLAUDE.md   # 代码维护指引（六阶段任务与模型架构细节）
├── TCPrefer.md         # TCP 原论文要点摘记
└── FishDriveWork.md    # 数据采集改造研究笔记
```

## 鸣谢

感谢 [华北电力大学 MOS 实验室](https://mos.ncepu.edu.cn/index.htm) 的培养并给予了我们创新的平台。  
感谢两位队友 [gyk521](https://gitee.com/gyk521) 与 [chenyanlingXIN](https://gitee.com/chenyanlingXIN) 的陪伴与支持！

## 引用

```bibtex
@inproceedings{wu2022trajectoryguided,
  title={Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline},
  author={Penghao Wu and Xiaosong Jia and Li Chen and Junchi Yan and Hongyang Li and Yu Qiao},
  booktitle={NeurIPS},
  year={2022}
}
```

## 许可证

[Apache License 2.0](LICENSE)
