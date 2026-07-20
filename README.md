# FishDrive

> 基于 TCP 的自动驾驶模型，集成 LiDAR 点云模块与 ROS2 部署方案。
> 由于计算平台几经变动，全部代码正在整理。

## 立项

本项目源自于 2026 年"Intel 杯"大学生电子设计竞赛 AI 专题赛赛题，团队考查开发平台性能后尝试构建自动驾驶模型。

在经典自动驾驶模型 [TCP（NeurIPS 2022）](https://arxiv.org/abs/2206.08129) 基础上进行创新设计，主要创新包括添加点云模块、设计特征融合方式、量化部署等。

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
```

| # | 任务 | 命令 |
|---|------|------|
| ① | Roach 采集训练集 | `bash scripts/collect_data.sh` |
| ② | TCP 模型训练 | `bash scripts/train.sh [gpus]` |
| ③ | PyTorch 直接推理 | `python scripts/test_torch.py --ckpt /path/to/ckpt` |
| ④ | OpenVINO 导出 | `python scripts/export_openvino.py --ckpt /path/to/ckpt` |
| ⑤ | OpenVINO 容器外推理 | `python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py --agent=leaderboard/team_code/ov_agent.py --agent-config=./weights/model.xml` |
| ⑥ | ROS2 容器推理 | `cd docker/ros_client && docker-compose up -d` |

> **注意**：本项目使用 **自采数据集**（含 LiDAR 点云），需通过 CARLA + Roach 采集。原 TCP 数据集不含点云，不适用。`docker/ros_client/` 依赖 ros-bridge 子模块（含 carla_msgs 嵌套子模块），克隆后请执行 `git submodule update --init --recursive`。

## 项目结构

```
├── TCP/                # 模型训练核心（model.py, config.py, train.py）
├── roach/              # Roach RL 专家模型（数据采集 teacher）
├── leaderboard/        # CARLA Leaderboard 评估框架 + Agent
├── ros_ws/             # ROS2 工作空间（vino_infer 推理包 + OpenVINO 权重）
├── docker/             # 容器配置（训练 CUDA 12.8 + 推理 ROS2 Humble）
├── scripts/            # 流水线入口脚本
├── tools/              # 数据处理（filter_data.py, gen_data.py）
└── scenario_runner/    # CARLA Scenario Runner
```

## 鸣谢

感谢 [华北电力大学 MOS 实验室](https://mos.ncepu.edu.cn/index.htm) 的培养并给予了我们创新的平台。  
感谢两位队友 [gyk521](https://gitee.com/gyk521) [chenyanlingXIN](https://gitee.com/chenyanlingXIN) 的陪伴与支持！

## 引用

```bibtex
@inproceedings{wu2022trajectoryguided,
  title={Trajectory-guided Control Prediction for End-to-end Autonomous Driving: A Simple yet Strong Baseline},
  author={Penghao Wu and Xiaosong Jia and Li Chen and Junchi Yan and Hongyang Li and Yu Qiao},
  booktitle={NeurIPS},
  year={2022}
}
```

## License

[Apache License 2.0](LICENSE)
