"""
v3: TCP + 双路 PointNet + 门控融合

=====================================================================
⚠️ 重要：这份实现是 AI 复现，不是当年的代码。
   原始 v3 源码在机器之间传输时丢失，只留下一个 34 行的占位文件
   （docstring 里写着设计、构造函数直接 raise NotImplementedError）。
   现在这份是 2026-10 依据那段设计描述重写的：
     - 从未训练过
     - 从未跑过闭环
     - 投影/外参方向未在真实数据上核对过
   请当作"设计的一种实现示例"，不要当作可复现的结果引用。
   当年真正训练并参赛的是第 3 代 LEAD/TFv6（见 CherryDrive/new_LEAD）。
=====================================================================

设计（沿用原占位文件的设计说明，2026-10 补实现）：
  1. 用相机内参与 LiDAR->相机 外参把点云投到图像平面，按"是否落在图像内"拆成两路：
     - 重合点：编码后与 ResNet34 的全局视觉特征做**门控融合**
     - 不重合点：单独编码，作为互补信息拼进来
  2. 两路压回 lidar_dim 维，下游三处注入点（速度 / 轨迹 / 控制）与 v2 完全一致，
     因此直接继承 v2 的分支结构、损失与 process_action。

与 v2 的差别只有一处：
  v2:  lidar_feat = SimplePointNet(pcl)                        # (B, lidar_dim)
  v3:  lidar_feat = fuse_proj([g*h_img + (1-g)*h_ov, h_non])    # (B, lidar_dim)
  其中 g = sigmoid(MLP([h_img, h_ov]))，逐通道门，h_img 由 1000 维视觉特征投影得到。

标定来源（均取本仓库内的 sensor spec，不用魔数）：
  - 相机: leaderboard/team_code/tcp_agent.py sensors() 的 rgb 项 —— 900x256, fov=100, 位姿 (-1.5, 0, 2.0)
  - LiDAR: 同文件 lidar 项 —— 位姿 (1.3, 0, 2.5), yaw=-90
  - 内参 fx = fy = (W/2)/tan(fov/2)，主点取图像中心（见 config.cam_* 字段）

已知局限（写清楚，别当它是标定好的）：
  - 投影只判"落在图像矩形内 + 深度为正"，不做遮挡剔除、不做镜头畸变；
  - 外参只取水平 yaw（两个传感器的 roll/pitch 都是 0），坐标系沿用 CARLA 的
    x 前 / y 右 / z 上，图像 v 轴向下；
  - 上述假设**未在真实数据上验证过**，若上线跑闭环，请先用一小段数据画出
    投影散点图核对左右上下没有翻转。
"""

import math

import torch
from torch import nn

from TCP.config import GlobalConfig as _GlobalConfig
from TCP.models.tcp_simplepointnet import (  # noqa: F401  (PIDController 供 model.py 统一导出)
    PIDController,
    SimplePointNet,
    TCP as _V2TCP,
)


def build_camera_intrinsics(config):
    """
    由 sensor spec 推导针孔相机内参。

    Args:
        config: GlobalConfig，需要 img_w / img_h / cam_fov。
    Returns:
        (fx, fy, cx, cy)
    """
    half_fov = math.radians(config.cam_fov) / 2.0
    fx = (config.img_w / 2.0) / math.tan(half_fov)
    fy = (config.img_h / 2.0) / math.tan(half_fov)
    cx = config.img_w / 2.0
    cy = config.img_h / 2.0
    return fx, fy, cx, cy


class TCP(_V2TCP):
    """
    第三代模型：双路 PointNet + 视觉门控融合。

    forward 的签名与输出字典和 v2 完全一致：
        img          (B, 3, H, W)
        state        (B, 9)        [speed, target_point(2), command_one_hot(6)]
        target_point (B, 2)
        pcl          (B, N, 4)     [x, y, z, intensity]，N=16384
    """

    def __init__(self, config=None, lidar_dim=256):
        cfg = config if config is not None else _GlobalConfig()
        super().__init__(config=cfg, lidar_dim=lidar_dim)

        # v2 的 lidar_encoder 在 v3 里不用，换成双路
        self.overlap_encoder = SimplePointNet(lidar_dim=lidar_dim)
        self.nonoverlap_encoder = SimplePointNet(lidar_dim=lidar_dim)

        # 门控：输入 [h_img(1000), h_ov(lidar_dim)] -> lidar_dim 维逐通道权重
        self.gate = nn.Sequential(
            nn.Linear(1000 + lidar_dim, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, lidar_dim),
            nn.Sigmoid(),
        )
        # 视觉特征降到 lidar_dim，才能与门控输出逐元素相乘
        self.image_proj = nn.Sequential(
            nn.Linear(1000, lidar_dim),
            nn.ReLU(inplace=True),
        )
        # [融合特征, 非重合特征] -> lidar_dim，保持下游接口不变
        self.fuse_proj = nn.Sequential(
            nn.Linear(2 * lidar_dim, lidar_dim),
            nn.ReLU(inplace=True),
        )

    # ------------------------------------------------------------------ #
    # 点云分解
    # ------------------------------------------------------------------ #
    def split_point_cloud(self, pcl):
        """
        按"是否落在相机视野内"拆点云，返回布尔掩码。

        Args:
            pcl: (B, N, 4)
        Returns:
            (B, N, 1) 的 bool 张量，True = 落在图像内
        """
        cfg = self.config
        xyz = pcl[..., :3]                                   # (B, N, 3)

        # 1) LiDAR -> 车体：绕 z 轴旋转 lidar_yaw，再平移 lidar_pos
        yaw = math.radians(cfg.lidar_yaw)
        cos_y, sin_y = math.cos(yaw), math.sin(yaw)
        x_l, y_l, z_l = xyz[..., 0], xyz[..., 1], xyz[..., 2]
        x_v = x_l * cos_y - y_l * sin_y + cfg.lidar_pos[0]
        y_v = x_l * sin_y + y_l * cos_y + cfg.lidar_pos[1]
        z_v = z_l + cfg.lidar_pos[2]

        # 2) 车体 -> 相机：相机 roll=pitch=0，只做平移
        x_c = x_v - cfg.cam_pos[0]                           # 深度（前向）
        y_c = y_v - cfg.cam_pos[1]                           # 右向
        z_c = z_v - cfg.cam_pos[2]                           # 上向

        # 3) 透视投影（图像 v 轴向下，z 取负号）
        fx, fy, cx, cy = build_camera_intrinsics(cfg)
        safe = torch.where(x_c.abs() < 1e-3, torch.full_like(x_c, 1e-3), x_c)
        u = fx * (y_c / safe) + cx
        v = cy - fy * (z_c / safe)

        inside = (x_c > 0.5) & (u >= 0) & (u < cfg.img_w) & (v >= 0) & (v < cfg.img_h)
        return inside.unsqueeze(-1)                          # (B, N, 1)

    def encode_lidar(self, pcl, feature_emb):
        """
        双路编码 + 门控融合。

        Args:
            pcl:         (B, N, 4)
            feature_emb: (B, 1000)
        Returns:
            (B, lidar_dim)
        """
        mask_ov = self.split_point_cloud(pcl)                # (B, N, 1) bool
        dtype = pcl.dtype
        pcl_ov = pcl * mask_ov.to(dtype)                     # 重合点，其余置零
        pcl_non = pcl * (~mask_ov).to(dtype)                 # 不重合点

        h_ov = self.overlap_encoder(pcl_ov)                  # (B, lidar_dim)
        h_non = self.nonoverlap_encoder(pcl_non)             # (B, lidar_dim)

        h_img = self.image_proj(feature_emb)                 # (B, lidar_dim)
        g = self.gate(torch.cat([feature_emb, h_ov], dim=1))  # (B, lidar_dim)
        h_fused = g * h_img + (1.0 - g) * h_ov               # (B, lidar_dim)

        return self.fuse_proj(torch.cat([h_fused, h_non], dim=1))   # (B, lidar_dim)

    # ------------------------------------------------------------------ #
    # 前向：复制 v2 的分支流程，只改 lidar_feat 的来源
    # ------------------------------------------------------------------ #
    def forward(self, img, state, target_point, pcl):
        feature_emb, cnn_feature = self.perception(img)       # (B,1000), (B,512,8,29)
        lidar_feat = self.encode_lidar(pcl, feature_emb)      # (B, lidar_dim)

        orig_feature_emb = feature_emb
        outputs = {}

        # ---------- 轨迹分支 ----------
        combined_feat_traj = torch.cat([orig_feature_emb, lidar_feat], dim=1)
        outputs['pred_speed'] = self.speed_branch(combined_feat_traj)

        measurement_feature = self.measurements(state)                          # (B,128)
        j_traj_input = torch.cat([combined_feat_traj, measurement_feature], dim=1)
        j_traj = self.join_traj(j_traj_input)                                   # (B,256)
        outputs['pred_value_traj'] = self.value_branch_traj(j_traj)
        outputs['pred_features_traj'] = j_traj

        z = j_traj
        output_wp = list()
        traj_hidden_state = list()
        x = torch.zeros(size=(z.shape[0], 2), dtype=z.dtype).type_as(z)
        for _ in range(self.config.pred_len):
            x_in = torch.cat([x, target_point], dim=1)
            z = self.decoder_traj(x_in, z)
            traj_hidden_state.append(z)
            dx = self.output_traj(z)
            x = dx + x
            output_wp.append(x)

        outputs['pred_wp'] = torch.stack(output_wp, dim=1)
        traj_hidden_state = torch.stack(traj_hidden_state, dim=1)

        # ---------- 控制分支 ----------
        init_att = self.init_att(measurement_feature).view(-1, 1, 8, 29)
        vis_feat_w = torch.sum(cnn_feature * init_att, dim=(2, 3))              # (B,512)

        combined_vis = torch.cat([vis_feat_w, lidar_feat], dim=1)
        j_ctrl_input = torch.cat([combined_vis, measurement_feature], dim=1)
        j_ctrl = self.join_ctrl(j_ctrl_input)                                   # (B,256)
        outputs['pred_value_ctrl'] = self.value_branch_ctrl(j_ctrl)
        outputs['pred_features_ctrl'] = j_ctrl

        policy = self.policy_head(j_ctrl)
        outputs['mu_branches'] = self.dist_mu(policy)
        outputs['sigma_branches'] = self.dist_sigma(policy)

        x = j_ctrl
        mu = outputs['mu_branches']
        sigma = outputs['sigma_branches']
        future_feature, future_mu, future_sigma = [], [], []
        h = torch.zeros(size=(x.shape[0], 256), dtype=x.dtype).type_as(x)

        for step in range(self.config.pred_len):
            x_in = torch.cat([x, mu, sigma], dim=1)
            h = self.decoder_ctrl(x_in, h)
            wp_att = self.wp_att(torch.cat([h, traj_hidden_state[:, step]], 1)).view(-1, 1, 8, 29)
            new_feature_emb = torch.sum(cnn_feature * wp_att, dim=(2, 3))
            merged_feature = self.merge(torch.cat([h, new_feature_emb], 1))
            dx = self.output_ctrl(merged_feature)
            x = dx + x

            policy = self.policy_head(x)
            mu = self.dist_mu(policy)
            sigma = self.dist_sigma(policy)
            future_feature.append(x)
            future_mu.append(mu)
            future_sigma.append(sigma)

        outputs['future_feature'] = future_feature
        outputs['future_mu'] = future_mu
        outputs['future_sigma'] = future_sigma
        return outputs
