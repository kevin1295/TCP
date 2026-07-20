"""
后处理函数集合：纯 numpy 实现，无 ROS 依赖。

包含 TCP 模型推理后的 action 提取和 PID 控制转换。
这些函数可被 ROS2 推理节点 (pilot.py) 和非 ROS 环境 (ov_agent.py) 共用。
"""

import numpy as np
from collections import deque

from TCP.model import PIDController
from TCP.config import GlobalConfig
from tools.gen_data import align_points


def _get_action_beta(alpha, beta):
    """从 Beta 分布参数中提取确定性 action（numpy 版）。"""
    x = np.zeros_like(alpha)
    x[:, 1] += 0.5
    mask1 = (alpha > 1) & (beta > 1)
    x[mask1] = (alpha[mask1] - 1) / (alpha[mask1] + beta[mask1] - 2)
    mask2 = (alpha <= 1) & (beta > 1)
    x[mask2] = 0.0
    mask3 = (alpha > 1) & (beta <= 1)
    x[mask3] = 1.0
    mask4 = (alpha <= 1) & (beta <= 1)
    x[mask4] = alpha[mask4] / np.clip(alpha[mask4] + beta[mask4], a_min=1e-5)
    x = x * 2 - 1
    return x


def process_action(pred, command, speed, target_point):
    """
    从预测分布中提取 action（numpy 版）。
    pred 是 OpenVINO 推理输出的 dict，key 含 'mu_branches' / 'sigma_branches'。
    与 TCP/model.py 中 TCP.process_action 逻辑一致。
    """
    mu = pred['mu_branches'].reshape(1, 2)
    sigma = pred['sigma_branches'].reshape(1, 2)
    action = _get_action_beta(mu, sigma)
    acc, steer = action[0].astype(np.float64)
    if acc >= 0.0:
        throttle = acc
        brake = 0.0
    else:
        throttle = 0.0
        brake = np.abs(acc)
    throttle = np.clip(throttle, 0, 1)
    steer = np.clip(steer, -1, 1)
    brake = np.clip(brake, 0, 1)
    metadata = {
        'speed': float(speed.astype(np.float64)),
        'steer': float(steer),
        'throttle': float(throttle),
        'brake': float(brake),
        'command': command,
        'target_point': tuple(target_point[0].astype(np.float64)),
    }
    return steer, throttle, brake, metadata


def control_pid(waypoints, velocity, target, turn_controller, speed_controller, config):
    """
    用 PID 控制器将 waypoints 转换为控制信号（numpy 版）。
    与 TCP/model.py 中 TCP.control_pid 逻辑一致。

    Args:
        waypoints: ndarray (pred_len, 2)
        velocity: ndarray scalar
        target: ndarray (2,)
        turn_controller: PIDController 实例
        speed_controller: PIDController 实例
        config: GlobalConfig 实例（或包含同名属性的对象）
    """
    waypoints = waypoints.copy()
    target = target.copy()
    waypoints[:, 1] *= -1
    target[1] *= -1

    num_pairs = len(waypoints) - 1
    best_norm = 1e5
    desired_speed = 0
    aim = waypoints[0]
    for i in range(num_pairs):
        desired_speed += np.linalg.norm(waypoints[i + 1] - waypoints[i]) * 2.0 / num_pairs
        norm = np.linalg.norm((waypoints[i + 1] + waypoints[i]) / 2.0)
        if abs(config.aim_dist - best_norm) > abs(config.aim_dist - norm):
            aim = waypoints[i]
            best_norm = norm

    aim_last = waypoints[-1] - waypoints[-2]
    angle = np.degrees(np.pi / 2 - np.arctan2(aim[1], aim[0])) / 90
    angle_last = np.degrees(np.pi / 2 - np.arctan2(aim_last[1], aim_last[0])) / 90
    angle_target = np.degrees(np.pi / 2 - np.arctan2(target[1], target[0])) / 90

    use_target_to_aim = np.abs(angle_target) < np.abs(angle)
    use_target_to_aim = use_target_to_aim or (
        np.abs(angle_target - angle_last) > config.angle_thresh
        and target[1] < config.dist_thresh
    )
    angle_final = angle_target if use_target_to_aim else angle

    steer = turn_controller.step(angle_final)
    steer = np.clip(steer, -1.0, 1.0)

    speed = float(velocity)
    brake = desired_speed < config.brake_speed or (speed / desired_speed) > config.brake_ratio

    delta = np.clip(desired_speed - speed, 0.0, config.clip_delta)
    throttle = speed_controller.step(delta)
    throttle = np.clip(throttle, 0.0, config.max_throttle)
    throttle = throttle if not brake else 0.0

    metadata = {
        'speed': float(speed),
        'steer': float(steer),
        'throttle': float(throttle),
        'brake': float(brake),
        'wp_4': tuple(waypoints[3].astype(np.float64)),
        'wp_3': tuple(waypoints[2].astype(np.float64)),
        'wp_2': tuple(waypoints[1].astype(np.float64)),
        'wp_1': tuple(waypoints[0].astype(np.float64)),
        'aim': tuple(aim.astype(np.float64)),
        'target': tuple(target.astype(np.float64)),
        'desired_speed': float(desired_speed.astype(np.float64)),
        'angle': float(angle.astype(np.float64)),
        'angle_last': float(angle_last.astype(np.float64)),
        'angle_target': float(angle_target.astype(np.float64)),
        'angle_final': float(angle_final.astype(np.float64)),
        'delta': float(delta.astype(np.float64)),
    }
    return steer, throttle, brake, metadata
