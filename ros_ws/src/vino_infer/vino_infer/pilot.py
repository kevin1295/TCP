import numpy as np
import cv2 as cv

from std_msgs.msg import Float32
from sensor_msgs.msg import Image, Imu, NavSatFix, PointCloud2
from sensor_msgs_py.point_cloud2 import read_points
from carla_msgs.msg import CarlaEgoVehicleControl, CarlaRoute, CarlaSpeedometer
from cv_bridge.core import CvBridge
import transforms3d

from collections import deque
import time

from .planner import RoutePlanner

# 复用后处理函数（纯 numpy，无 ROS 依赖）
from .postprocess import PIDController, process_action, control_pid, align_points
from TCP.config import GlobalConfig

# ===========================================================
# ROS 消息 → numpy 转换（pilot.py 独有，不需共享）
# ===========================================================

def _image_transform(data: Image):
    """
    图像数据预处理。
    注意：归一化常数应用于 [0,1] 范围（与 torchvision.ToTensor 一致）。
    """
    bridge = CvBridge()
    img = bridge.imgmsg_to_cv2(data, "rgb8").astype(np.float32)  # (H, W, 3), uint8 → float32

    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    img = img / 255.0                # [0,255] → [0,1]
    img = (img - mean) / std
    return img


def _speed_transform(data: CarlaSpeedometer):
    return data.speed / 12.0


def _gps_transform(data: NavSatFix):
    return np.array([data.longitude, data.latitude])


def _lidar_transform(data: PointCloud2):
    data = read_points(data)
    data = align_points(data)
    return data


def _imu_transform(data: Imu):
    q_x = data.orientation.x
    q_y = data.orientation.y
    q_z = data.orientation.z
    q_w = data.orientation.w
    roll, pitch, yaw = transforms3d.euler.quat2euler([q_x, q_y, q_z, q_w])
    # ROS (ENU) → Carla (compass) 坐标系转换
    carla_compass = - (yaw - np.pi / 2)
    carla_compass = carla_compass % (2 * np.pi)
    return carla_compass


def _route_transform(data: CarlaRoute):
    cmd_seq = data.road_options
    pose_seq = data.poses
    plan = []
    assert len(cmd_seq) == len(pose_seq), \
        "Command sequence and pose sequence must have the same length"
    for cmd, pose in zip(cmd_seq, pose_seq):
        plan.append(
            (
                {
                    'lat': pose.position.y,
                    'lon': pose.position.x,
                    'alt': pose.position.z,
                },
                cmd
            )
        )
    return plan


# ===========================================================
# 后处理：numpy 版 action/control（从 TCP/model.py 提取，适配 OpenVINO）
# ===========================================================

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
    与 TCP/model.py 中 TCP.control_pid 逻辑一致，但接收 PIDController 实例和 config。

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
    # flip y (forward is negative in our waypoints)
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


# ===========================================================
# Pilot：编排预处理 → 推理 → 后处理
# ===========================================================

class Pilot:
    def __init__(self, config=None):
        config = config if config is not None else GlobalConfig()
        self.turn_controller = PIDController(
            K_P=config.turn_KP, K_I=config.turn_KI, K_D=config.turn_KD, n=config.turn_n
        )
        self.speed_controller = PIDController(
            K_P=config.speed_KP, K_I=config.speed_KI, K_D=config.speed_KD, n=config.speed_n
        )
        self.config = config

        self.alpha = 0.3
        self.status = 0
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = deque()

        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

    def set_global_plan(self, global_plan: CarlaRoute):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(_route_transform(global_plan), True)
        self.initialized = True

    def preprocess(self, rgb, speed, gps, lidar, imu):
        rgb = _image_transform(rgb)
        speed_val = _speed_transform(speed)
        gps = _gps_transform(gps)
        lidar = _lidar_transform(lidar)
        compass = _imu_transform(imu)

        if np.isnan(compass):
            compass = 0.0

        pos = (gps - self._route_planner.mean) * self._route_planner.scale
        next_wp, next_cmd = self._route_planner.run_step(pos)

        theta = compass + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)]
        ])
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        target_point = local_command_point.reshape(1, 2)

        cmd_one_hot = [0] * 6
        cmd_one_hot[4 if next_cmd < 0 else next_cmd] = 1

        rgb = rgb.reshape(1, 900, 256, 3)
        state = np.concatenate([speed_val, target_point, cmd_one_hot], axis=0).reshape(1, -1)
        target_point = target_point.reshape(1, 2)
        lidar = lidar.reshape(1, 16384, 4)

        metadata = {'next_cmd': next_cmd, 'speed': speed_val, 'next_wp': next_wp}
        return rgb, state, target_point, lidar, metadata

    def postprocess(self, pred, nxt_cmd, gt_velocity, target_point):
        steer_ctrl, throttle_ctrl, brake_ctrl, _ = process_action(
            pred, nxt_cmd, gt_velocity, target_point
        )
        steer_traj, throttle_traj, brake_traj, metadata_traj = control_pid(
            pred['pred_wp'], gt_velocity, target_point,
            self.turn_controller, self.speed_controller, self.config
        )
        if brake_traj < 0.05:
            brake_traj = 0.0
        if throttle_traj > brake_traj:
            brake_traj = 0.0

        self.pid_metadata = metadata_traj
        control = {}

        if self.status == 0:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'traj'
            control['steer'] = np.clip(
                self.alpha * steer_ctrl + (1 - self.alpha) * steer_traj, -1, 1
            )
            control['throttle'] = np.clip(
                self.alpha * throttle_ctrl + (1 - self.alpha) * throttle_traj, 0, 0.75
            )
            control['brake'] = np.clip(
                self.alpha * brake_ctrl + (1 - self.alpha) * brake_traj, 0, 1
            )
        else:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'ctrl'
            control['steer'] = np.clip(
                self.alpha * steer_traj + (1 - self.alpha) * steer_ctrl, -1, 1
            )
            control['throttle'] = np.clip(
                self.alpha * throttle_traj + (1 - self.alpha) * throttle_ctrl, 0, 0.75
            )
            control['brake'] = np.clip(
                self.alpha * brake_traj + (1 - self.alpha) * brake_ctrl, 0, 1
            )

        self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
        self.pid_metadata['brake_traj'] = float(brake_traj)

        if control['brake'] > 0.5:
            control['throttle'] = float(0)

        if len(self.last_steers) >= 20:
            self.last_steers.popleft()
        self.last_steers.append(abs(float(control['steer'])))
        num = sum(1 for s in self.last_steers if s > 0.10)
        if num > 10:
            self.status = 1
            self.steer_step += 1
        else:
            self.status = 0
        self.pid_metadata['status'] = self.status

        return control
