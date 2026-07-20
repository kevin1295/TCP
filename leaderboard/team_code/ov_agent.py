#!/usr/bin/env python3
"""
⑤ OpenVINO 容器外推理 Agent

在 CARLA Leaderboard 中直接使用 OpenVINO 加载 IR 模型进行推理。
（不依赖 ROS2，不依赖 PyTorch）

用法:
    export SAVE_PATH=./results
    python ${LEADERBOARD_ROOT}/leaderboard/leaderboard_evaluator.py \
        --agent=${LEADERBOARD_ROOT}/leaderboard/team_code/ov_agent.py \
        --agent-config=${LEADERBOARD_ROOT}/leaderboard/team_code/weights/tcp_deploy_fp16.xml
"""

import os
import json
import datetime
import pathlib
import time
import math
from collections import deque

import cv2
import carla
import numpy as np
from PIL import Image

from leaderboard.autoagents import autonomous_agent

from TCP.config import GlobalConfig
from ros_ws.src.vino_infer.vino_infer.postprocess import (
    align_points,
    process_action,
    control_pid,
    PIDController,
)
from team_code.planner import RoutePlanner


SAVE_PATH = os.environ.get('SAVE_PATH', None)
POINTS_SCALE = 16384

_im_mean = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_im_std = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def get_entry_point():
    return 'OVAgent'


class OVAgent(autonomous_agent.AutonomousAgent):
    """
    OpenVINO 版 TCP Agent。
    用法同 otcp_agent.py，但使用 OpenVINO Runtime 而非 PyTorch。
    """
    def setup(self, path_to_conf_file):
        import openvino as ov

        self.track = autonomous_agent.Track.SENSORS
        self.alpha = 0.3
        self.status = 0
        self.steer_step = 0
        self.last_moving_status = 0
        self.last_moving_step = -1
        self.last_steers = deque()

        self.config_path = path_to_conf_file
        self.step = -1
        self.wall_start = time.time()
        self.initialized = False

        self.config = GlobalConfig()
        self.turn_controller = PIDController(
            K_P=self.config.turn_KP, K_I=self.config.turn_KI,
            K_D=self.config.turn_KD, n=self.config.turn_n,
        )
        self.speed_controller = PIDController(
            K_P=self.config.speed_KP, K_I=self.config.speed_KI,
            K_D=self.config.speed_KD, n=self.config.speed_n,
        )

        # 加载 OpenVINO IR 模型
        core = ov.Core()
        self.ov_model = core.compile_model(path_to_conf_file)
        self.infer_request = self.ov_model.create_infer_request()
        print(f"[OVAgent] 模型已加载: {path_to_conf_file}")

        self.takeover = False
        self.stop_time = 0
        self.takeover_time = 0

        self.save_path = None
        self.last_steers = deque()

        if SAVE_PATH is not None:
            now = datetime.datetime.now()
            string = pathlib.Path(os.environ['ROUTES']).stem + '_'
            string += '_'.join(map(lambda x: '%02d' % x,
                                   (now.month, now.day, now.hour, now.minute, now.second)))
            self.save_path = pathlib.Path(os.environ['SAVE_PATH']) / string
            self.save_path.mkdir(parents=True, exist_ok=False)
            (self.save_path / 'rgb').mkdir()
            (self.save_path / 'meta').mkdir()
            (self.save_path / 'bev').mkdir()
            (self.save_path / 'lidar').mkdir()

    def _init(self):
        self._route_planner = RoutePlanner(4.0, 50.0)
        self._route_planner.set_route(self._global_plan, True)
        self.initialized = True

    def _get_position(self, tick_data):
        gps = tick_data['gps']
        gps = (gps - self._route_planner.mean) * self._route_planner.scale
        return gps

    def sensors(self):
        return [
            {
                'type': 'sensor.camera.rgb',
                'x': -1.5, 'y': 0.0, 'z': 2.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'width': 900, 'height': 256, 'fov': 100,
                'id': 'rgb',
            },
            {
                'type': 'sensor.camera.rgb',
                'x': 0.0, 'y': 0.0, 'z': 50.0,
                'roll': 0.0, 'pitch': -90.0, 'yaw': 0.0,
                'width': 512, 'height': 512, 'fov': 5 * 10.0,
                'id': 'bev',
            },
            {
                'type': 'sensor.other.imu',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.05,
                'id': 'imu',
            },
            {
                'type': 'sensor.other.gnss',
                'x': 0.0, 'y': 0.0, 'z': 0.0,
                'roll': 0.0, 'pitch': 0.0, 'yaw': 0.0,
                'sensor_tick': 0.01,
                'id': 'gps',
            },
            {
                'type': 'sensor.speedometer',
                'reading_frequency': 20,
                'id': 'speed',
            },
            {
                'type': 'sensor.lidar.ray_cast',
                'x': 1.3, 'y': 0.0, 'z': 2.5,
                'roll': 0.0, 'pitch': 0.0, 'yaw': -90.0,
                'id': 'lidar',
            },
        ]

    def tick(self, input_data):
        self.step += 1

        rgb = cv2.cvtColor(input_data['rgb'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        bev = cv2.cvtColor(input_data['bev'][1][:, :, :3], cv2.COLOR_BGR2RGB)
        gps = input_data['gps'][1][:2]
        speed = input_data['speed'][1]['speed']
        compass = input_data['imu'][1][-1]
        lidar = input_data['lidar'][1]

        if math.isnan(compass):
            compass = 0.0

        result = {
            'rgb': rgb,
            'gps': gps,
            'speed': speed,
            'compass': compass,
            'bev': bev,
            'lidar': lidar,
        }

        pos = self._get_position(result)
        result['gps'] = pos
        next_wp, next_cmd = self._route_planner.run_step(pos)
        result['next_command'] = next_cmd.value

        theta = compass + np.pi / 2
        R = np.array([
            [np.cos(theta), -np.sin(theta)],
            [np.sin(theta), np.cos(theta)],
        ])
        local_command_point = np.array([next_wp[0] - pos[0], next_wp[1] - pos[1]])
        local_command_point = R.T.dot(local_command_point)
        result['target_point'] = tuple(local_command_point)

        return result

    def run_step(self, input_data, timestamp):
        if not self.initialized:
            self._init()

        tick_data = self.tick(input_data)

        if self.step < self.config.seq_len:
            control = carla.VehicleControl()
            control.steer = 0.0
            control.throttle = 0.0
            control.brake = 0.0
            return control

        gt_velocity = np.float64(tick_data['speed'])
        command = tick_data['next_command']
        if command < 0:
            command = 4
        command -= 1
        assert command in [0, 1, 2, 3, 4, 5]
        cmd_one_hot = [0] * 6
        cmd_one_hot[command] = 1

        speed = np.float64(tick_data['speed']) / 12.0

        # 图像预处理：归一化到 [0,1] 再减均值除方差
        rgb = tick_data['rgb'].astype(np.float32) / 255.0
        rgb = (rgb - _im_mean) / _im_std
        rgb = rgb.transpose(2, 0, 1)          # (H,W,C) → (C,H,W)
        rgb = rgb.reshape(1, 3, 256, 900)

        # LiDAR 对齐
        lidar = align_points(tick_data['lidar']).astype(np.float32)
        lidar = lidar.reshape(1, 16384, 4)

        target_point = np.array(tick_data['target_point'], dtype=np.float32).reshape(1, 2)
        state = np.concatenate([
            np.array([speed], dtype=np.float32),
            target_point[0],
            np.array(cmd_one_hot, dtype=np.float32),
        ], axis=0).reshape(1, -1)

        # OpenVINO 推理
        outputs = self.infer_request.infer(inputs={
            'rgb': rgb,
            'state': state,
            'target_point': target_point,
            'pcl': lidar,
        })

        # 后处理
        steer_ctrl, throttle_ctrl, brake_ctrl, _ = process_action(
            outputs, tick_data['next_command'], gt_velocity, target_point
        )
        steer_traj, throttle_traj, brake_traj, metadata_traj = control_pid(
            outputs['pred_wp'], gt_velocity, target_point,
            self.turn_controller, self.speed_controller, self.config,
        )
        if brake_traj < 0.05:
            brake_traj = 0.0
        if throttle_traj > brake_traj:
            brake_traj = 0.0

        self.pid_metadata = metadata_traj
        control = carla.VehicleControl()

        if self.status == 0:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'traj'
            control.steer = np.clip(self.alpha * steer_ctrl + (1 - self.alpha) * steer_traj, -1, 1)
            control.throttle = np.clip(
                self.alpha * throttle_ctrl + (1 - self.alpha) * throttle_traj, 0, 0.75
            )
            control.brake = np.clip(self.alpha * brake_ctrl + (1 - self.alpha) * brake_traj, 0, 1)
        else:
            self.alpha = 0.3
            self.pid_metadata['agent'] = 'ctrl'
            control.steer = np.clip(self.alpha * steer_traj + (1 - self.alpha) * steer_ctrl, -1, 1)
            control.throttle = np.clip(
                self.alpha * throttle_traj + (1 - self.alpha) * throttle_ctrl, 0, 0.75
            )
            control.brake = np.clip(self.alpha * brake_traj + (1 - self.alpha) * brake_ctrl, 0, 1)

        self.pid_metadata['steer_ctrl'] = float(steer_ctrl)
        self.pid_metadata['steer_traj'] = float(steer_traj)
        self.pid_metadata['throttle_ctrl'] = float(throttle_ctrl)
        self.pid_metadata['throttle_traj'] = float(throttle_traj)
        self.pid_metadata['brake_ctrl'] = float(brake_ctrl)
        self.pid_metadata['brake_traj'] = float(brake_traj)

        if control.brake > 0.5:
            control.throttle = float(0)

        if len(self.last_steers) >= 20:
            self.last_steers.popleft()
        self.last_steers.append(abs(float(control.steer)))
        num = sum(1 for s in self.last_steers if s > 0.10)
        if num > 10:
            self.status = 1
            self.steer_step += 1
        else:
            self.status = 0
        self.pid_metadata['status'] = self.status

        if SAVE_PATH is not None and self.step % 10 == 0:
            self.save(tick_data)
        return control

    def save(self, tick_data):
        frame = self.step // 10
        Image.fromarray(tick_data['rgb']).save(self.save_path / 'rgb' / ('%04d.png' % frame))
        Image.fromarray(tick_data['bev']).save(self.save_path / 'bev' / ('%04d.png' % frame))
        with open(self.save_path / 'meta' / ('%04d.json' % frame), 'w') as f:
            json.dump(self.pid_metadata, f, indent=4)

    def destroy(self):
        pass
