#!/usr/bin/env python3

import os

import rclpy
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Bool, Float32
from sensor_msgs.msg import Image, Imu, NavSatFix, PointCloud2
from message_filters import Subscriber, ApproximateTimeSynchronizer
from carla_msgs.msg import CarlaRoute, CarlaEgoVehicleControl, CarlaSpeedometer
from ament_index_python.packages import get_package_share_directory

import openvino as ov
import numpy as np
from .pilot import Pilot

class InferNode(Node):
    def __init__(self):
        super().__init__('infer_node')
        
        # load model
        package_shared_dir = get_package_share_directory('vino_infer')
        self.declare_parameter(
            'model_path',
            value=os.path.join(package_shared_dir, 'weights', 'model_fp32.xml')
        )
        self.model_path = self.get_parameter('model_path').value
        if os.path.exists(self.model_path):
            self.xml_path = os.path.splitext(self.model_path)[0] + '.xml'
            self.bin_path = os.path.splitext(self.model_path)[0] + '.bin'
        else:
            self.get_logger().warning(
                f'Model path {self.model_path} does not exist'
            )
            try:
                self.xml_path = os.path.join(
                    package_shared_dir, 'weights', 'model_fp32.xml'
                )
                self.bin_path = os.path.join(
                    package_shared_dir, 'weights', 'model_fp32.bin'
                )
            except:
                self.get_logger().fatal(f'Failed to load model')
                raise FileNotFoundError(f'Failed to load model')

        # load data interface
        self._rgb_sub = Subscriber(self, Image, '/carla/hero/rgb/image')
        self._gps_sub = Subscriber(self, NavSatFix, '/carla/hero/gps')
        self._speed_sub = Subscriber(self, CarlaSpeedometer, '/carla/hero/speed')
        self._imu_sub = Subscriber(self, Imu, '/carla/hero/imu')
        self._lidar_sub = Subscriber(self, PointCloud2, '/carla/hero/lidar')

        self._synchronizer = ApproximateTimeSynchronizer(
            [self._rgb_sub, self._gps_sub, self._speed_sub,
             self._imu_sub, self._lidar_sub],
            1, # my computer is too slow to provide the sim data in time, so I have to do this.
            3 # ensure timely reaction
        )
        self._synchronizer.registerCallback(self.infer_once)

        self._hero_status = self.create_publisher(
            Bool,
            '/carla/hero/status',
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
        )

        self._global_plan_sub = self.create_subscription(
            CarlaRoute,
            '/carla/hero/global_plan',
            self._global_plan_callback,
            10
        )

        self._ctrl_pub = self.create_publisher(
            CarlaEgoVehicleControl,
            '/carla/hero/vehicle_control_cmd',
            qos_profile=QoSProfile(
                depth=1,
                durability=DurabilityPolicy.TRANSIENT_LOCAL
            )
        )

        # load algorithm
        core = ov.Core()
        self.model = core.compile_model(model=self.xml_path)
        self.infer_interface = self.model.create_infer_request()
        self.pilot = Pilot()

        self.get_logger().info('InferNode initialized')
        self._hero_status.publish(Bool(data=True))
        
    def _global_plan_callback(self, global_plan: CarlaRoute):
        self.get_logger().info(f'Global plan received: {global_plan}')
        self.pilot.set_global_plan(global_plan)

    def infer_once(self, _rgb_msg:Image, _gps_msg:NavSatFix,
                   _speed_msg:Float32, _imu_msg:Imu, _lidar_msg:PointCloud2):
        self.get_logger().info('Infer once triggered')
        rgb, state, target_point, lidar, metadata = self.pilot.preprocess(
            _rgb_msg, _speed_msg, _gps_msg, _lidar_msg, _imu_msg
        )
        inputs = {
            'rgb': rgb,
            'state': state,
            'target_point': target_point,
            'pcl': lidar
        }
        self.get_logger().info('Preprocess completed.')
        outputs = self.infer_interface.infer(inputs=inputs)
        self.get_logger().info('Infer completed.')
        control = self.pilot.postprocess(outputs, metadata['next_cmd'], metadata['speed'], target_point)
        self.get_logger().info(f'Postprocess completed. {control}')
        ctrl = CarlaEgoVehicleControl(**control)
        self._ctrl_pub.publish(ctrl)

def main(args=None):
    rclpy.init(args=args)
    infer_node = InferNode()

    rclpy.spin(infer_node)
    infer_node.destroy_node()
    rclpy.shutdown()

if __name__ == '__main__':
    main()