#!/usr/bin/env python3
import os
from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument, TimerAction, IncludeLaunchDescription
from launch.substitutions import LaunchConfiguration, PathJoinSubstitution
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():

    host = LaunchConfiguration('host', default='localhost')
    port = LaunchConfiguration('port', default='2000')
    town = LaunchConfiguration('town', default='Town01')
    timeout = LaunchConfiguration('timeout', default='10.0')
    passive = LaunchConfiguration('passive', default='false')
    ego_role_name = LaunchConfiguration('ego_vehicle_role_name', default='hero')

    infer = Node(
        package='vino_infer',
        executable='infer_node',
        name='infer_node',
        output='screen'
    )

    return LaunchDescription([
        infer,
    ])