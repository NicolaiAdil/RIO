#!/usr/bin/env python3
"""
Launch file for Revolt State Estimator:

 1) Dynamically loads all static TFs from config/static_transforms.yaml
 2) Immediately starts static_transform_publisher nodes for those frames
 3) Logs a “Please wait” message and then, AFTER A DELAY, launches
    navsat_transform_node and ekf_node.
"""

import os
import yaml

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import TimerAction, LogInfo, ExecuteProcess
from launch.substitutions import PathJoinSubstitution
from launch_ros.actions import Node
from launch_ros.substitutions import FindPackageShare

def load_static_tfs():
    pkg_share = get_package_share_directory('revolt_state_estimator')
    yaml_path = os.path.join(pkg_share, 'config', 'internal_sensors_static_tf.yaml')
    with open(yaml_path, 'r') as f:
        entries = yaml.safe_load(f).get('internal_sensors_static_tf', [])

    nodes = []
    for e in entries:
        px, py, pz = e['translation']
        rr, rp, ry = e['rotation_rpy']
        parent = e['parent_frame']
        child  = e['child_frame']

        args = [
            '--x', str(px), '--y', str(py), '--z', str(pz),
            '--roll', str(rr), '--pitch', str(rp), '--yaw', str(ry),
            '--frame-id', parent, '--child-frame-id', child,
        ]
        nodes.append(Node(
            package='tf2_ros', executable='static_transform_publisher',
            name=f'static_{parent}_to_{child}', output='screen',
            arguments=args
        ))
    return nodes

def generate_launch_description():
    # --- locate configs ---
    pkg_share  = FindPackageShare('revolt_state_estimator')
    navsat_cfg = PathJoinSubstitution([pkg_share, 'config', 'navsat_transform.yaml'])
    ekf_cfg    = PathJoinSubstitution([pkg_share, 'config', 'ekf.yaml'])

    # --- 1) STATIC TFs ---
    static_tfs = load_static_tfs()

    # !!! WARNING !!!
    # We must wait at least 1.0 second here to give static TFs time
    # to register in the TF tree *before* navsat_transform_node & ekf_node start.
    # If you shorten this below 1 second, the EKF may silently drop data!
    DELAY_BEFORE_FUSION = 10.0  # !!! DO NOT TOUCH, MUST BE 1.0 !!!

    # --- 2) Delayed launch of sensor–fusion nodes ---
    navsat_and_ekf = TimerAction(
        period=DELAY_BEFORE_FUSION,
        actions=[

            # --- navsat_transform_node ---
            Node(
                package='robot_localization',
                executable='navsat_transform_node',
                name='navsat_transform_node',
                output='screen',
                parameters=[navsat_cfg],
                remappings=[
                    # hardware topics → node’s expected topics
                    ('gps/fix',           '/fix'),              # real GNSS → gps/fix
                    ('imu',               '/imu/data'),         # IMU driver → imu
                    ('odometry/filtered', '/odometry/filtered'),# seed EKF pose → odometry/filtered
                ],
                arguments=[
                    '--ros-args',
                    '--log-level', 'navsat_transform_node:=DEBUG',
                ],
            ),

            # --- ekf_node ---
            Node(
                package='robot_localization',
                executable='ekf_node',
                name='ekf_filter_node', 
                output='screen',
                parameters=[ekf_cfg],
                arguments=[
                    '--ros-args',
                    '--log-level', 'ekf_node:=DEBUG',
                ],
            ),
        ]
    )

    # After navsat_transform_node is up, give it 1 s, then tell it “Zone 32 N”
    set_utm_zone = TimerAction(
        period=1.0,  # wait for the node to come up
        actions=[
            LogInfo(msg="[revolt_state_estimator] Setting UTM zone to 32N…"),
            ExecuteProcess(
                cmd=[
                    'ros2', 'service', 'call',
                    '/navsat_transform_node/setUTMZone',
                    'robot_localization/srv/SetUTMZone',
                    "utm_zone: 32"
                ],
                output='screen'
            )
        ]
    )

    # --- assemble everything ---
    return LaunchDescription([
        *static_tfs,
        navsat_and_ekf,
        set_utm_zone,
    ])
