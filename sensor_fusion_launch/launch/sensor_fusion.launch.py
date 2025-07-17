#!/usr/bin/env python3
import os

from ament_index_python.packages import get_package_share_directory
from launch import LaunchDescription
from launch.actions import IncludeLaunchDescription
from launch.launch_description_sources import PythonLaunchDescriptionSource

def generate_launch_description():
    # ReVolt State Estimate
    revolt_state_estimator_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('SensorFusion'),
                'launch',
                'revolt_state_estimator.launch.py'
            )
        )
    )

    # Heartbeat
    heartbeat_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('heartbeat'),
                'launch',
                'heartbeat.launch.py'
            )
        )
    )

    # Xsens IMU
    xsens_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('xsens'),
                'launch',
                'xsens.launch.py'
            )
        )
    )

    # GNSS (NMEA) driver
    nmea_navsat_gnss_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('nmea_navsat'),
                'launch',
                'nmea_serial_driver.launch.py'
            )
        )
    )

    # AIS
    ais_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('ais'),
                'launch',
                'ais.launch.py'
            )
        )
    )

    # Velodyne LiDAR
    velodyne_lidar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('velodyne'),
                'launch',
                'velodyne-all-nodes-VLP16-launch.py'
            )
        )
    )

    # Furuno Radar
    radar_launch = IncludeLaunchDescription(
        PythonLaunchDescriptionSource(
            os.path.join(
                get_package_share_directory('radar'),
                'launch',
                'radar.launch.py'
            )
        )
    )

    return LaunchDescription([
        actuators_launch,
        heartbeat_launch,
        xsens_launch,
        nmea_navsat_gnss_launch,
        ais_launch,
        velodyne_lidar_launch,
        radar_launch,
    ])
