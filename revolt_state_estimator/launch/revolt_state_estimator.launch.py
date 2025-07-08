from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    # 1) Locate this package’s share directory so we can find our YAMLs
    pkg_share = FindPackageShare('revolt_state_estimator')

    # 2) Point to the two config files
    navsat_cfg = PathJoinSubstitution([pkg_share, 'config', 'navsat.yaml'])
    ekf_cfg   = PathJoinSubstitution([pkg_share, 'config', 'ekf.yaml'])

    return LaunchDescription([

        # =============================================================================
        # STATIC TRANSFORMS
        # =============================================================================

        # 1) map → odom 
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_map_to_odom',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'map',
                '--child-frame-id', 'odom',
            ],
        ),

        # 2) odom → base_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_odom_to_base_link',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'odom',
                '--child-frame-id', 'base_link',
            ],
        ),

        # 3) base_link → imu_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_base_link_to_imu_link',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'imu_link',
            ],
        ),

        # 4) base_link → gps_link
        Node(
            package='tf2_ros',
            executable='static_transform_publisher',
            name='static_base_link_to_gps_link',
            output='screen',
            arguments=[
                '--x', '0', '--y', '0', '--z', '0',
                '--roll', '0', '--pitch', '0', '--yaw', '0',
                '--frame-id', 'base_link',
                '--child-frame-id', 'gps_link',
            ],
        ),

        # =============================================================================
        # navsat_transform_node
        # =============================================================================

        Node(
            package='robot_localization',
            executable='navsat_transform_node',
            name='navsat_transform_node',
            output='screen',
            parameters=[navsat_cfg],
            remappings=[
                # remap your raw sensor topics into what the node expects:
                ('gps/fix',     '/fix'),      # your GNSS driver → gps/fix
                ('imu',         '/imu/data'), # your IMU driver → imu
                ('odometry/filtered',     '/odometry/filtered'),  # seed odom → /odometry/filtered
            ],
            arguments=[
                '--ros-args',
                '--log-level', 'navsat_transform_node:=DEBUG'
            ],
        ),


        # =============================================================================
        # ekf_node
        # =============================================================================

        Node(
            package='robot_localization',
            executable='ekf_node',
            name='ekf_node',
            output='screen',
            parameters=[ekf_cfg],
            arguments=[
                '--ros-args',
                '--log-level', 'ekf_node:=DEBUG'
            ],
        ),

    ])
