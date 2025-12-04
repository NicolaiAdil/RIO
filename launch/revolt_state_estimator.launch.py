from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_path_ekf = PathJoinSubstitution([
        FindPackageShare('revolt_state_estimator'),
        'config',
        'revolt_ekf.yaml'
    ])

    return LaunchDescription([
        Node(
            package='revolt_state_estimator',
            executable='revolt_ekf_node',
            name='revolt_ekf_node',
            output='screen',
            parameters=[
                config_path_ekf,
            ],
        ),
    ])
