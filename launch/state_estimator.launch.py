from launch import LaunchDescription
from launch_ros.actions import Node
from launch.substitutions import PathJoinSubstitution
from launch_ros.substitutions import FindPackageShare

def generate_launch_description():
    config_path_ekf = PathJoinSubstitution([
        FindPackageShare('state_estimator'),
        'config',
        'parameters.yaml'
    ])

    return LaunchDescription([
        Node(
            package='state_estimator',
            executable='state_estimator_node',
            name='state_estimator_node',
            output='screen',
            parameters=[
                config_path_ekf,
            ],
        ),
    ])
