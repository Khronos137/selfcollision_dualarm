from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('log_dir', default_value='~/benchmark_logs'),
        DeclareLaunchArgument('publicar_rviz', default_value='false'),
        Node(
            package='dual_manipulator_h_spheres',
            executable='mainDualControl_sph.py',
            name='main_dual_brazo_sph',
            output='screen',
            emulate_tty=True,
            parameters=[{'log_dir': LaunchConfiguration('log_dir'),
                         'publicar_rviz': LaunchConfiguration('publicar_rviz')}],
        ),
    ])
