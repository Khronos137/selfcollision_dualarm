from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node


def generate_launch_description():
    return LaunchDescription([
        DeclareLaunchArgument('trajectory_file', default_value='trayectorias_50.yaml'),
        DeclareLaunchArgument('csv_file', default_value='resultados_benchmark.csv'),
        DeclareLaunchArgument('contacts_file', default_value='contactos_benchmark.csv'),
        DeclareLaunchArgument('calibration_s', default_value='3.0'),
        DeclareLaunchArgument('n_trials', default_value='0', description='0 = all'),
        Node(
            package='dual_manipulator_benchmark',
            executable='benchmark_director.py',
            name='benchmark_director',
            output='screen',
            emulate_tty=True,
            parameters=[{'trajectory_file': LaunchConfiguration('trajectory_file'),
                         'csv_file': LaunchConfiguration('csv_file'),
                         'contacts_file': LaunchConfiguration('contacts_file'),
                         'calibration_s': LaunchConfiguration('calibration_s'),
                         'n_trials': LaunchConfiguration('n_trials')}],
        ),
    ])