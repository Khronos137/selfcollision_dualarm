from launch import LaunchDescription
from launch.actions import DeclareLaunchArgument
from launch.substitutions import LaunchConfiguration
from launch_ros.actions import Node

# Gains left at -1.0 keep the value of Table 2 / the selected variant.
GANANCIAS = ['d_safe', 'd_crit', 'rho_L', 'rho_R', 'eta_vort_L', 'eta_vort_R']


def generate_launch_description():
    args = [
        DeclareLaunchArgument('variant', default_value='full',
                              description='full | no_weighting | no_vortex | '
                                          'no_weighting_no_vortex | swapped | symmetric'),
        DeclareLaunchArgument('etiqueta', default_value='auto',
                              description='label for CSV/logs (auto = from the variant)'),
        DeclareLaunchArgument('log_dir', default_value='~/benchmark_logs'),
        DeclareLaunchArgument('publicar_rviz', default_value='false'),
    ] + [DeclareLaunchArgument(g, default_value='-1.0') for g in GANANCIAS]

    parametros = {'variant': LaunchConfiguration('variant'),
                  'etiqueta': LaunchConfiguration('etiqueta'),
                  'log_dir': LaunchConfiguration('log_dir'),
                  'publicar_rviz': LaunchConfiguration('publicar_rviz')}
    parametros.update({g: LaunchConfiguration(g) for g in GANANCIAS})

    return LaunchDescription(args + [
        Node(
            package='dual_manipulator_h',
            executable='mainDualControl_lum.py',
            name='main_dual_brazo_lum',
            output='screen',
            emulate_tty=True,
            parameters=[parametros],
        ),
    ])