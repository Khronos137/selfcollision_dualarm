import os
import subprocess
import tempfile
import xacro

from launch import LaunchDescription
from launch.actions import RegisterEventHandler, IncludeLaunchDescription
from launch.event_handlers import OnProcessExit
from launch.launch_description_sources import PythonLaunchDescriptionSource
from launch_ros.actions import Node
from ament_index_python.packages import get_package_share_directory

def generate_launch_description():
    package_name = 'dual_manipulator_h'
    pkg_share = get_package_share_directory(package_name)

    # 1. Procesar XACRO
    xacro_path = os.path.join(
        pkg_share, 'description', 'urdf', 'dual_manipulator_v2.xacro'
    )
    doc = xacro.process_file(xacro_path)
    robot_description = doc.toxml()

    # 2. Convertir URDF → SDF usando ign sdf
    urdf_tmp = tempfile.mktemp(suffix='.urdf')
    sdf_tmp = tempfile.mktemp(suffix='.sdf')
    with open(urdf_tmp, 'w') as f:
        f.write(robot_description)
    conv = subprocess.run(['ign', 'sdf', '-p', urdf_tmp], capture_output=True, text=True)
    if conv.returncode == 0 and conv.stdout.strip():
        with open(sdf_tmp, 'w') as f:
            f.write(conv.stdout)
        spawn_args = ['-name', 'dual_manipulator_h', '-file', sdf_tmp]
    else:
        spawn_args = ['-name', 'dual_manipulator_h', '-topic', 'robot_description']

    # 3. Robot State Publisher
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description}]
    )

    # 4. Gazebo Fortress (Ignition Sim 6)
    world_path = os.path.join(pkg_share, 'worlds', 'empty.sdf')
    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(
                get_package_share_directory('ros_gz_sim'),
                'launch', 'gz_sim.launch.py'
            )
        ]),
        launch_arguments={'gz_args': f'-r {world_path}'}.items(),
    )

    # 5. Spawn del robot desde SDF
    spawn_robot_node = Node(
        package='ros_gz_sim',
        executable='create',
        arguments=spawn_args,
        output='screen'
    )

    # 6. Bridge: reloj de Gazebo → ROS
    gz_bridge = Node(
        package='ros_gz_bridge',
        executable='parameter_bridge',
        arguments=['/clock@rosgraph_msgs/msg/Clock[ignition.msgs.Clock'],
        output='screen'
    )

    # 7. Joint State Broadcaster
    spawn_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster',
                   '--controller-manager', '/controller_manager'],
    )

    # 8. Left arm velocity controller
    spawn_left_velocity_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_velocity_controller',
                   '--controller-manager', '/controller_manager'],
    )

    # 9. Right arm velocity controller
    spawn_right_velocity_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_velocity_controller',
                   '--controller-manager', '/controller_manager'],
    )

    # 10. RViz
    rviz_config = os.path.join(pkg_share, 'config', 'control.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
    )

    # Cadena de lanzamiento con eventos
    return LaunchDescription([
        robot_state_publisher_node,
        gazebo,
        gz_bridge,
        spawn_robot_node,

        # Controllers después de que el robot aparezca
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_robot_node,
                on_exit=[spawn_joint_state_broadcaster]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_joint_state_broadcaster,
                on_exit=[spawn_left_velocity_controller]
            )
        ),
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_left_velocity_controller,
                on_exit=[spawn_right_velocity_controller]
            )
        ),

        # RViz se lanza al final (SIN EL CONTROLADOR CUSTOM)
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_right_velocity_controller,
                on_exit=[rviz_node]
            )
        ),
    ])