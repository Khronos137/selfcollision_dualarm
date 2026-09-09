import os
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

    # 1. Procesar XACRO (URDF)
    xacro_path = os.path.join(
        pkg_share, 'description', 'urdf', 'dual_manipulator_v2.xacro'
    )
    doc = xacro.process_file(xacro_path)
    robot_description = doc.toxml()

    # 2. Robot State Publisher (Aseguramos que use el tiempo de simulación)
    robot_state_publisher_node = Node(
        package='robot_state_publisher',
        executable='robot_state_publisher',
        output='screen',
        parameters=[{'robot_description': robot_description, 'use_sim_time': True}]
    )

    # 3. Lanzar Gazebo Classic con tu empty.world
    gazebo_ros_dir = get_package_share_directory('gazebo_ros')
    world_path = os.path.join(pkg_share, 'worlds', 'empty.world')

    gazebo = IncludeLaunchDescription(
        PythonLaunchDescriptionSource([
            os.path.join(gazebo_ros_dir, 'launch', 'gazebo.launch.py')
        ]),
        launch_arguments={
            'world': world_path,
            'pause': 'false'
        }.items()
    )

    # 4. Spawn del robot usando gazebo_ros
    spawn_robot_node = Node(
        package='gazebo_ros',
        executable='spawn_entity.py',
        arguments=['-topic', 'robot_description', '-entity', 'dual_manipulator_h'],
        output='screen'
    )

    # 5. Joint State Broadcaster
    spawn_joint_state_broadcaster = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['joint_state_broadcaster'],
    )

    # 6. Left arm velocity controller
    spawn_left_velocity_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['left_velocity_controller'],
    )

    # 7. Right arm velocity controller
    spawn_right_velocity_controller = Node(
        package='controller_manager',
        executable='spawner',
        arguments=['right_velocity_controller'],
    )

    # 8. RViz
    rviz_config = os.path.join(pkg_share, 'config', 'control.rviz')
    rviz_node = Node(
        package='rviz2',
        executable='rviz2',
        name='rviz2',
        output='screen',
        arguments=['-d', rviz_config],
        parameters=[{'use_sim_time': True}]
    )

    # Cadena de lanzamiento con eventos
    return LaunchDescription([
        robot_state_publisher_node,
        gazebo,
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

        # RViz se lanza al final
        RegisterEventHandler(
            event_handler=OnProcessExit(
                target_action=spawn_right_velocity_controller,
                on_exit=[rviz_node]
            )
        ),
    ])