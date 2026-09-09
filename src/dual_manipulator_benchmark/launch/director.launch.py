import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Nodo del Director (El Árbitro del Benchmark)
    director_node = Node(
        package='dual_manipulator_benchmark',
        executable='benchmark_director.py',
        name='benchmark_director',
        output='screen',
        emulate_tty=True # Para ver los emojis y colores en la terminal
    )

    return LaunchDescription([
        director_node
    ])