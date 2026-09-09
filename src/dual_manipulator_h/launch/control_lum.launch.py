import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Nodo del Controlador de Lumelsky
    controlador_lumelsky = Node(
        package='dual_manipulator_h',
        executable='mainDualControl_lum.py',
        name='main_dual_brazo_lum',
        output='screen',
        emulate_tty=True # Esto asegura que los colores de los logs (rojo para FATAL, amarillo para WARN) se vean bien en la terminal
    )

    # Empaquetamos y devolvemos la descripción
    return LaunchDescription([
        controlador_lumelsky
    ])