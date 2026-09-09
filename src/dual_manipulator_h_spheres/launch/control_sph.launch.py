import os
from launch import LaunchDescription
from launch_ros.actions import Node

def generate_launch_description():
    
    # Nodo del Controlador Estricto de Lei et al. (Spheres)
    controlador_spheres = Node(
        package='dual_manipulator_h_spheres',
        executable='mainDualControl_sph.py',
        name='main_dual_brazo_sph',
        output='screen',
        emulate_tty=True # Habilita la telemetría en colores (rojo para los fallos matemáticos)
    )

    # Empaquetamos y devolvemos la descripción
    return LaunchDescription([
        controlador_spheres
    ])