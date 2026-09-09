#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
import numpy as np

# Importamos nuestras librerías limpias
from cinematica import obtener_puntos_globales
from colisiones import revisar_estado_seguridad

class DetectorColisionesNode(Node):
    def __init__(self):
        super().__init__('detector_colisiones')
        
        self.q_left  = np.zeros(6)
        self.q_right = np.zeros(6)
        
        self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        
        # Evaluamos a 20 Hz (cada 0.05 segundos)
        self.create_timer(0.05, self.bucle_seguridad)
        self.get_logger().info("Nodo Detector de Colisiones Iniciado.")

    def cb_joints(self, msg: JointState):
        """Actualiza la lectura de los encoders."""
        for i in range(6):
            nl, nr = f'left_joint{i+1}', f'right_joint{i+1}'
            if nl in msg.name: self.q_left[i]  = msg.position[msg.name.index(nl)]
            if nr in msg.name: self.q_right[i] = msg.position[msg.name.index(nr)]

    def bucle_seguridad(self):
        """El corazón de la vigilancia."""
        # 1. Obtener puntos espaciales (Cinemática Directa)
        puntos = obtener_puntos_globales(self.q_left, self.q_right)
        
        # 2. Verificar matemática de colisiones
        peligro, reportes = revisar_estado_seguridad(puntos)
        
        # 3. Acción
        if peligro:
            self.get_logger().error("¡COLISIÓN INMINENTE DETECTADA!")
            for r in reportes:
                self.get_logger().warn(f" -> {r}")
            # AQUI A FUTURO LLAMAREMOS AL FRENO DE EMERGENCIA

def main(args=None):
    rclpy.init(args=args)
    nodo = DetectorColisionesNode()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()