#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

# Importamos la librería literal del brazo IZQUIERDO
from libreria_cinematica_izq import cinemDirecta6IzqManH, derJac6IzqManH

class MainIzquierda(Node):
    def __init__(self):
        super().__init__('main_izquierda')

        # 1. Definiciones de Parámetros (Tus valores ORIGINALES de MATLAB)
        self.h = 0.7
        self.b = 0.159 + 0.1 # El offset del torso (0.1) se suma igual
        self.l1 = 0.264
        self.l1b = 0.03
        self.l2 = 0.258
        self.l3 = 0.123
        
        # L = [h, b, l1, l1b, l2, l3]
        self.L = [self.h, self.b, self.l1, self.l1b, self.l2, self.l3]
        
        # 2. Control General (pDes en el lado izquierdo: Y positivo)
        self.pDes = np.array([0.3, 0.3, 0.9])
        self.Ke = 0.1 * np.diag([1, 1, 1])
        
        self.q_left = np.zeros(6)
        self.estado_recibido = False

        # 3. Comunicaciones ROS 2
        self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        
        self.pub_left = self.create_publisher(Float64MultiArray, '/left_velocity_controller/commands', 10)
        self.pub_right = self.create_publisher(Float64MultiArray, '/right_velocity_controller/commands', 10)

        # 4. Bucle a 100 Hz (ts = 0.01 como en tu MATLAB)
        self.ts = 0.01
        self.create_timer(self.ts, self.bucle_principal)
        
        self.get_logger().info("NODO IZQUIERDO INICIADO: Matemática 100% original de MATLAB (Brazo Izquierdo).")

    def cb_joints(self, msg: JointState):
        """Extrae las posiciones articulares del brazo izquierdo"""
        for i in range(6):
            nl = f'left_joint{i+1}'
            if nl in msg.name:
                self.q_left[i] = msg.position[msg.name.index(nl)]
        self.estado_recibido = True

    def bucle_principal(self):
        if not self.estado_recibido:
            return

        # 1. Cinemática Directa Izquierda
        pReal = cinemDirecta6IzqManH(self.L, self.q_left)

        # 2. Error
        err = self.pDes - pReal

        # 3. Jacobiana Analítica Izquierda
        Jl = derJac6IzqManH(self.L, self.q_left)

        # 4. Ley de Control (Traducción exacta de MATLAB)
        tanh_term = np.tanh(self.Ke @ err)
        uL = np.linalg.pinv(Jl) @ tanh_term

        # q_p(k,:) = uL(1:6)
        q_p = uL[0:6] 

        # 5. Publicar al controlador izquierdo
        msg_left = Float64MultiArray()
        msg_left.data = q_p.tolist()
        self.pub_left.publish(msg_left)
        
        # Mantenemos el brazo derecho quieto
        msg_right = Float64MultiArray(data=[0.0]*6)
        self.pub_right.publish(msg_right)

        # Monitor de datos cada 0.5 segundos reales
        if int(self.get_clock().now().nanoseconds / 1e9 * 100) % 50 == 0:
            self.get_logger().info(f"Err_L: [{err[0]:.4f}, {err[1]:.4f}, {err[2]:.4f}] | pReal_L: [{pReal[0]:.3f}, {pReal[1]:.3f}, {pReal[2]:.3f}]")

def main(args=None):
    rclpy.init(args=args)
    nodo = MainIzquierda()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        stop_msg = Float64MultiArray(data=[0.0]*6)
        nodo.pub_left.publish(stop_msg)
        nodo.pub_right.publish(stop_msg)
        nodo.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()