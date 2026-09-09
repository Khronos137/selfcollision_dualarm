#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np

# Importamos TU librería literal
from libreria_cinematica_der import derCinemDirecta6ManH, derJac6ManH

class MainDerecha(Node):
    def __init__(self):
        super().__init__('main_derecha')

        # 1. Definiciones de Parámetros (Tus valores ORIGINALES de MATLAB)
        self.h = 0.7
        self.b = 0.159 + 0.1 # Ese valor de 0.1 es el offset que le dimos al brazo derecho en el XACRO
        self.l1 = 0.264
        self.l1b = 0.03
        self.l2 = 0.258
        self.l3 = 0.123
        
        # L = [h, b, l1, l1b, l2, l3]
        self.L = [self.h, self.b, self.l1, self.l1b, self.l2, self.l3]
        
        # 2. Control General
        self.pDes = np.array([0.4, -0.5, 0.7])
        self.Ke = 0.1 * np.diag([1, 1, 1])
        
        self.q_right = np.zeros(6)
        self.estado_recibido = False

        # 3. Comunicaciones ROS 2
        self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        
        self.pub_left = self.create_publisher(Float64MultiArray, '/left_velocity_controller/commands', 10)
        self.pub_right = self.create_publisher(Float64MultiArray, '/right_velocity_controller/commands', 10)

        # 4. Bucle a 100 Hz (ts = 0.01 como en tu MATLAB)
        self.ts = 0.01
        self.create_timer(self.ts, self.bucle_principal)
        
        self.get_logger().info("NODO DERECHO INICIADO: Matemática 100% original de MATLAB.")

    def cb_joints(self, msg: JointState):
        """Extrae las posiciones articulares del brazo derecho"""
        for i in range(6):
            nr = f'right_joint{i+1}'
            if nr in msg.name:
                self.q_right[i] = msg.position[msg.name.index(nr)]
        self.estado_recibido = True

    def bucle_principal(self):
        if not self.estado_recibido:
            return

        # 1. Cinemática Directa Original
        pReal = derCinemDirecta6ManH(self.L, self.q_right)

        # 2. Error
        err = self.pDes - pReal

        # 3. Jacobiana Analítica Original
        Jr = derJac6ManH(self.L, self.q_right)

        # 4. Ley de Control (Traducción exacta de MATLAB)
        # uR = pinv(Jr)*(tanh(Ke*err(k,:)'))
        tanh_term = np.tanh(self.Ke @ err)
        uR = np.linalg.pinv(Jr) @ tanh_term

        # q_p(k,:) = uR(1:6)
        q_p = uR[0:6] 

        # 5. Publicar al controlador
        msg_right = Float64MultiArray()
        msg_right.data = q_p.tolist()
        self.pub_right.publish(msg_right)
        
        # Mantenemos el brazo izquierdo quieto
        msg_left = Float64MultiArray(data=[0.0]*6)
        self.pub_left.publish(msg_left)

        # Monitor de datos cada 0.5 segundos reales
        if int(self.get_clock().now().nanoseconds / 1e9 * 100) % 50 == 0:
            self.get_logger().info(f"Err_R: [{err[0]:.4f}, {err[1]:.4f}, {err[2]:.4f}] | pReal_R: [{pReal[0]:.3f}, {pReal[1]:.3f}, {pReal[2]:.3f}]")

def main(args=None):
    rclpy.init(args=args)
    nodo = MainDerecha()
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