#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
import numpy as np
import math

# Importamos nuestros módulos (AHORA USANDO EL CORE SIMBÓLICO)
from cinematica import obtener_puntos_globales
from colisiones import revisar_estado_seguridad
from control_core_simbolic import calcular_velocidad_articular
from visualizador import VisualizadorRViz 


# =========================================================
# ZONA DE EDICIÓN DEL EXPERIMENTO
# =========================================================
def trayectoria_experimento(t, brazo='left'):
    """Círculos opuestos en el plano HORIZONTAL global"""
    r = 0.15      
    omega = 0.2   
    
    # 1. Calculamos el círculo horizontal como lo imaginamos en el mundo real
    x_global = r * math.cos(omega * t) + 0.18
    
    # El izquierdo va hacia un lado, el derecho hacia el opuesto para chocar
    if brazo == 'left':
        y_global = r * math.sin(omega * t)
    else:
        y_global = r * math.sin(-omega * t)
        
    z_global = 0.22 # Altura global constante que deseamos
    
    # 2. TRADUCCIÓN AL MARCO LOCAL DEL HOMBRO (Rotado 90 grados)
    # Matemáticamente mapeamos los ejes globales a los ejes del motor base
    if brazo == 'left':
        # Hombro izquierdo rotado -90 en X y 180 en Z
        x_local = x_global
        y_local = -z_global  # La altura global define el Y local
        z_local = -y_global  # El movimiento lateral define el Z local
    else:
        # Hombro derecho rotado +90 en X y 180 en Z
        x_local = x_global
        y_local = z_global   # La altura global define el Y local
        z_local = y_global   # El movimiento lateral define el Z local

    return np.array([x_local, y_local, z_local])
# =========================================================


class DualManipulatorBrain(Node):
    def __init__(self):
        super().__init__('dual_manipulator_brain')

        # --- PARÁMETROS OPTIMIZADOS PARA CONTROL SIMBÓLICO ---
        self.dt = 0.01  # Frecuencia de publicación (100 Hz), ya no afecta la matemática
        self.K = 4.5    # Ganancia para la ley de control (K * tanh(error))
        self.t = 0.0
        
        self.q_left = np.zeros(6)
        self.q_right = np.zeros(6)
        self.estado_recibido = False

        self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.pub_left = self.create_publisher(Float64MultiArray, '/left_velocity_controller/commands', 10)
        self.pub_right = self.create_publisher(Float64MultiArray, '/right_velocity_controller/commands', 10)

        # INYECTAMOS LA TRAYECTORIA AL VISUALIZADOR
        # (Se mantiene self.dt para RViz, ya que necesita saber cada cuánto graficar)
        self.viz = VisualizadorRViz(self, trayectoria_experimento, To=self.dt, tfin=60.0)

        self.create_timer(self.dt, self.bucle_principal)
        self.get_logger().info(f"CEREBRO INICIADO: Control Simbólico a {1/self.dt:.0f} Hz.")

    def cb_joints(self, msg: JointState):
        for i in range(6):
            nl, nr = f'left_joint{i+1}', f'right_joint{i+1}'
            if nl in msg.name: self.q_left[i] = msg.position[msg.name.index(nl)]
            if nr in msg.name: self.q_right[i] = msg.position[msg.name.index(nr)]
        self.estado_recibido = True

    def bucle_principal(self):
        if not self.estado_recibido: return

        # 1. VIGILANCIA DE COLISIONES
        puntos_3d = obtener_puntos_globales(self.q_left, self.q_right)
        peligro, reportes = revisar_estado_seguridad(puntos_3d)

        # Preparación para inyectar el espacio nulo
        q_null_l = None
        q_null_r = None

        if peligro:
            self.get_logger().error("¡COLISIÓN INMINENTE!")
            for r in reportes: self.get_logger().warn(f"Causa: {r}")
            # Próximamente: q_null_l = calcular_gradiente_repulsion(...)
        
        # 2. GENERACIÓN DE PUNTOS DE TRAYECTORIA DESDE EL CEREBRO
        hd_l = trayectoria_experimento(self.t, 'left')
        hd_r = trayectoria_experimento(self.t, 'right')

        # 3. CÁLCULO DE CONTROL (Core Simbólico a 100 Hz)
        vl, err_l, h_real_l = calcular_velocidad_articular(
            self.q_left, hd_l, self.K, prefix='left', q_null=q_null_l
        )
        vr, err_r, h_real_r = calcular_velocidad_articular(
            self.q_right, hd_r, self.K, prefix='right', q_null=q_null_r
        )

        # 4. ACCIÓN
        self.pub_left.publish(Float64MultiArray(data=vl.tolist()))
        self.pub_right.publish(Float64MultiArray(data=vr.tolist()))

        # 5. VISUALIZACIÓN RVIZ
        tiempo_actual = self.get_clock().now().to_msg()
        self.viz.publicar(tiempo_actual)
        self.viz.publicar_cilindros_colision(tiempo_actual, puntos_3d)
        
        if int(self.t * 100) % 50 == 0:  # Log cada 0.5s
            self.get_logger().info(
                f"t={self.t:.2f}s | "
                f"h_L: [{h_real_l[0]:.3f}, {h_real_l[1]:.3f}, {h_real_l[2]:.3f}] | "
                f"h_R: [{h_real_r[0]:.3f}, {h_real_r[1]:.3f}, {h_real_r[2]:.3f}]"
            )
        self.t += self.dt

def main(args=None):
    rclpy.init(args=args)
    nodo = DualManipulatorBrain()
    try: rclpy.spin(nodo)
    except KeyboardInterrupt: pass
    finally:
        nodo.pub_left.publish(Float64MultiArray(data=[0.0]*6))
        nodo.pub_right.publish(Float64MultiArray(data=[0.0]*6))
        nodo.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()