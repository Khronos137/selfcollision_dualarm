#!/usr/bin/env python3
"""
Controlador cinemático dual 6DOF - Robotis Manipulator-H
Mismo algoritmo que el script de MATLAB, adaptado a ROS 2.
"""

import math
import numpy as np
import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray
from visualization_msgs.msg import Marker
from geometry_msgs.msg import Point


# ─────────────────────────────────────────────────────────────────────────────
# Geometría del robot (extraída del URDF, igual para ambos brazos)
# ─────────────────────────────────────────────────────────────────────────────
JOINTS = [
    ([0.000,  0.000,  0.126], 'z'),   # joint1
    ([0.000,  0.069,  0.033], 'y'),   # joint2
    ([0.030, -0.0115, 0.264], 'y'),   # joint3
    ([0.195, -0.0575, 0.030], 'x'),   # joint4
    ([0.063,  0.045,  0.000], 'y'),   # joint5
    ([0.123, -0.045,  0.000], 'x'),   # joint6
]


def fk(q):
    """Cinemática directa: posición del efector en frame local de link1."""
    T = np.eye(4)
    for i, (xyz, axis) in enumerate(JOINTS):
        c, s = math.cos(q[i]), math.sin(q[i])
        Ti = np.eye(4)
        Ti[:3, 3] = xyz
        if axis == 'z':
            Ti[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        elif axis == 'y':
            Ti[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        elif axis == 'x':
            Ti[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        T = T @ Ti
    return T[:3, 3].copy()


def jacobian(q):
    """Jacobiano numérico 3×6 por diferencias finitas."""
    h0 = fk(q)
    J = np.zeros((3, 6))
    for i in range(6):
        qp = q.copy()
        qp[i] += 1e-6
        J[:, i] = (fk(qp) - h0) / 1e-6
    return J


def trayectoria(t):
    """Trayectoria deseada del efector (frame local link1). Modificar aquí."""
    r     = 0.1    # Radio del círculo [m]
    omega = 0.2    # Velocidad angular [rad/s]
    return np.array([r * math.cos(omega * t) + 0.18,
                     r * math.sin(omega * t),
                     0.42])


# ─────────────────────────────────────────────────────────────────────────────
# Nodo ROS 2
# ─────────────────────────────────────────────────────────────────────────────
class DualKinematicController(Node):
    def __init__(self):
        super().__init__('dual_kinematic_controller')

        # Parámetros de control (igual que en MATLAB)
        self.To   = 0.1                           # Período de muestreo [s]
        self.tfin = 360.0                          # Duración de la simulación [s]
        self.W    = np.diag([0.55, 0.95, 0.95])   # Ganancia del controlador

        # Estado de los joints (se actualizan desde /joint_states)
        self.q_left   = np.zeros(6)
        self.q_right  = np.zeros(6)
        self.received = False
        self.t        = 0.0
        self.finished = False

        # Datos para gráficas al final
        self.data_t     = []
        self.data_left  = {'hx': [], 'hy': [], 'hz': [],
                           'hxd': [], 'hyd': [], 'hzd': [],
                           'ex': [], 'ey': [], 'ez': [],
                           'q': [[] for _ in range(6)]}
        self.data_right = {'hx': [], 'hy': [], 'hz': [],
                           'hxd': [], 'hyd': [], 'hzd': [],
                           'ex': [], 'ey': [], 'ez': [],
                           'q': [[] for _ in range(6)]}

        # Suscripción a estados de joints
        self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)

        # Publicadores de velocidad
        self.pub_left  = self.create_publisher(Float64MultiArray, '/left_velocity_controller/commands', 10)
        self.pub_right = self.create_publisher(Float64MultiArray, '/right_velocity_controller/commands', 10)

        # Publicadores de markers para RViz
        self.pub_ml_des  = self.create_publisher(Marker, '/left/trajectory/desired_marker', 10)
        self.pub_ml_real = self.create_publisher(Marker, '/left/trajectory/real_marker', 10)
        self.pub_ml_ef   = self.create_publisher(Marker, '/left/trajectory/efector', 10)
        self.pub_mr_des  = self.create_publisher(Marker, '/right/trajectory/desired_marker', 10)
        self.pub_mr_real = self.create_publisher(Marker, '/right/trajectory/real_marker', 10)
        self.pub_mr_ef   = self.create_publisher(Marker, '/right/trajectory/efector', 10)

        # Crear markers (LINE_STRIP para trayectoria, SPHERE para efector)
        def make_marker(frame, ns, mid, mtype, r, g, b):
            m = Marker()
            m.header.frame_id = frame
            m.ns = ns; m.id = mid; m.type = mtype; m.action = Marker.ADD
            m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
            if mtype == Marker.LINE_STRIP:
                m.scale.x = 0.005
            else:
                m.scale.x = m.scale.y = m.scale.z = 0.02
                m.pose.orientation.w = 1.0
            return m

        self.ml_des  = make_marker('left_link1',  'left_desired',  0, Marker.LINE_STRIP, 1.0, 0.0, 0.0)
        self.ml_real = make_marker('left_link1',  'left_real',     1, Marker.LINE_STRIP, 0.0, 1.0, 0.0)
        self.ml_ef   = make_marker('left_link1',  'left_efector',  2, Marker.SPHERE,     0.0, 0.0, 1.0)
        self.mr_des  = make_marker('right_link1', 'right_desired', 3, Marker.LINE_STRIP, 1.0, 0.0, 1.0)
        self.mr_real = make_marker('right_link1', 'right_real',    4, Marker.LINE_STRIP, 0.0, 1.0, 1.0)
        self.mr_ef   = make_marker('right_link1', 'right_efector', 5, Marker.SPHERE,     1.0, 1.0, 0.0)

        # Pre-calcular trayectoria deseada para el marker
        for i in range(int(self.tfin / self.To) + 1):
            hd = trayectoria(i * self.To)
            p = Point(x=hd[0], y=hd[1], z=hd[2])
            self.ml_des.points.append(p)
            self.mr_des.points.append(p)

        # Timer principal a 10 Hz
        self.create_timer(self.To, self.control_loop)

        h0 = fk(np.zeros(6))
        self.get_logger().info(
            f'Controlador DUAL iniciado. FK(q=0)=[{h0[0]:.4f}, {h0[1]:.4f}, {h0[2]:.4f}]')

    # ── Callback: lectura de joints ───────────────────────────────────────────
    def cb_joints(self, msg: JointState):
        for i in range(6):
            nl = f'left_joint{i+1}'
            nr = f'right_joint{i+1}'
            if nl in msg.name:
                self.q_left[i]  = msg.position[msg.name.index(nl)]
            if nr in msg.name:
                self.q_right[i] = msg.position[msg.name.index(nr)]
        self.received = True

    # ── Loop de control principal (equivalente al for-loop de MATLAB) ─────────
    def control_loop(self):
        if not self.received or self.finished:
            return

        # Fin de simulación: parar y graficar
        if self.t >= self.tfin:
            self.finished = True
            self.pub_left.publish(Float64MultiArray(data=[0.0] * 6))
            self.pub_right.publish(Float64MultiArray(data=[0.0] * 6))
            self.get_logger().info(f'Simulación terminada ({self.tfin}s). Mostrando gráficas...')
            self.show_plots()
            return

        hd_k    = trayectoria(self.t)
        hd_next = trayectoria(self.t + self.To)

        # ── Brazo izquierdo ───────────────────────────────────────────────────
        h_left   = fk(self.q_left)
        he_left  = hd_k - h_left
        J_left   = jacobian(self.q_left)
        v_left   = np.linalg.pinv(J_left) @ (hd_next - self.W @ he_left - h_left) / self.To
        self.pub_left.publish(Float64MultiArray(data=v_left.tolist()))

        # ── Brazo derecho ─────────────────────────────────────────────────────
        h_right  = fk(self.q_right)
        he_right = hd_k - h_right
        J_right  = jacobian(self.q_right)
        v_right  = np.linalg.pinv(J_right) @ (hd_next - self.W @ he_right - h_right) / self.To
        self.pub_right.publish(Float64MultiArray(data=v_right.tolist()))

        # ── Guardar datos para gráficas ───────────────────────────────────────
        self.data_t.append(self.t)
        for d, h, he, q in [(self.data_left,  h_left,  he_left,  self.q_left),
                            (self.data_right, h_right, he_right, self.q_right)]:
            d['hx'].append(h[0]);   d['hy'].append(h[1]);   d['hz'].append(h[2])
            d['hxd'].append(hd_k[0]); d['hyd'].append(hd_k[1]); d['hzd'].append(hd_k[2])
            d['ex'].append(he[0]);  d['ey'].append(he[1]);  d['ez'].append(he[2])
            for i in range(6):
                d['q'][i].append(q[i])

        # ── Markers RViz ──────────────────────────────────────────────────────
        now = self.get_clock().now().to_msg()
        self.ml_real.points.append(Point(x=float(h_left[0]),  y=float(h_left[1]),  z=float(h_left[2])))
        self.mr_real.points.append(Point(x=float(h_right[0]), y=float(h_right[1]), z=float(h_right[2])))
        for m, pub in [(self.ml_des,  self.pub_ml_des),  (self.ml_real, self.pub_ml_real),
                       (self.mr_des,  self.pub_mr_des),  (self.mr_real, self.pub_mr_real)]:
            m.header.stamp = now
            pub.publish(m)
        for m, pub, h in [(self.ml_ef, self.pub_ml_ef, h_left), (self.mr_ef, self.pub_mr_ef, h_right)]:
            m.header.stamp = now
            m.pose.position.x = float(h[0])
            m.pose.position.y = float(h[1])
            m.pose.position.z = float(h[2])
            pub.publish(m)

        # ── Log cada 5 s ──────────────────────────────────────────────────────
        if int(self.t * 10) % 50 == 0:
            self.get_logger().info(
                f't={self.t:.1f}s | '
                f'L=[{h_left[0]:.3f},{h_left[1]:.3f},{h_left[2]:.3f}] '
                f'eL=[{he_left[0]:.3f},{he_left[1]:.3f},{he_left[2]:.3f}] | '
                f'R=[{h_right[0]:.3f},{h_right[1]:.3f},{h_right[2]:.3f}] '
                f'eR=[{he_right[0]:.3f},{he_right[1]:.3f},{he_right[2]:.3f}]')

        self.t += self.To

    # ── Gráficas al terminar (equivalente a los plot() de MATLAB) ────────────
    # def show_plots(self):
    #     import matplotlib; matplotlib.use('TkAgg')
    #     import matplotlib.pyplot as plt

    #     t    = np.array(self.data_t)
    #     arms = [('Brazo Izquierdo', self.data_left,  'b', 'c'),
    #             ('Brazo Derecho',   self.data_right, 'r', 'm')]

    #     # Errores de control
    #     fig1, axs = plt.subplots(3, 1, figsize=(12, 8))
    #     fig1.suptitle('Errores de Control')
    #     for arm_name, d, c, _ in arms:
    #         for j, (key, lbl) in enumerate([('ex', 'e_x'), ('ey', 'e_y'), ('ez', 'e_z')]):
    #             axs[j].plot(t, d[key], color=c, label=arm_name, linewidth=1.5)
    #             axs[j].set_ylabel(f'{lbl} [m]'); axs[j].grid(True); axs[j].legend()
    #     axs[2].set_xlabel('Tiempo [s]'); fig1.tight_layout()

    #     # Posición real vs deseada
    #     fig2, axs2 = plt.subplots(3, 2, figsize=(14, 8))
    #     fig2.suptitle('Posición Real vs Deseada')
    #     for col, (arm_name, d, _, _) in enumerate(arms):
    #         for row, (kr, kd, lbl) in enumerate([('hx', 'hxd', 'X'), ('hy', 'hyd', 'Y'), ('hz', 'hzd', 'Z')]):
    #             axs2[row, col].plot(t, d[kd], '--r', label='Deseada', linewidth=1.5)
    #             axs2[row, col].plot(t, d[kr], 'b',   label='Real',    linewidth=1.5)
    #             axs2[row, col].set_title(f'{arm_name} - {lbl}')
    #             axs2[row, col].legend(); axs2[row, col].grid(True)
    #         axs2[2, col].set_xlabel('Tiempo [s]')
    #     fig2.tight_layout()

    #     # Variables articulares
    #     fig3, axs3 = plt.subplots(3, 4, figsize=(16, 9))
    #     fig3.suptitle('Variables Articulares')
    #     for col_offset, (arm_name, d, c, _) in enumerate(arms):
    #         for j in range(6):
    #             ax = axs3[j // 2, col_offset * 2 + j % 2]
    #             ax.plot(t, d['q'][j], color=c, linewidth=1.5)
    #             ax.set_title(f'{arm_name[:3]} q{j+1}'); ax.set_ylabel('[rad]'); ax.grid(True)
    #     for col in range(4):
    #         axs3[2, col].set_xlabel('Tiempo [s]')
    #     fig3.tight_layout()

    #     # Trayectoria 3D
    #     fig4 = plt.figure(figsize=(10, 8))
    #     ax4  = fig4.add_subplot(111, projection='3d')
    #     for arm_name, d, c, c2 in arms:
    #         ax4.plot(d['hx'],  d['hy'],  d['hz'],  color=c,  linewidth=2, label=f'{arm_name} Real')
    #         ax4.plot(d['hxd'], d['hyd'], d['hzd'], '--', color=c2, linewidth=2, label=f'{arm_name} Deseada')
    #     ax4.set_xlabel('X [m]'); ax4.set_ylabel('Y [m]'); ax4.set_zlabel('Z [m]')
    #     ax4.set_title('Trayectorias 3D'); ax4.legend(); ax4.grid(True)

    #     plt.show()


def main(args=None):
    rclpy.init(args=args)
    node = DualKinematicController()
    try:
        rclpy.spin(node)
    except KeyboardInterrupt:
        node.get_logger().info('Interrumpido. Mostrando gráficas...')
        node.pub_left.publish(Float64MultiArray(data=[0.0] * 6))
        node.pub_right.publish(Float64MultiArray(data=[0.0] * 6))
        if len(node.data_t) > 0:
            node.show_plots()
    finally:
        node.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()
