"""
ROS 2 node shared by the proposed controller and the baseline.

SHARED MODULE: this file must be identical in
    dual_manipulator_h/scripts/ and dual_manipulator_h_spheres/scripts/

It contains everything that must be IDENTICAL for a fair comparison:
trajectory reception, trial termination, homing, metrics, timing and logging.
Each controller only provides an object with
    reiniciar()
    calcular(q_L, q_R, pdes_L, vdes_L, pdes_R, vdes_R) -> (qdot_L, qdot_R, info)
"""
import os
import json
import time
import numpy as np

from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String

from dual_manipulator_benchmark.msg import BezierTrajectory
from libreria_cinematica_der import derCinemDirecta6ManH
from libreria_cinematica_izq import cinemDirecta6IzqManH
from cinematica_puntos import L_ROBOT
from benchmark_common import (DT, TOL_BAJA, evaluar_bezier, MetricasBrazo,
                              estadisticas_tiempo, RegistroEnsayo)


class NodoBenchmark(Node):

    def __init__(self, nombre_nodo, etiqueta_algoritmo, controlador):
        super().__init__(nombre_nodo)
        self.declare_parameter('log_dir', os.path.expanduser('~/benchmark_logs'))
        self.declare_parameter('publicar_rviz', False)
        self.etiqueta = etiqueta_algoritmo
        self.controlador = controlador
        self.log_dir = os.path.expanduser(str(self.get_parameter('log_dir').value))
        self.publicar_rviz = bool(self.get_parameter('publicar_rviz').value)
        self.L = L_ROBOT

        # --- state ---
        self.q_L = np.zeros(6); self.q_R = np.zeros(6)
        self.qd_L = np.zeros(6); self.qd_R = np.zeros(6)
        self.js_stamp = np.nan
        self.t_js = None
        self.estado_recibido = False
        self.modo = 'IDLE'            # IDLE | BENCHMARK | HOME  (IDLE and HOME hold q = 0)
        self.ciclos_home = 0
        self.t_home = 0.0
        self.traj_L = None; self.traj_R = None
        self.trial_id = -1
        self.t = 0.0
        self.t_cb_prev = None

        # --- ROS ---
        self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        self.create_subscription(BezierTrajectory, '/benchmark/left_arm_traj', self.cb_traj_left, 10)
        self.create_subscription(BezierTrajectory, '/benchmark/right_arm_traj', self.cb_traj_right, 10)
        self.create_subscription(String, '/benchmark/logger_cmd', self.cb_logger_cmd, 10)
        self.pub_left = self.create_publisher(Float64MultiArray, '/left_velocity_controller/commands', 10)
        self.pub_right = self.create_publisher(Float64MultiArray, '/right_velocity_controller/commands', 10)
        self.pub_status = self.create_publisher(String, '/benchmark/status', 10)
        self.pub_metricas = self.create_publisher(String, '/benchmark/metricas', 10)
        self.create_timer(DT, self.bucle_principal)
        self.get_logger().info(f'[{self.etiqueta}] controller ready. Logs -> {self.log_dir}')

    # -----------------------------------------------------------------
    # CALLBACKS
    # -----------------------------------------------------------------
    def cb_joints(self, msg: JointState):
        for i in range(6):
            for nombre, q, qd in ((f'left_joint{i+1}', self.q_L, self.qd_L),
                                  (f'right_joint{i+1}', self.q_R, self.qd_R)):
                if nombre in msg.name:
                    k = msg.name.index(nombre)
                    q[i] = msg.position[k]
                    if len(msg.velocity) > k:
                        qd[i] = msg.velocity[k]
        self.js_stamp = msg.header.stamp.sec + 1e-9 * msg.header.stamp.nanosec
        self.t_js = time.perf_counter()
        self.estado_recibido = True

    def cb_logger_cmd(self, msg: String):
        if msg.data.startswith('START_'):
            self.trial_id = int(msg.data.split('_')[1])

    def _recibir_traj(self, msg, brazo):
        if msg.t_total < 0.0:              # director requests homing
            self.modo = 'HOME'
            self.ciclos_home = 0
            self.t_home = 0.0
            return
        P = np.array([msg.p0, msg.p1, msg.p2, msg.p3], dtype=float)
        if brazo == 'L':
            self.traj_L = (P, float(msg.t_total))
        else:
            self.traj_R = (P, float(msg.t_total))
        if self.traj_L is not None and self.traj_R is not None:
            self._iniciar_ensayo()

    def cb_traj_left(self, msg):
        self._recibir_traj(msg, 'L')

    def cb_traj_right(self, msg):
        self._recibir_traj(msg, 'R')

    def _iniciar_ensayo(self):
        self.t = 0.0
        self.met_L = MetricasBrazo(self.traj_L[1])
        self.met_R = MetricasBrazo(self.traj_R[1])
        self.registro = RegistroEnsayo()
        self.pipeline_ms, self.callback_ms, self.period_ms = [], [], []
        self.state_age_ms, self.col_ms = [], []
        self.controlador.reiniciar()
        self.modo = 'BENCHMARK'

    def publicar(self, uL, uR):
        self.pub_left.publish(Float64MultiArray(data=[float(x) for x in uL]))
        self.pub_right.publish(Float64MultiArray(data=[float(x) for x in uR]))

    # -----------------------------------------------------------------
    # MAIN LOOP
    # -----------------------------------------------------------------
    def bucle_principal(self):
        t_cb0 = time.perf_counter()
        if not self.estado_recibido:
            return
        if self.modo != 'BENCHMARK':
            # IDLE and HOME: actively hold the home posture q = 0 (a zero velocity
            # command does not hold the arms against gravity in Gazebo Classic).
            self._retener_home()
            self.t_cb_prev = None
            return

        PL, TL = self.traj_L
        PR, TR = self.traj_R
        xL = cinemDirecta6IzqManH(self.L, self.q_L)
        xR = derCinemDirecta6ManH(self.L, self.q_R)
        eL_goal = float(np.linalg.norm(PL[3] - xL))
        eR_goal = float(np.linalg.norm(PR[3] - xR))

        # Trial termination (identical for both algorithms)
        if eL_goal < TOL_BAJA and eR_goal < TOL_BAJA:
            self._finalizar_ensayo(eL_goal, eR_goal)
            return
        if self.t >= 2.0 * TL or self.t >= 2.0 * TR:
            self._finalizar_ensayo(eL_goal, eR_goal)
            return

        period = np.nan if self.t_cb_prev is None else (t_cb0 - self.t_cb_prev) * 1e3
        self.t_cb_prev = t_cb0
        state_age = np.nan if self.t_js is None else (t_cb0 - self.t_js) * 1e3

        # ---------------- control pipeline (timed) ----------------
        t0 = time.perf_counter()
        pdL, vdL = evaluar_bezier(self.t, TL, PL)
        pdR, vdR = evaluar_bezier(self.t, TR, PR)
        uL, uR, info = self.controlador.calcular(self.q_L.copy(), self.q_R.copy(), pdL, vdL, pdR, vdR)
        self.publicar(uL, uR)
        pipeline = (time.perf_counter() - t0) * 1e3
        # ------------------------------------------------------------

        self.met_L.paso(self.t, pdL - xL, eL_goal, uL, self.qd_L, info['clear_L'])
        self.met_R.paso(self.t, pdR - xR, eR_goal, uR, self.qd_R, info['clear_R'])
        self.pipeline_ms.append(pipeline)
        self.col_ms.append(info['collision_ms'])
        self.period_ms.append(period)
        self.state_age_ms.append(state_age)
        self.registro.agregar(t=self.t, q_L=self.q_L.copy(), q_R=self.q_R.copy(),
                              qdot_meas_L=self.qd_L.copy(), qdot_meas_R=self.qd_R.copy(),
                              qdot_cmd_L=uL, qdot_cmd_R=uR, x_L=xL, x_R=xR, xdes_L=pdL, xdes_R=pdR,
                              clear_L=info['clear_L'], clear_R=info['clear_R'],
                              scale_L=info['scale_L'], scale_R=info['scale_R'],
                              pipeline_ms=pipeline, collision_ms=info['collision_ms'],
                              period_ms=period, state_age_ms=state_age, js_stamp=self.js_stamp)
        self.t += DT

        if self.publicar_rviz:
            self.visualizar(info)

        # Full callback time, including metrics, logging and (optional) RViz
        cb = (time.perf_counter() - t_cb0) * 1e3
        self.callback_ms.append(cb)
        self.registro.datos['callback_ms'][-1] = cb

    def visualizar(self, info):
        """Optional RViz output, implemented by each controller."""
        pass

    # -----------------------------------------------------------------
    def _finalizar_ensayo(self, eL, eR):
        self.publicar(np.zeros(6), np.zeros(6))
        tiempos = estadisticas_tiempo(self.pipeline_ms, self.callback_ms, self.period_ms,
                                      self.state_age_ms, self.col_ms)
        for brazo, met, err in (('LEFT', self.met_L, eL), ('RIGHT', self.met_R, eR)):
            d = {'algorithm': self.etiqueta, 'arm': brazo, 'trial_id': self.trial_id,
                 'result': 'SUCCESS' if err < TOL_BAJA else 'TIMEOUT'}
            d.update(met.resumen(self.t, err))
            d.update(tiempos)
            self.pub_metricas.publish(String(data=json.dumps(d)))
        ruta = os.path.join(self.log_dir, self.etiqueta, f'trial_{self.trial_id:03d}.npz')
        try:
            self.registro.guardar(ruta, {'trial_id': self.trial_id, 'algorithm': self.etiqueta,
                                         'T_L': self.traj_L[1], 'T_R': self.traj_R[1],
                                         'P_L': self.traj_L[0], 'P_R': self.traj_R[0]})
        except Exception as e:  # logging must never stop the benchmark
            self.get_logger().error(f'Could not save log {ruta}: {e}')
        self.get_logger().info(f'[{self.etiqueta}] trial {self.trial_id} finished: '
                               f'eL={eL:.3f} m, eR={eR:.3f} m, t={self.t:.2f} s, '
                               f"misses={tiempos['deadline_misses']}")
        self.traj_L = None; self.traj_R = None
        self.modo = 'IDLE'
        self.pub_status.publish(String(data='REACHED'))

    # Home hold: joint-space PI control towards q = 0. A pure P loop leaves a steady-state
    # offset because gravity keeps pulling the arms down between velocity updates; the
    # integral term removes it. The sign test (tools/prueba_signos.py) showed that every
    # joint moves in the commanded direction, so no sign flips are used.
    K_HOME = 3.0          # [1/s]
    KI_HOME = 3.0         # [1/s^2]
    I_MAX = 0.5           # [rad/s] anti-windup limit of the integral contribution
    TOL_HOME = 0.01       # [rad] max |q| of joints 1-5 (joint 6 does not move the end-effector)
    CICLOS_QUIETO = 50    # cycles (0.5 s) inside the tolerance before reporting HOME_REACHED
    T_MAX_HOME = 8.0      # [s] after this, HOME_REACHED is reported anyway (with a warning)

    def _retener_home(self):
        if not hasattr(self, 'int_L'):
            self.int_L = np.zeros(6); self.int_R = np.zeros(6)
        self.int_L = np.clip(self.int_L - self.KI_HOME * self.q_L * DT, -self.I_MAX, self.I_MAX)
        self.int_R = np.clip(self.int_R - self.KI_HOME * self.q_R * DT, -self.I_MAX, self.I_MAX)
        uL = np.clip(-self.K_HOME * self.q_L + self.int_L, -1.0, 1.0)
        uR = np.clip(-self.K_HOME * self.q_R + self.int_R, -1.0, 1.0)
        self.publicar(uL, uR)
        if self.modo == 'HOME':
            err = max(np.max(np.abs(self.q_L[:5])), np.max(np.abs(self.q_R[:5])))
            self.ciclos_home = self.ciclos_home + 1 if err < self.TOL_HOME else 0
            self.t_home += DT
            self.n_home = getattr(self, 'n_home', 0) + 1
            if self.n_home % 100 == 0:
                self.get_logger().info(f'Homing: max|q| = {err:.4f} rad')
            if self.ciclos_home >= self.CICLOS_QUIETO or self.t_home >= self.T_MAX_HOME:
                if self.ciclos_home < self.CICLOS_QUIETO:
                    self.get_logger().warn(f'Home tolerance not reached after {self.T_MAX_HOME} s '
                                           f'(max|q| = {err:.4f} rad); continuing.')
                self.pub_status.publish(String(data='HOME_REACHED'))
                self.modo = 'IDLE'