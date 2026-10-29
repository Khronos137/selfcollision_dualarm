#!/usr/bin/env python3
"""
Benchmark director: sends the trajectories, arbitrates collisions from the Gazebo
contact sensors and writes one CSV row per arm and trial.

Changes in revision 2:
  * contact sensors on every moving link (link2..link6, end_link) of both arms (R2-C1);
  * the colliding pair (collision1 / collision2 names) is stored for every event;
  * contacts present at the collision-free home posture during a short calibration
    window (mesh overlaps of the model itself) are reported and ignored;
  * explicit Trial_ID column, so the statistical pairing is unambiguous (R2-C3);
  * the controllers report their metrics as JSON (all timing statistics included).

Parameters: trajectory_file, csv_file, contacts_file, calibration_s, n_trials
"""
import os
import csv
import json
from datetime import datetime

import rclpy
import yaml
from rclpy.node import Node
from std_msgs.msg import String
from gazebo_msgs.msg import ContactsState
from ament_index_python.packages import get_package_share_directory

from dual_manipulator_benchmark.msg import BezierTrajectory

ENLACES = ['link2', 'link3', 'link4', 'link5', 'link6', 'end_link']

COLUMNAS = ['timestamp', 'trial_id', 'algorithm', 'arm', 'result', 'rmse_m', 'ce', 'jpl_rad',
            'jpl_meas_rad', 'min_clearance_m', 'final_err_m', 'time_factor', 't_total_s',
            't_reach_002_s', 't_reach_005_s', 'pipeline_mean_ms', 'pipeline_p99_ms',
            'pipeline_max_ms', 'collision_mean_ms', 'callback_mean_ms', 'callback_p99_ms',
            'callback_max_ms', 'deadline_misses', 'n_cycles', 'period_mean_ms', 'period_std_ms',
            'period_max_ms', 'state_age_max_ms', 'contact_pairs']


def _limpiar(nombre):
    # 'dual_arm_manipulator::left_link5::left_link5_collision' -> 'left_link5'
    partes = nombre.split('::')
    return partes[1] if len(partes) >= 2 else nombre


class BenchmarkDirector(Node):

    def __init__(self):
        super().__init__('benchmark_director')
        self.declare_parameter('trajectory_file', 'trayectorias_50.yaml')
        self.declare_parameter('csv_file', 'resultados_benchmark.csv')
        self.declare_parameter('contacts_file', 'contactos_benchmark.csv')
        self.declare_parameter('calibration_s', 3.0)
        self.declare_parameter('n_trials', 0)          # 0 = all trajectories in the file
        self.csv_file = self.get_parameter('csv_file').value
        self.contacts_file = self.get_parameter('contacts_file').value
        self.calibracion_s = float(self.get_parameter('calibration_s').value)

        if not os.path.exists(self.csv_file):
            with open(self.csv_file, 'w', newline='') as f:
                csv.writer(f).writerow(COLUMNAS)
        if not os.path.exists(self.contacts_file):
            with open(self.contacts_file, 'w', newline='') as f:
                csv.writer(f).writerow(['timestamp', 'trial_id', 'phase', 'body_1', 'body_2'])

        ruta = self.get_parameter('trajectory_file').value
        if not os.path.isabs(ruta):
            ruta = os.path.join(get_package_share_directory('dual_manipulator_benchmark'), 'config', ruta)
        with open(ruta) as f:
            self.rutas = yaml.safe_load(f).get('trayectorias', [])
        n = int(self.get_parameter('n_trials').value)
        if n > 0:
            self.rutas = self.rutas[:n]
        self.get_logger().info(f'{len(self.rutas)} trajectories loaded from {ruta}')

        for lado in ('left', 'right'):
            for enlace in ENLACES:
                self.create_subscription(ContactsState, f'/benchmark/colisiones/{lado}_{enlace}',
                                         self.callback_bumper, 10)
        self.create_subscription(String, '/benchmark/metricas', self.callback_metricas, 10)
        self.create_subscription(String, '/benchmark/status', self.cb_status, 10)
        self.pub_traj_L = self.create_publisher(BezierTrajectory, '/benchmark/left_arm_traj', 10)
        self.pub_traj_R = self.create_publisher(BezierTrajectory, '/benchmark/right_arm_traj', 10)
        self.pub_logger_cmd = self.create_publisher(String, '/benchmark/logger_cmd', 10)

        self.ignorados = set()                 # pairs in contact at the home posture
        self.contactos = set()                 # pairs in contact during the current trial
        self.trayectoria_actual = 0
        # Start: bring both arms to home, then calibrate contacts AT home, then run trials
        self.estado = 'HOMING_INICIAL'
        self.t_inicio = self.ahora()
        self.t_ultimo_home = -1e9
        self.timer = self.create_timer(0.5, self.maquina_estados)

    def ahora(self):
        return self.get_clock().now().nanoseconds / 1e9

    # -----------------------------------------------------------------
    def callback_bumper(self, msg):
        for st in msg.states:
            par = tuple(sorted((_limpiar(st.collision1_name), _limpiar(st.collision2_name))))
            if self.estado == 'CALIBRANDO':
                if par not in self.ignorados:
                    self.ignorados.add(par)
                    self.get_logger().warn(f'Contact at home posture (will be ignored): {par}')
                    self._guardar_contacto(-1, 'calibration', par)
            elif self.estado == 'EJECUTANDO' and par not in self.ignorados:
                if par not in self.contactos:
                    self.contactos.add(par)
                    self.get_logger().fatal(f'COLLISION in trial {self.trayectoria_actual}: {par}')
                    self._guardar_contacto(self.trayectoria_actual, 'trial', par)

    def _guardar_contacto(self, trial, fase, par):
        with open(self.contacts_file, 'a', newline='') as f:
            csv.writer(f).writerow([datetime.now().strftime('%H:%M:%S'), trial, fase, par[0], par[1]])

    def _brazo_en_colision(self, brazo):
        prefijo = 'left_' if brazo == 'LEFT' else 'right_'
        return [p for p in self.contactos if p[0].startswith(prefijo) or p[1].startswith(prefijo)]

    def callback_metricas(self, msg):
        try:
            d = json.loads(msg.data)
        except json.JSONDecodeError:
            self.get_logger().error(f'Unreadable metrics message: {msg.data[:80]}')
            return
        if d.get('trial_id') != self.trayectoria_actual:
            self.get_logger().warn(f"Controller trial_id {d.get('trial_id')} != director "
                                   f'{self.trayectoria_actual}; using the director index.')
        pares = self._brazo_en_colision(d['arm'])
        if pares:
            d['result'] = 'COLLISION'
        d['trial_id'] = self.trayectoria_actual
        d['timestamp'] = datetime.now().strftime('%H:%M:%S')
        d['contact_pairs'] = ';'.join(f'{a}|{b}' for a, b in sorted(pares))
        with open(self.csv_file, 'a', newline='') as f:
            csv.writer(f).writerow(['' if d.get(c) is None else d.get(c) for c in COLUMNAS])
        self.get_logger().info(f"Trial {d['trial_id']} | {d['algorithm']} | {d['arm']} | {d['result']}")

    def _pedir_home(self):
        home = BezierTrajectory()
        home.t_total = -1.0
        self.pub_traj_L.publish(home)
        self.pub_traj_R.publish(home)

    def cb_status(self, msg):
        if msg.data == 'HOME_REACHED' and self.estado == 'HOMING_INICIAL':
            self.get_logger().info('Arms at home. Calibrating contact sensors...')
            self.estado = 'CALIBRANDO'
            self.t_inicio = self.ahora()
            return
        if msg.data == 'REACHED' and self.estado == 'EJECUTANDO':
            self.estado = 'RETORNANDO_HOME'
            home = BezierTrajectory()
            home.t_total = -1.0
            self.pub_traj_L.publish(home)
            self.pub_traj_R.publish(home)
        elif msg.data == 'HOME_REACHED' and self.estado == 'RETORNANDO_HOME':
            self.t_llegada_home = self.ahora()
            self.estado = 'ESPERANDO'

    def maquina_estados(self):
        if self.estado == 'HOMING_INICIAL':
            # (re)send the home request every 2 s until the controller confirms it
            if self.ahora() - self.t_ultimo_home > 2.0:
                self.get_logger().info('Waiting for the controller to reach home...')
                self._pedir_home()
                self.t_ultimo_home = self.ahora()
        elif self.estado == 'CALIBRANDO':
            if self.ahora() - self.t_inicio >= self.calibracion_s:
                self.get_logger().info(f'Calibration finished. Ignored home contacts: {self.ignorados}')
                self.estado = 'INICIO'
        elif self.estado == 'INICIO':
            if self.trayectoria_actual < len(self.rutas):
                self.enviar_trayectoria(self.trayectoria_actual)
                self.estado = 'EJECUTANDO'
            else:
                self.get_logger().info('BENCHMARK FINISHED.')
                self.estado = 'FIN'
                self.timer.cancel()
        elif self.estado == 'ESPERANDO':
            if self.ahora() - self.t_llegada_home >= 1.5:
                self.trayectoria_actual += 1
                self.estado = 'INICIO'

    def enviar_trayectoria(self, i):
        self.contactos = set()
        datos = self.rutas[i]
        # The trial index is sent first so the controllers can label their logs
        self.pub_logger_cmd.publish(String(data=f'START_{i}'))
        msg_L, msg_R = BezierTrajectory(), BezierTrajectory()
        msg_L.t_total = msg_R.t_total = float(datos['t_total'])
        for m, brazo in ((msg_L, 'left_arm'), (msg_R, 'right_arm')):
            m.p0 = [float(x) for x in datos[brazo]['p0']]
            m.p1 = [float(x) for x in datos[brazo]['p1']]
            m.p2 = [float(x) for x in datos[brazo]['p2']]
            m.p3 = [float(x) for x in datos[brazo]['p3']]
        self.pub_traj_L.publish(msg_L)
        self.pub_traj_R.publish(msg_R)
        self.get_logger().info(f"Trial #{i} started (T = {msg_L.t_total} s)")


def main(args=None):
    rclpy.init(args=args)
    nodo = BenchmarkDirector()
    try:
        rclpy.spin(nodo)
    except KeyboardInterrupt:
        pass
    finally:
        nodo.destroy_node()
        rclpy.try_shutdown()


if __name__ == '__main__':
    main()