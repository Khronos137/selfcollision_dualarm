#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point
from builtin_interfaces.msg import Time
import numpy as np
import math

from libreria_cinematica_der import derCinemDirecta6ManH, derJac6ManH
from libreria_cinematica_izq import cinemDirecta6IzqManH, derJac6IzqManH
from libreria_rviz import crear_marcador_trayectoria, generar_esqueleto_colisiones

# Importamos la configuración centralizada de nuestro módulo de esferas
from collision import (obtener_puntos_actuales, revisar_estado_seguridad,
                       TORSO_N_ESFERAS, RADIO_TORSO,
                       RADIOS_BRAZOS, CELL_SIZE)

# Importamos el mensaje personalizado del benchmark
from dual_manipulator_benchmark.msg import BezierTrajectory

class MainDualBrazoSpheres(Node):
    def __init__(self):
        super().__init__('main_dual_brazo_sph')

        self.tipo_traj_L = "IDLE"
        self.tipo_traj_R = "IDLE"

        # 1. PARÁMETROS GEOMÉTRICOS (Del robot)
        self.h = 0.70
        self.b = 0.159 + 0.10
        self.l1 = 0.264
        self.l1b = 0.030
        self.l2 = 0.258
        self.l3 = 0.123
        self.L = [self.h, self.b, self.l1, self.l1b, self.l2, self.l3]
        
        # 2. PARÁMETROS DE CONTROL Y DLS
        self.Ke = 5.5 * np.diag([1.0, 1.0, 1.0]) 
        self.lam = 0.05  
        
        self.q_left = np.zeros(6)
        self.q_right = np.zeros(6)
        self.estado_recibido = False
        
        # 3. VARIABLES DE ESTADO DEL BENCHMARK BÉZIER
        self.t_left = 0.0
        self.t_right = 0.0
        self.ejecutando_benchmark_L = False
        self.ejecutando_benchmark_R = False
        
        self.T_total_L = 1.0
        self.T_total_R = 1.0
        self.P_L = np.zeros((4, 3))
        self.P_R = np.zeros((4, 3))

        self.contador = 0
        self.ts = 0.01

        # 4. COMUNICACIONES ROS 2
        self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        
        # Suscriptores del Benchmark
        self.create_subscription(BezierTrajectory, '/benchmark/left_arm_traj', self.cb_traj_left, 10)
        self.create_subscription(BezierTrajectory, '/benchmark/right_arm_traj', self.cb_traj_right, 10)
        
        self.pub_left = self.create_publisher(Float64MultiArray, '/left_velocity_controller/commands', 10)
        self.pub_right = self.create_publisher(Float64MultiArray, '/right_velocity_controller/commands', 10)
        self.pub_status = self.create_publisher(String, '/benchmark/status', 10)
        
        # Publicadores visuales
        self.pub_rviz = self.create_publisher(Marker, '/trayectorias_rviz', 10)
        self.pub_esqueleto = self.create_publisher(MarkerArray, '/colisiones/esqueleto', 10)
        self.pub_col_pts    = self.create_publisher(MarkerArray, '/collision_viz/points',  10)
        self.pub_col_links  = self.create_publisher(MarkerArray, '/collision_viz/links',   10)
        self.pub_col_torso  = self.create_publisher(MarkerArray, '/collision_viz/torso',   10)
        self.pub_col_vol    = self.create_publisher(MarkerArray, '/collision_viz/volumen', 10)

        # Marcadores RViz vacíos iniciales
        self.marker_izq = Marker()
        self.marker_der = Marker()

        self.create_timer(self.ts, self.bucle_principal)
        self.get_logger().info("CONTROLADOR PAPER BASE (SPHERES) INICIADO: Esperando comandos del Benchmark...")

        # =====================================================================
        # MOTOR DE MÉTRICAS
        # =====================================================================
        self.metricas_L = self.reset_metricas()
        self.metricas_R = self.reset_metricas()
        self.pub_metricas = self.create_publisher(String, '/benchmark/metricas', 10)

    # Añade esta función de apoyo justo debajo del __init__
    def reset_metricas(self):
        return {
            'rmse_sum': 0.0,          
            'steps': 0,               
            'tjv': 0.0,               
            'jerk_sum': 0.0,          
            'prev_vel': np.zeros(6),  
            'min_dist_obst': 999.0,
            # --- NUEVAS MÉTRICAS DE CPU ---
            't_cpu_sum': 0.0,
            't_cpu_max': 0.0,
            't_col_sum': 0.0
        }

    def cb_joints(self, msg: JointState):
        for i in range(6):
            nl = f'left_joint{i+1}'
            nr = f'right_joint{i+1}'
            if nl in msg.name: self.q_left[i] = msg.position[msg.name.index(nl)]
            if nr in msg.name: self.q_right[i] = msg.position[msg.name.index(nr)]
        self.estado_recibido = True

    def cb_traj_left(self, msg: BezierTrajectory):
        if msg.t_total < 0.0:
            # BANDERA DE RETORNO A HOME DETECTADA
            self.tipo_traj_L = "HOME"
            self.ejecutando_benchmark_L = False
        else:
            self.P_L = np.array([msg.p0, msg.p1, msg.p2, msg.p3])
            self.T_total_L = msg.t_total
            self.tipo_traj_L = "BENCHMARK"

            self.metricas_L = self.reset_metricas()
            
            puntos = [self.evaluar_bezier(t_sim, self.T_total_L, self.P_L)[0] for t_sim in np.linspace(0, self.T_total_L, 50)]
            self.marker_izq = crear_marcador_trayectoria(puntos, 'world', 'trayectorias', 0, 0.0, 1.0, 0.0)
            self.t_left = 0.0
            self.ejecutando_benchmark_L = True

    def cb_traj_right(self, msg: BezierTrajectory):
        if msg.t_total < 0.0:
            # BANDERA DE RETORNO A HOME DETECTADA
            self.tipo_traj_R = "HOME"
            self.ejecutando_benchmark_R = False
        else:
            self.P_R = np.array([msg.p0, msg.p1, msg.p2, msg.p3])
            self.T_total_R = msg.t_total
            self.tipo_traj_R = "BENCHMARK"

            self.metricas_R = self.reset_metricas()
            
            puntos = [self.evaluar_bezier(t_sim, self.T_total_R, self.P_R)[0] for t_sim in np.linspace(0, self.T_total_R, 50)]
            self.marker_der = crear_marcador_trayectoria(puntos, 'world', 'trayectorias', 1, 1.0, 0.0, 0.0)
            self.t_right = 0.0
            self.ejecutando_benchmark_R = True

    # =====================================================================
    # MATEMÁTICA ANALÍTICA DE BÉZIER
    # =====================================================================
    def evaluar_bezier(self, t, T_total, P):
        if T_total <= 0.001: return P[0], np.zeros(3)
        if t >= T_total: return P[3], np.zeros(3)
        s = t / T_total
        p_des = ((1 - s)**3 * P[0] + 3 * (1 - s)**2 * s * P[1] + 3 * (1 - s) * s**2 * P[2] + s**3 * P[3])
        v_des = (1.0 / T_total) * (3 * (1 - s)**2 * (P[1] - P[0]) + 6 * (1 - s) * s * (P[2] - P[1]) + 3 * s**2 * (P[3] - P[2]))
        return p_des, v_des

    def damped_pseudoinverse(self, J):
        J_T = J.T
        JJ_T = J @ J_T
        I = np.eye(JJ_T.shape[0]) 
        inversa_dampeada = np.linalg.inv(JJ_T + (self.lam**2) * I)
        return J_T @ inversa_dampeada

    def enviar_comandos_cero(self):
        msg = Float64MultiArray(data=[0.0]*6)
        self.pub_left.publish(msg)
        self.pub_right.publish(msg)

    # =====================================================================
    # ALGORITMO ESTRICTO: LEI ET AL. (2020) - EN 3D
    # =====================================================================
    def controlador_lei_et_al(self, v_end_3d, err_3d, J_completa, matriz_dist, puntos_3d, brazo):
        beta_min = 999.0
        v_max = 2.5 
        alfa = 20.0
        
        # AUMENTAMOS EL RADIO A 25 cm PARA ASEGURAR QUE LO DETECTE A TIEMPO
        d_influencia = 0.05 
        
        punto_colision_propio = None
        punto_colision_ajeno = None
        idx_esfera_propia = 1
        
        # 1. Encontrar el índice de sensibilidad mínimo (Ignorando base estática)
        for i in range(4, 11): 
            if brazo == 'left':
                for j in range(4, 11):
                    beta_actual = matriz_dist[i, j] / d_influencia 
                    if beta_actual < beta_min: 
                        beta_min = beta_actual
                        idx_esfera_propia = i
                        punto_colision_propio = puntos_3d[f'left_j{i}']
                        punto_colision_ajeno = puntos_3d[f'right_j{j}']
            else:
                for i_L in range(4, 11):
                    beta_actual = matriz_dist[i_L, i] / d_influencia
                    if beta_actual < beta_min: 
                        beta_min = beta_actual
                        idx_esfera_propia = i
                        punto_colision_propio = puntos_3d[f'right_j{i}']
                        punto_colision_ajeno = puntos_3d[f'left_j{i_L}']

        # Tarea de seguimiento puramente 3D
        v_tarea_3d = v_end_3d + np.tanh(self.Ke @ err_3d)
        
        # Inversa Pseudo-inversa (6x3)
        J_pinv = np.linalg.pinv(J_completa)
        
        # TELEMETRÍA DE SINGULARIDADES (Límite del espacio de trabajo)
        condicion_jacobiana = np.linalg.cond(J_completa)
        if condicion_jacobiana > 30.0 and self.contador % 20 == 0:
            self.get_logger().fatal(f"[SINGULARIDAD 3D] Límite de alcance. Condición: {condicion_jacobiana:.1f}")

        if beta_min >= 1.0:
            return J_pinv @ v_tarea_3d

        # =====================================================================
        # PROTOCOLO DE EVASIÓN ACTIVADO (beta_min < 1.0)
        # =====================================================================
        v_rep_mag = v_max / (1.0 + math.exp(alfa * (beta_min - 0.7)))
        
        vector_escape = punto_colision_propio - punto_colision_ajeno
        norm_escape = np.linalg.norm(vector_escape)
        if norm_escape > 1e-4:
            v_rv_k_3d = (vector_escape / norm_escape) * v_rep_mag
        else:
            v_rv_k_3d = np.zeros(3)

        articulacion_tope = min(6, (idx_esfera_propia + 1) // 2)
        J_parcial = np.copy(J_completa)
        J_parcial[:, articulacion_tope:] = 0.0 
        
        # Evasión puramente 3D
        J_parcial_pinv = np.linalg.pinv(J_parcial) 
        q_dot_re = J_parcial_pinv @ v_rv_k_3d  
        v_re_end_3d = J_completa @ q_dot_re    

        # El defecto de Lei et al. (Coseno)
        norm_v_tarea = np.linalg.norm(v_tarea_3d)
        norm_v_re_end = np.linalg.norm(v_re_end_3d)
        
        cos_gamma = 0.0
        if norm_v_tarea > 1e-4 and norm_v_re_end > 1e-4:
            cos_gamma = np.dot(v_re_end_3d, v_tarea_3d) / (norm_v_re_end * norm_v_tarea)
            v_pro_k_3d = norm_v_re_end * cos_gamma * (v_tarea_3d / norm_v_tarea)
        else:
            v_pro_k_3d = np.zeros(3)

        # TELEMETRÍA AGRESIVA
        if brazo == 'left' and self.contador % 10 == 0:
            self.get_logger().warn(f"[LEI] PELIGRO DETECTADO | Beta: {beta_min:.2f} | Dist: {beta_min * d_influencia:.3f}m")
            if abs(cos_gamma) < 0.2:
                self.get_logger().fatal("      --> [FALLO] Coseno ~ 0. El vector de evasión es lateral y será anulado.")

        v_new_k_3d = v_tarea_3d + v_pro_k_3d
        return J_pinv @ v_new_k_3d

    # =====================================================================
    # NÚCLEO DE CONTROL PRINCIPAL
    # =====================================================================
    def bucle_principal(self):
        if not self.estado_recibido: return

        # 1. EVALUACIÓN DE ESTADO DEL BENCHMARK
        if self.ejecutando_benchmark_L and self.t_left >= self.T_total_L: 
            # Ya no apagamos la bandera del benchmark aquí por tiempo nominal.
            # Dejamos que el reloj avance para medir el retraso real inducido por la evasión.
            pass
        if self.ejecutando_benchmark_R and self.t_right >= self.T_total_R: 
            pass
            
        if self.tipo_traj_L == "BENCHMARK" or self.tipo_traj_R == "BENCHMARK":
            # Calculamos la distancia física actual de las manos hacia sus metas finales (P3)
            pReal_L_check = cinemDirecta6IzqManH(self.L, self.q_left)
            pReal_R_check = derCinemDirecta6ManH(self.L, self.q_right)
            
            err_L_goal = np.linalg.norm(self.P_L[3] - pReal_L_check)
            err_r_goal = np.linalg.norm(self.P_R[3] - pReal_R_check)
            
            # --- CONDICIÓN A: LLEGADA FÍSICA REAL (ÉXITO) ---
            if err_L_goal < 0.02 and err_r_goal < 0.02:
                self.enviar_comandos_cero()
                
                # CÁLCULO FINAL DE MÉTRICAS
                rmse_L = np.sqrt(self.metricas_L['rmse_sum'] / max(1, self.metricas_L['steps']))
                rmse_R = np.sqrt(self.metricas_R['rmse_sum'] / max(1, self.metricas_R['steps']))
                
                factor_tiempo_L = self.t_left / self.T_total_L
                factor_tiempo_R = self.t_right / self.T_total_R
                
                # CÁLCULO FINAL DE MEDIAS COMPUTACIONALES (CPU)
                t_cpu_mean = self.metricas_L['t_cpu_sum'] / max(1, self.metricas_L['steps'])
                t_col_mean = self.metricas_L['t_col_sum'] / max(1, self.metricas_L['steps'])
                t_cpu_max = self.metricas_L['t_cpu_max']
                
                # STRING FORMATEADO (Añadidas las 3 variables al final de str_L)
                str_L = f"METRICS|LEI|LEFT|SUCCESS|{rmse_L:.4f}|{self.metricas_L['tjv']:.4f}|{self.metricas_L['jerk_sum']:.4f}|{self.metricas_L['min_dist_obst']:.4f}|{err_L_goal:.4f}|{factor_tiempo_L:.4f}|{t_cpu_mean:.4f}|{t_cpu_max:.4f}|{t_col_mean:.4f}"
                str_R = f"METRICS|LEI|RIGHT|SUCCESS|{rmse_R:.4f}|{self.metricas_R['tjv']:.4f}|{self.metricas_R['jerk_sum']:.4f}|{self.metricas_R['min_dist_obst']:.4f}|{err_r_goal:.4f}|{factor_tiempo_R:.4f}"
                
                msg = String(); msg.data = str_L; self.pub_metricas.publish(msg)
                msg.data = str_R; self.pub_metricas.publish(msg)
                
                # Mantenemos el REACHED original para no romper tu Director actual
                msg.data = "REACHED"; self.pub_status.publish(msg)
                
                self.t_left = 0.0; self.t_right = 0.0
                self.tipo_traj_L = "IDLE"; self.tipo_traj_R = "IDLE"
                self.ejecutando_benchmark_L = False; self.ejecutando_benchmark_R = False
                self.contador += 1
                return
                
            # --- CONDICIÓN B: TIEMPO LÍMITE (TIMEOUT / FALLO) ---
            elif self.t_left >= (2.0 * self.T_total_L) or self.t_right >= (2.0 * self.T_total_R):
                self.get_logger().error("⏰ TIMEOUT: Exceso de evasión.")
                self.enviar_comandos_cero()
                
                # MISMO CÁLCULO FINAL PARA EL TIMEOUT
                rmse_L = np.sqrt(self.metricas_L['rmse_sum'] / max(1, self.metricas_L['steps']))
                rmse_R = np.sqrt(self.metricas_R['rmse_sum'] / max(1, self.metricas_R['steps']))
                factor_tiempo_L = self.t_left / self.T_total_L
                factor_tiempo_R = self.t_right / self.T_total_R
                
                # CÁLCULO FINAL DE MEDIAS COMPUTACIONALES (CPU)
                t_cpu_mean = self.metricas_L['t_cpu_sum'] / max(1, self.metricas_L['steps'])
                t_col_mean = self.metricas_L['t_col_sum'] / max(1, self.metricas_L['steps'])
                t_cpu_max = self.metricas_L['t_cpu_max']
                
                # STRING FORMATEADO (Añadidas las 3 variables al final de str_L)
                str_L = f"METRICS|LEI|LEFT|TIMEOUT|{rmse_L:.4f}|{self.metricas_L['tjv']:.4f}|{self.metricas_L['jerk_sum']:.4f}|{self.metricas_L['min_dist_obst']:.4f}|{err_L_goal:.4f}|{factor_tiempo_L:.4f}|{t_cpu_mean:.4f}|{t_cpu_max:.4f}|{t_col_mean:.4f}"
                str_R = f"METRICS|LEI|RIGHT|TIMEOUT|{rmse_R:.4f}|{self.metricas_R['tjv']:.4f}|{self.metricas_R['jerk_sum']:.4f}|{self.metricas_R['min_dist_obst']:.4f}|{err_r_goal:.4f}|{factor_tiempo_R:.4f}"
                
                msg = String(); msg.data = str_L; self.pub_metricas.publish(msg)
                msg.data = str_R; self.pub_metricas.publish(msg)
                
                msg.data = "REACHED"; self.pub_status.publish(msg) 
                
                self.t_left = 0.0; self.t_right = 0.0
                self.tipo_traj_L = "IDLE"; self.tipo_traj_R = "IDLE"
                self.ejecutando_benchmark_L = False; self.ejecutando_benchmark_R = False
                self.contador += 1
                return

        # --- CASO DE RETORNO A HOME (Mantenemos tu excelente lógica de control articular) ---
        elif self.tipo_traj_L == "HOME" or self.tipo_traj_R == "HOME":
            q_home_L = np.zeros(6); q_home_R = np.zeros(6)
            err_q_L = q_home_L - self.q_left; err_q_R = q_home_R - self.q_right
            
            norm_L = np.linalg.norm(err_q_L)
            norm_R = np.linalg.norm(err_q_R)
            
            if norm_L < 0.2 and norm_R < 0.2:
                self.enviar_comandos_cero()
                msg = String(); msg.data = "HOME_REACHED"
                self.pub_status.publish(msg)
                self.tipo_traj_L = "IDLE"; self.tipo_traj_R = "IDLE"
                self.contador += 1
                return
            else:
                # 1. Calculamos las velocidades matemáticas correctas
                uL_home = 0.9 * err_q_L; uR_home = 0.9 * err_q_R
                
                # -------------------------------------------------------------
                # 2. ⚠️ PARCHE DE HARDWARE PARA GAZEBO CLASSIC (INVERSIÓN) ⚠️
                # Invertimos el signo físico de j4 (índice 3) y j6 (índice 5)
                # que están montados al revés en la cadena cinemática visual.
                # -------------------------------------------------------------
                uL_home[3] = -uL_home[3]
                uL_home[5] = -uL_home[5]
                
                uR_home[3] = -uR_home[3]
                uR_home[5] = -uR_home[5]
                
                # 3. Recortamos a zona segura
                uL_seguro = np.clip(uL_home, -1.0, 1.0); uR_seguro = np.clip(uR_home, -1.0, 1.0)
                
                # 4. Publicamos al robot
                self.pub_left.publish(Float64MultiArray(data=uL_seguro.tolist()))
                self.pub_right.publish(Float64MultiArray(data=uR_seguro.tolist()))
                self.contador += 1
                return

        elif self.tipo_traj_L == "IDLE" and self.tipo_traj_R == "IDLE":
            self.contador += 1
            return

        # =====================================================================
        # TODO EL CÓDIGO CARTESIANO DE ABAJO SOLO SE EJECUTA SI ESTÁ EN "BENCHMARK"
        # =====================================================================
        import time
        t_start_loop = time.perf_counter() # ⏱️ INICIO CONTROL LOOP PRINCIPAL

        radios_dict = {f'j{k}': v for k, v in RADIOS_BRAZOS.items()}
        
        t_start_col = time.perf_counter() # ⏱️ INICIO CRONÓMETRO DE COLISIÓN (Lei et al.)
        puntos_3d = obtener_puntos_actuales(self.q_left, self.q_right, self.L)
        peligro, reportes, matriz_dist, viz_data = revisar_estado_seguridad(puntos_3d, radios_dict)
        t_end_col = time.perf_counter() # ⏱️ FIN CRONÓMETRO DE COLISIÓN



        # 3. MATEMÁTICA PURA (Lei et al. 2020)
        pDes_L, vDes_L = self.evaluar_bezier(self.t_left, self.T_total_L, self.P_L)
        pDes_R, vDes_R = self.evaluar_bezier(self.t_right, self.T_total_R, self.P_R)

        pReal_L = cinemDirecta6IzqManH(self.L, self.q_left)
        Jl = derJac6IzqManH(self.L, self.q_left)
        
        pReal_R = derCinemDirecta6ManH(self.L, self.q_right)
        Jr = derJac6ManH(self.L, self.q_right)

        err_L = pDes_L - pReal_L
        err_R = pDes_R - pReal_R

        # Aplicar el algoritmo base EXACTO: Lei et al. 2020
        uL = self.controlador_lei_et_al(vDes_L, err_L, Jl, matriz_dist, puntos_3d, 'left')
        uR = self.controlador_lei_et_al(vDes_R, err_R, Jr, matriz_dist, puntos_3d, 'right')

        limite_velocidad = 3.14 
        uL_seguro = np.clip(uL, -limite_velocidad, limite_velocidad)
        uR_seguro = np.clip(uR, -limite_velocidad, limite_velocidad)




        if self.ejecutando_benchmark_L: self.pub_left.publish(Float64MultiArray(data=uL_seguro[0:6].tolist()))
        if self.ejecutando_benchmark_R: self.pub_right.publish(Float64MultiArray(data=uR_seguro[0:6].tolist()))

        t_end_loop = time.perf_counter() # ⏱️ FIN CONTROL LOOP PRINCIPAL
        t_loop_ms = (t_end_loop - t_start_loop) * 1000.0
        t_col_ms = (t_end_col - t_start_col) * 1000.0

        # =====================================================================
        # EXTRACCIÓN DE LA DISTANCIA MÍNIMA (Matemática de Esferas - Lei)
        # =====================================================================
        dist_min_global = min([matriz_dist[i, j] for i in range(4, 11) for j in range(4, 11)])
        dist_min_L = dist_min_global
        dist_min_R = dist_min_global

        # =====================================================================
        # RECOLECCIÓN DE MÉTRICAS EN TIEMPO REAL
        # =====================================================================
        if self.ejecutando_benchmark_L:
            self.metricas_L['rmse_sum'] += np.linalg.norm(err_L)**2
            self.metricas_L['steps'] += 1
            
            if dist_min_L < self.metricas_L['min_dist_obst']:
                self.metricas_L['min_dist_obst'] = dist_min_L
                
            self.metricas_L['tjv'] += np.sum(np.abs(uL_seguro)) * self.ts
            
            aceleracion_L = (uL_seguro - self.metricas_L['prev_vel']) / self.ts
            self.metricas_L['jerk_sum'] += np.sum(np.abs(aceleracion_L)) * self.ts
            self.metricas_L['prev_vel'] = uL_seguro.copy()
            
            # --- NUEVAS MÉTRICAS COMPUTACIONALES ACUMULADAS ---
            self.metricas_L['t_cpu_sum'] += t_loop_ms
            self.metricas_L['t_col_sum'] += t_col_ms
            if t_loop_ms > self.metricas_L['t_cpu_max']:
                self.metricas_L['t_cpu_max'] = t_loop_ms

        if self.ejecutando_benchmark_R:
            self.metricas_R['rmse_sum'] += np.linalg.norm(err_R)**2
            self.metricas_R['steps'] += 1
            if dist_min_R < self.metricas_R['min_dist_obst']:
                self.metricas_R['min_dist_obst'] = dist_min_R
            self.metricas_R['tjv'] += np.sum(np.abs(uR_seguro)) * self.ts
            aceleracion_R = (uR_seguro - self.metricas_R['prev_vel']) / self.ts
            self.metricas_R['jerk_sum'] += np.sum(np.abs(aceleracion_R)) * self.ts
            self.metricas_R['prev_vel'] = uR_seguro.copy()
        # =====================================================================

        if self.ejecutando_benchmark_L: self.t_left += self.ts
        if self.ejecutando_benchmark_R: self.t_right += self.ts
        self.contador += 1

        # =====================================================================
        # 4. VISUALIZACIÓN (IDÉNTICA A LUMELSKY PARA JUSTICIA VISUAL)
        # =====================================================================
        tiempo_cero = Time()
        if self.contador % 5 == 0:
            arreglo_esqueleto = generar_esqueleto_colisiones(puntos_3d, radios_dict, RADIO_TORSO, 'world')
            for m in arreglo_esqueleto.markers: m.header.stamp = tiempo_cero
            self.pub_esqueleto.publish(arreglo_esqueleto)

            # Volúmenes visuales (Esferas)
            pc_L, pc_R = viz_data['pc_L'], viz_data['pc_R']
            centros_torso = viz_data['centros_torso']
            ma_vol = MarkerArray()
            
            for side, pc_w in (('L', pc_L), ('R', pc_R)):
                for vi, (seg, p) in enumerate(pc_w):
                    r = RADIOS_BRAZOS.get(seg, 0.04)
                    mv = Marker()
                    mv.header.frame_id = 'world'; mv.header.stamp = tiempo_cero
                    mv.ns = f'vol_{side}'; mv.id = vi
                    mv.type = Marker.SPHERE; mv.action = Marker.ADD
                    mv.pose.position.x = float(p[0])
                    mv.pose.position.y = float(p[1])
                    mv.pose.position.z = float(p[2])
                    mv.pose.orientation.w = 1.0
                    mv.scale.x = r * 2; mv.scale.y = r * 2; mv.scale.z = r * 2
                    mv.color.r = 0.3; mv.color.g = 0.6; mv.color.b = 1.0; mv.color.a = 0.5
                    ma_vol.markers.append(mv)
            self.pub_col_vol.publish(ma_vol)

        if self.contador % 10 == 0:
            if self.marker_izq.points:
                self.marker_izq.header.stamp = tiempo_cero
                self.pub_rviz.publish(self.marker_izq)
            if self.marker_der.points:
                self.marker_der.header.stamp = tiempo_cero
                self.pub_rviz.publish(self.marker_der)

            self.get_logger().info(f"--- T_L: {self.t_left:.1f}/{self.T_total_L}s | T_R: {self.t_right:.1f}/{self.T_total_R}s | Err_L: {np.linalg.norm(err_L):.4f} | Err_R: {np.linalg.norm(err_R):.4f} ---")
            
            if peligro:
                self.get_logger().warn("⚠️ ALERTA: ZONA DE COLISIÓN DETECTADA (LEY ET AL. ACTIVADO)")

def main(args=None):
    rclpy.init(args=args)
    nodo = MainDualBrazoSpheres()
    try: rclpy.spin(nodo)
    except KeyboardInterrupt: pass
    finally:
        nodo.enviar_comandos_cero()
        nodo.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()