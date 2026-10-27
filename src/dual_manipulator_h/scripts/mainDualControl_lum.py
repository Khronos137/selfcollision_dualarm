#!/usr/bin/env python3
import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import rclpy
from rclpy.node import Node
from sensor_msgs.msg import JointState
from std_msgs.msg import Float64MultiArray, String
from visualization_msgs.msg import Marker, MarkerArray
from builtin_interfaces.msg import Time
import numpy as np

# =====================================================================
# LIBRERIAS DE CINEMATICA Y VISUALIZACION
# =====================================================================
from libreria_cinematica_der import derCinemDirecta6ManH, derJac6ManH
from libreria_cinematica_izq import cinemDirecta6IzqManH, derJac6IzqManH
from libreria_rviz import crear_marcador_trayectoria, generar_esqueleto_colisiones, generar_flecha_fuerza
from collision import obtener_puntos_actuales, revisar_estado_seguridad

# LIBRERIA DE ESTRATEGIA MULTIPUNTO
from selfCollisionStrategy import calcular_matriz_W_multipunto, calcular_fuerzas_multipunto

# IMPORTAMOS EL MENSAJE PERSONALIZADO DEL BENCHMARK
from dual_manipulator_benchmark.msg import BezierTrajectory

class MainDualBrazoLumelsky(Node):
    def __init__(self):
        super().__init__('main_dual_brazo_lum')

        self.tipo_traj_L = "IDLE"
        self.tipo_traj_R = "IDLE"

        # 1. PARAMETROS GEOMETRICOS (Del robot)
        self.h = 0.70
        self.b = 0.159 + 0.10
        self.l1 = 0.264
        self.l1b = 0.030
        self.l2 = 0.258
        self.l3 = 0.123
        self.L = [self.h, self.b, self.l1, self.l1b, self.l2, self.l3]
        
        # 2. PARAMETROS DE CONTROL Y DLS
        self.Ke = 5.5 * np.diag([1.0, 1.0, 1.0]) 
        self.lam = 0.05  
        
        # 3. PARAMETROS DE SUPERVIVENCIA
        self.rho_L = 30.0            
        self.eta_col_L = 2.0         
        self.eta_vort_L = 0.0        

        self.rho_R = 500.0           
        self.eta_col_R = 2.0        
        self.eta_vort_R = 8.0        

        self.d_safe = 0.12              
        self.d_crit = 0.06                   
        self.eta_lim = 0.1           
        self.margen_limite = 0.1     

        self.q_min = np.array([-np.pi, -np.pi/2, -np.pi/2, -np.pi, -np.pi/2, -np.pi])
        self.q_max = np.array([ np.pi,  np.pi/2,  3*np.pi/4,  np.pi,  np.pi/2,  np.pi])
        
        self.q_left = np.zeros(6)
        self.q_right = np.zeros(6)
        self.estado_recibido = False
        
        # 4. VARIABLES DE ESTADO DEL BENCHMARK BEZIER
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
        self.latencias = [] # <--- LISTA PARA MEDIR TIEMPOS DEL REVISOR

        # =====================================================================
        # MEMORIA DEL FILTRO PASA BAJOS (ANTI VIBRACIONES)
        # =====================================================================
        self.alpha_filtro = 0.15
        self.q_rep_L_prev = np.zeros(6)
        self.q_rep_R_prev = np.zeros(6)
        self.fuerzas_L_prev = {i: np.zeros(3) for i in range(1, 11)}
        self.fuerzas_R_prev = {i: np.zeros(3) for i in range(1, 11)}

        # =====================================================================
        # COMUNICACIONES ROS 2
        # =====================================================================
        self.create_subscription(JointState, '/joint_states', self.cb_joints, 10)
        
        self.create_subscription(BezierTrajectory, '/benchmark/left_arm_traj', self.cb_traj_left, 10)
        self.create_subscription(BezierTrajectory, '/benchmark/right_arm_traj', self.cb_traj_right, 10)
        
        self.pub_left = self.create_publisher(Float64MultiArray, '/left_velocity_controller/commands', 10)
        self.pub_right = self.create_publisher(Float64MultiArray, '/right_velocity_controller/commands', 10)
        self.pub_status = self.create_publisher(String, '/benchmark/status', 10)
        
        self.pub_rviz = self.create_publisher(Marker, '/trayectorias_rviz', 10)
        self.pub_esqueleto = self.create_publisher(MarkerArray, '/colisiones/esqueleto', 10)
        self.pub_fuerzas_R = self.create_publisher(MarkerArray, '/fuerzas_repulsion_R', 10)
        self.pub_fuerzas_L = self.create_publisher(MarkerArray, '/fuerzas_repulsion_L', 10)

        self.marker_izq = Marker()
        self.marker_der = Marker()

        self.create_timer(self.ts, self.bucle_principal)
        
        print("CEREBRO DUAL LUMELSKY INICIADO: Esperando comandos del Benchmark...\n")

        self.metricas_L = self.reset_metricas()
        self.metricas_R = self.reset_metricas()
        self.pub_metricas = self.create_publisher(String, '/benchmark/metricas', 10)

    def reset_metricas(self):
        return {
            'rmse_sum': 0.0, 'steps': 0, 'kce': 0.0, 'accel_sum': 0.0,
            'prev_vel': np.zeros(6), 'min_dist_obst': 999.0,
            't_cpu_sum': 0.0, 't_cpu_max': 0.0, 't_col_sum': 0.0
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

    def evaluar_bezier(self, t, T_total, P):
        if T_total <= 0.001:
            return P[0], np.zeros(3)
        if t >= T_total:
            return P[3], np.zeros(3)
            
        s = t / T_total
        p_des = ((1 - s)**3 * P[0] + 
                 3 * (1 - s)**2 * s * P[1] + 
                 3 * (1 - s) * s**2 * P[2] + 
                 s**3 * P[3])
                 
        v_des = (1.0 / T_total) * (
                 3 * (1 - s)**2 * (P[1] - P[0]) + 
                 6 * (1 - s) * s * (P[2] - P[1]) + 
                 3 * s**2 * (P[3] - P[2]))
        return p_des, v_des

    def damped_weighted_pseudoinverse(self, J, W):
        W_inv = np.diag(1.0 / np.diag(W))
        J_T = J.T
        matriz_central = J @ W_inv @ J_T
        I = np.eye(matriz_central.shape[0]) 
        inversa_dampeada = np.linalg.inv(matriz_central + (self.lam**2) * I)
        return W_inv @ J_T @ inversa_dampeada

    def enviar_comandos_cero(self):
        msg = Float64MultiArray(data=[0.0]*6)
        self.pub_left.publish(msg)
        self.pub_right.publish(msg)

    # =====================================================================
    # NUCLEO DE CONTROL PRINCIPAL
    # =====================================================================
    def bucle_principal(self):
        if not self.estado_recibido: return

        if self.tipo_traj_L == "BENCHMARK" or self.tipo_traj_R == "BENCHMARK":
            pReal_L_check = cinemDirecta6IzqManH(self.L, self.q_left)
            pReal_R_check = derCinemDirecta6ManH(self.L, self.q_right)
            
            err_L_goal = np.linalg.norm(self.P_L[3] - pReal_L_check)
            err_r_goal = np.linalg.norm(self.P_R[3] - pReal_R_check)
            
            if err_L_goal < 0.02 and err_r_goal < 0.02:
                self.enviar_comandos_cero()
                
                rmse_L = np.sqrt(self.metricas_L['rmse_sum'] / max(1, self.metricas_L['steps']))
                rmse_R = np.sqrt(self.metricas_R['rmse_sum'] / max(1, self.metricas_R['steps']))
                factor_tiempo_L = self.t_left / self.T_total_L
                factor_tiempo_R = self.t_right / self.T_total_R
                
                t_cpu_mean = self.metricas_L['t_cpu_sum'] / max(1, self.metricas_L['steps'])
                t_col_mean = self.metricas_L['t_col_sum'] / max(1, self.metricas_L['steps'])
                t_cpu_max = self.metricas_L['t_cpu_max']
                
                str_L = f"METRICS|LUMELSKY|LEFT|SUCCESS|{rmse_L:.4f}|{self.metricas_L['kce']:.4f}|{self.metricas_L['accel_sum']:.4f}|{self.metricas_L['min_dist_obst']:.4f}|{err_L_goal:.4f}|{factor_tiempo_L:.4f}|{t_cpu_mean:.4f}|{t_cpu_max:.4f}|{t_col_mean:.4f}"
                str_R = f"METRICS|LUMELSKY|RIGHT|SUCCESS|{rmse_R:.4f}|{self.metricas_R['kce']:.4f}|{self.metricas_R['accel_sum']:.4f}|{self.metricas_R['min_dist_obst']:.4f}|{err_r_goal:.4f}|{factor_tiempo_R:.4f}"
                
                msg = String(); msg.data = str_L; self.pub_metricas.publish(msg)
                msg.data = str_R; self.pub_metricas.publish(msg)
                
                msg.data = "REACHED"; self.pub_status.publish(msg)
                
                self.t_left = 0.0; self.t_right = 0.0
                self.tipo_traj_L = "IDLE"; self.tipo_traj_R = "IDLE"
                self.ejecutando_benchmark_L = False; self.ejecutando_benchmark_R = False
                self.contador += 1
                return
                
            elif self.t_left >= (2.0 * self.T_total_L) or self.t_right >= (2.0 * self.T_total_R):
                self.enviar_comandos_cero()
                
                rmse_L = np.sqrt(self.metricas_L['rmse_sum'] / max(1, self.metricas_L['steps']))
                rmse_R = np.sqrt(self.metricas_R['rmse_sum'] / max(1, self.metricas_R['steps']))
                factor_tiempo_L = self.t_left / self.T_total_L
                factor_tiempo_R = self.t_right / self.T_total_R
                
                t_cpu_mean = self.metricas_L['t_cpu_sum'] / max(1, self.metricas_L['steps'])
                t_col_mean = self.metricas_L['t_col_sum'] / max(1, self.metricas_L['steps'])
                t_cpu_max = self.metricas_L['t_cpu_max']
                
                str_L = f"METRICS|LUMELSKY|LEFT|TIMEOUT|{rmse_L:.4f}|{self.metricas_L['kce']:.4f}|{self.metricas_L['accel_sum']:.4f}|{self.metricas_L['min_dist_obst']:.4f}|{err_L_goal:.4f}|{factor_tiempo_L:.4f}|{t_cpu_mean:.4f}|{t_cpu_max:.4f}|{t_col_mean:.4f}"
                str_R = f"METRICS|LUMELSKY|RIGHT|TIMEOUT|{rmse_R:.4f}|{self.metricas_R['kce']:.4f}|{self.metricas_R['accel_sum']:.4f}|{self.metricas_R['min_dist_obst']:.4f}|{err_r_goal:.4f}|{factor_tiempo_R:.4f}"
                
                msg = String(); msg.data = str_L; self.pub_metricas.publish(msg)
                msg.data = str_R; self.pub_metricas.publish(msg)
                
                msg.data = "REACHED"; self.pub_status.publish(msg) 
                
                self.t_left = 0.0; self.t_right = 0.0
                self.tipo_traj_L = "IDLE"; self.tipo_traj_R = "IDLE"
                self.ejecutando_benchmark_L = False; self.ejecutando_benchmark_R = False
                self.contador += 1
                return

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
                uL_home = 0.9 * err_q_L; uR_home = 0.9 * err_q_R
                uL_home[3] = -uL_home[3]
                uL_home[5] = -uL_home[5]
                uR_home[3] = -uR_home[3]
                uR_home[5] = -uR_home[5]
                
                uL_seguro = np.clip(uL_home, -1.0, 1.0); uR_seguro = np.clip(uR_home, -1.0, 1.0)
                self.pub_left.publish(Float64MultiArray(data=uL_seguro.tolist()))
                self.pub_right.publish(Float64MultiArray(data=uR_seguro.tolist()))
                self.contador += 1
                return

        elif self.tipo_traj_L == "IDLE" and self.tipo_traj_R == "IDLE":
            self.contador += 1
            return

        # =====================================================================
        # TODO EL CODIGO CARTESIANO DE ABAJO SOLO SE EJECUTA SI ESTA EN "BENCHMARK"
        # =====================================================================
        import time
        t_start_loop = time.perf_counter() 

        radios_dict = {
            'j1': 0.035, 'j2': 0.075, 'j3': 0.075, 'j4': 0.05, 'j5': 0.065, 
            'j6': 0.065, 'j7': 0.045, 'j8': 0.055, 'j9': 0.055, 'j10': 0.03
        }
        
        t_start_col = time.perf_counter()
        
        puntos_3d = obtener_puntos_actuales(self.q_left, self.q_right, self.L)
        peligro, reportes, matriz_dist = revisar_estado_seguridad(puntos_3d, radios_dict)

        radares_L = {i: {'dist': 999.0, 'pos': np.zeros(3), 'nombre': 'Libre'} for i in range(1, 11)}
        radares_R = {i: {'dist': 999.0, 'pos': np.zeros(3), 'nombre': 'Libre'} for i in range(1, 11)}

        radio_torso = 0.10 
        centro_torso_xy = np.array([0.0, 0.0])
        
        for i in range(1, 11): 
            if i > 3: 
                pL_xy = puntos_3d[f'left_j{i}'][0:2]
                dL = np.linalg.norm(pL_xy - centro_torso_xy) - radio_torso
                if dL < radares_L[i]['dist']:
                    radares_L[i]['dist'] = dL
                    vec = pL_xy - centro_torso_xy
                    vec_norm = vec / (np.linalg.norm(vec) + 1e-6)
                    radares_L[i]['pos'] = np.array([radio_torso * vec_norm[0], radio_torso * vec_norm[1], puntos_3d[f'left_j{i}'][2]])
                    radares_L[i]['nombre'] = f"TORSO(j{i})"
                    
                pR_xy = puntos_3d[f'right_j{i}'][0:2]
                dR = np.linalg.norm(pR_xy - centro_torso_xy) - radio_torso
                if dR < radares_R[i]['dist']:
                    radares_R[i]['dist'] = dR
                    vec = pR_xy - centro_torso_xy
                    vec_norm = vec / (np.linalg.norm(vec) + 1e-6)
                    radares_R[i]['pos'] = np.array([radio_torso * vec_norm[0], radio_torso * vec_norm[1], puntos_3d[f'right_j{i}'][2]])
                    radares_R[i]['nombre'] = f"TORSO(j{i})"

            for j in range(1, 11):
                if i <= 3 and j <= 3: continue 
                dist_brazos = matriz_dist[i, j]
                
                if dist_brazos < radares_L[i]['dist']:
                    radares_L[i]['dist'] = dist_brazos
                    radares_L[i]['pos'] = puntos_3d[f'right_j{j}']
                    radares_L[i]['nombre'] = f"Der(j{j})"
                    
                if dist_brazos < radares_R[j]['dist']:
                    radares_R[j]['dist'] = dist_brazos
                    radares_R[j]['pos'] = puntos_3d[f'left_j{i}']
                    radares_R[j]['nombre'] = f"Izq(j{i})"
                    
        t_end_col = time.perf_counter() 

        pDes_L, vDes_L = self.evaluar_bezier(self.t_left, self.T_total_L, self.P_L)
        pDes_R, vDes_R = self.evaluar_bezier(self.t_right, self.T_total_R, self.P_R)

        W_L = calcular_matriz_W_multipunto(radares_L, self.q_left, self.q_min, self.q_max, self.d_safe, self.d_crit, self.rho_L, self.margen_limite)
        q_rep_L, fuerzas_L_dict = calcular_fuerzas_multipunto(radares_L, puntos_3d, 'left', self.L, self.d_safe, self.q_left, self.q_min, self.q_max, self.eta_col_L, self.eta_lim, self.margen_limite, self.eta_vort_L, vDes_L)

        W_R = calcular_matriz_W_multipunto(radares_R, self.q_right, self.q_min, self.q_max, self.d_safe, self.d_crit, self.rho_R, self.margen_limite)
        q_rep_R, fuerzas_R_dict = calcular_fuerzas_multipunto(radares_R, puntos_3d, 'right', self.L, self.d_safe, self.q_right, self.q_min, self.q_max, self.eta_col_R, self.eta_lim, self.margen_limite, self.eta_vort_R, vDes_R)

        self.q_rep_L_prev = self.alpha_filtro * q_rep_L + (1.0 - self.alpha_filtro) * self.q_rep_L_prev
        q_rep_L_suave = self.q_rep_L_prev
        
        self.q_rep_R_prev = self.alpha_filtro * q_rep_R + (1.0 - self.alpha_filtro) * self.q_rep_R_prev
        q_rep_R_suave = self.q_rep_R_prev

        for i in range(1, 11):
            self.fuerzas_L_prev[i] = self.alpha_filtro * fuerzas_L_dict[i] + (1.0 - self.alpha_filtro) * self.fuerzas_L_prev[i]
            fuerzas_L_dict[i] = self.fuerzas_L_prev[i] 
            
            self.fuerzas_R_prev[i] = self.alpha_filtro * fuerzas_R_dict[i] + (1.0 - self.alpha_filtro) * self.fuerzas_R_prev[i]
            fuerzas_R_dict[i] = self.fuerzas_R_prev[i] 

        pReal_L = cinemDirecta6IzqManH(self.L, self.q_left)
        Jl = derJac6IzqManH(self.L, self.q_left) 
        
        pReal_R = derCinemDirecta6ManH(self.L, self.q_right)
        Jr = derJac6ManH(self.L, self.q_right)   

        err_L = pDes_L - pReal_L
        err_R = pDes_R - pReal_R

        Jl_pinv_W = self.damped_weighted_pseudoinverse(Jl, W_L)
        Jr_pinv_W = self.damped_weighted_pseudoinverse(Jr, W_R)

        I_6x6 = np.eye(6)
        N_L = I_6x6 - (Jl_pinv_W @ Jl)
        N_R = I_6x6 - (Jr_pinv_W @ Jr)

        u_task_L = Jl_pinv_W @ (vDes_L + np.tanh(self.Ke @ err_L))
        u_task_R = Jr_pinv_W @ (vDes_R + np.tanh(self.Ke @ err_R))

        u_null_L = N_L @ q_rep_L_suave
        u_null_R = N_R @ q_rep_R_suave

        uL = u_task_L + u_null_L
        uR = u_task_R + u_null_R

        limite_velocidad = 3.14 
        uL_seguro = np.clip(uL, -limite_velocidad, limite_velocidad)
        uR_seguro = np.clip(uR, -limite_velocidad, limite_velocidad)

        if self.ejecutando_benchmark_L: self.pub_left.publish(Float64MultiArray(data=uL_seguro[0:6].tolist()))
        if self.ejecutando_benchmark_R: self.pub_right.publish(Float64MultiArray(data=uR_seguro[0:6].tolist()))

        t_end_loop = time.perf_counter() 
        
        t_loop_ms = (t_end_loop - t_start_loop) * 1000.0
        t_col_ms = (t_end_col - t_start_col) * 1000.0
        
        self.latencias.append(t_loop_ms) 

        # =====================================================================
        # IMPRESION EN TIEMPO REAL (Cada 100 ciclos = 1 segundo)
        # =====================================================================
        if self.contador % 100 == 0 and len(self.latencias) > 0:
            lats = np.array(self.latencias)
            misses = np.sum(lats > 10.0)
            miss_rate = (misses / len(lats)) * 100.0
            self.get_logger().info(f"CPU METRICS -> Max: {np.max(lats):.3f} ms | P99: {np.percentile(lats, 99):.3f} ms | Promedio: {np.mean(lats):.3f} ms | Fallos: {misses}")

        dist_min_L = min([radar['dist'] for radar in radares_L.values()])
        dist_min_R = min([radar['dist'] for radar in radares_R.values()])

        if self.ejecutando_benchmark_L:
            self.metricas_L['rmse_sum'] += np.linalg.norm(err_L)**2
            self.metricas_L['steps'] += 1
            
            if dist_min_L < self.metricas_L['min_dist_obst']:
                self.metricas_L['min_dist_obst'] = dist_min_L
                
            self.metricas_L['kce'] += np.sum(np.abs(uL_seguro)) * self.ts
            
            aceleracion_L = (uL_seguro - self.metricas_L['prev_vel']) / self.ts
            self.metricas_L['accel_sum'] += np.sum(np.abs(aceleracion_L)) * self.ts
            self.metricas_L['prev_vel'] = uL_seguro.copy()
            
            self.metricas_L['t_cpu_sum'] += t_loop_ms
            self.metricas_L['t_col_sum'] += t_col_ms
            if t_loop_ms > self.metricas_L['t_cpu_max']:
                self.metricas_L['t_cpu_max'] = t_loop_ms

        if self.ejecutando_benchmark_R:
            self.metricas_R['rmse_sum'] += np.linalg.norm(err_R)**2
            self.metricas_R['steps'] += 1
            if dist_min_R < self.metricas_R['min_dist_obst']:
                self.metricas_R['min_dist_obst'] = dist_min_R
            self.metricas_R['kce'] += np.sum(np.abs(uR_seguro)) * self.ts
            aceleracion_R = (uR_seguro - self.metricas_R['prev_vel']) / self.ts
            self.metricas_R['accel_sum'] += np.sum(np.abs(aceleracion_R)) * self.ts
            self.metricas_R['prev_vel'] = uR_seguro.copy()

        if self.ejecutando_benchmark_L: self.t_left += self.ts
        if self.ejecutando_benchmark_R: self.t_right += self.ts

        self.contador += 1
        tiempo_cero = Time()

        arreglo_flechas = MarkerArray()
        m_purga = Marker()
        m_purga.action = Marker.DELETEALL
        arreglo_flechas.markers.append(m_purga)
        
        for idx_hueso, F_3d in fuerzas_R_dict.items():
            if np.linalg.norm(F_3d) > 0.1: 
                pos_origen = puntos_3d[f'right_j{idx_hueso}']
                flecha = generar_flecha_fuerza(pos_origen, F_3d, id_flecha=idx_hueso, factor_escala=0.08)
                flecha.header.stamp = tiempo_cero
                flecha.ns = "fuerza_der"
                arreglo_flechas.markers.append(flecha)
                
        for idx_hueso, F_3d in fuerzas_L_dict.items():
            if np.linalg.norm(F_3d) > 0.1:
                pos_origen = puntos_3d[f'left_j{idx_hueso}']
                flecha = generar_flecha_fuerza(pos_origen, F_3d, id_flecha=idx_hueso+20, factor_escala=0.08)
                flecha.header.stamp = tiempo_cero
                flecha.ns = "fuerza_izq"
                flecha.color.r, flecha.color.g, flecha.color.b = 1.0, 0.2, 0.2
                arreglo_flechas.markers.append(flecha)

        self.pub_fuerzas_R.publish(arreglo_flechas) 

        if self.contador % 5 == 0:
            arreglo_esqueleto = generar_esqueleto_colisiones(puntos_3d, radios_dict, 0.10, 'world')
            for m in arreglo_esqueleto.markers: m.header.stamp = tiempo_cero
            self.pub_esqueleto.publish(arreglo_esqueleto)

        if self.contador % 10 == 0:
            if self.marker_izq.points:
                self.marker_izq.header.stamp = tiempo_cero
                self.pub_rviz.publish(self.marker_izq)
            if self.marker_der.points:
                self.marker_der.header.stamp = tiempo_cero
                self.pub_rviz.publish(self.marker_der)

def main(args=None):
    rclpy.init(args=args)
    nodo = MainDualBrazoLumelsky()
    try: 
        rclpy.spin(nodo)
    except KeyboardInterrupt: 
        pass
    finally:
        nodo.enviar_comandos_cero()
        nodo.destroy_node()
        rclpy.try_shutdown()

if __name__ == '__main__':
    main()