#!/usr/bin/env python3
import rclpy
from rclpy.node import Node
from std_msgs.msg import String
from dual_manipulator_benchmark.msg import BezierTrajectory
from gazebo_msgs.msg import ContactsState  # IMPORTANTE: Para leer los Bumpers
import yaml
import os
import csv                                 # IMPORTANTE: Para escribir el archivo
from datetime import datetime              # IMPORTANTE: Para las marcas de tiempo
from ament_index_python.packages import get_package_share_directory
from functools import partial              # IMPORTANTE: Para pasar argumentos extra a los callbacks

class BenchmarkDirector(Node):
    def __init__(self):
        super().__init__('benchmark_director')
        
        # --- VARIABLES DEL ÁRBITRO (¡Ahora son independientes!) ---
        self.colision_L = False
        self.colision_R = False
        
        # EL CSV MAESTRO E INDESTRUCTIBLE
        self.csv_filename = "resultados_benchmark_master.csv"
        
        # Revisamos si el archivo ya existe para NO sobreescribirlo ni repetir encabezados
        if not os.path.exists(self.csv_filename):
            with open(self.csv_filename, mode='w', newline='') as file:
                writer = csv.writer(file)
                # --- MODIFICACIÓN: Añadidas las 3 columnas de tiempo al final ---
                writer.writerow(['Timestamp', 'Algorithm', 'Arm', 'Result', 'RMSE_m', 'TJV_rad', 'Jerk_rad_s', 'MinDist_m', 'FinalErr_m', 'TimeFactor', 'TC_Mean_ms', 'TC_Max_ms', 'TCol_Mean_ms'])
            self.get_logger().info(f"📄 Archivo CSV creado: {self.csv_filename}")
        else:
            self.get_logger().info(f"📄 Archivo CSV detectado. Se añadirán datos a: {self.csv_filename}")
            
        # --- SUSCRIPTORES DEL ÁRBITRO ---
        self.create_subscription(String, '/benchmark/metricas', self.callback_metricas, 10)
        
        # Suscriptores a los Bumpers Izquierdos
        topicos_left = [
            '/benchmark/colisiones/left_end_link', '/benchmark/colisiones/left_link5', '/benchmark/colisiones/left_link6'
        ]
        for topico in topicos_left:
            self.create_subscription(ContactsState, topico, partial(self.callback_bumper, brazo="LEFT"), 10)

        # Suscriptores a los Bumpers Derechos
        topicos_right = [
            '/benchmark/colisiones/right_end_link', '/benchmark/colisiones/right_link5', '/benchmark/colisiones/right_link6'
        ]
        for topico in topicos_right:
            self.create_subscription(ContactsState, topico, partial(self.callback_bumper, brazo="RIGHT"), 10)

        # --- PUBLICADORES Y SUSCRIPTORES NORMALES ---
        self.pub_traj_L = self.create_publisher(BezierTrajectory, '/benchmark/left_arm_traj', 10)
        self.pub_traj_R = self.create_publisher(BezierTrajectory, '/benchmark/right_arm_traj', 10)
        self.create_subscription(String, '/benchmark/status', self.cb_status, 10)
        self.pub_logger_cmd = self.create_publisher(String, '/benchmark/logger_cmd', 10)
        
        self.trayectoria_actual = 0
        self.total_trayectorias = 0
        self.estado = "INICIO" # INICIO, EJECUTANDO, RETORNANDO_HOME, ESPERANDO
        self.tiempo_llegada_home = 0.0
        
        paquete_share = get_package_share_directory('dual_manipulator_benchmark')
        ruta_yaml = os.path.join(paquete_share, 'config', 'trayectorias_50.yaml')
        self.rutas = self.cargar_rutas_yaml(ruta_yaml)
        
        if self.rutas:
            self.total_trayectorias = len(self.rutas)
            self.get_logger().info(f"DIRECTOR INICIADO: {self.total_trayectorias} trayectorias. Arrancando...")
            self.timer = self.create_timer(0.5, self.maquina_estados)
        else:
            self.get_logger().error("YAML vacío o corrupto.")

    def callback_bumper(self, msg, brazo):
        # Si hay contacto, penalizamos SOLO al brazo que reportó el golpe
        if len(msg.states) > 0:
            if brazo == "LEFT" and not self.colision_L:
                self.get_logger().fatal("💥 ÁRBITRO: ¡COLISIÓN DETECTADA EN EL BRAZO IZQUIERDO!")
                self.colision_L = True
            elif brazo == "RIGHT" and not self.colision_R:
                self.get_logger().fatal("💥 ÁRBITRO: ¡COLISIÓN DETECTADA EN EL BRAZO DERECHO!")
                self.colision_R = True

    def callback_metricas(self, msg):
        datos = msg.data.split('|')
        
        if datos[0] == "METRICS":
            algoritmo = datos[1]  # LUMELSKY o LEI
            brazo = datos[2]      # LEFT o RIGHT
            resultado = datos[3]  # SUCCESS o TIMEOUT
            
            # 🛑 JUSTICIA DEL ÁRBITRO: Castigamos solo al culpable
            if brazo == "LEFT" and self.colision_L:
                resultado = "COLLISION"
            elif brazo == "RIGHT" and self.colision_R:
                resultado = "COLLISION"
                
            rmse = datos[4]
            tjv = datos[5]
            jerk = datos[6]
            min_dist = datos[7]
            final_err = datos[8]
            time_factor = datos[9]
            
            # --- MODIFICACIÓN: LECTURA INTELIGENTE DE CPU ---
            # Si el mensaje trae 13 partes (Brazo Izquierdo), extraemos los tiempos.
            # Si trae menos (Brazo Derecho), dejamos las celdas en blanco.
            t_cpu_mean = datos[10] if len(datos) >= 13 else ""
            t_cpu_max  = datos[11] if len(datos) >= 13 else ""
            t_col_mean = datos[12] if len(datos) >= 13 else ""
            
            timestamp = datetime.now().strftime("%H:%M:%S")
            
            # Escribir la fila en el CSV usando modo Append ('a')
            with open(self.csv_filename, mode='a', newline='') as file:
                writer = csv.writer(file)
                # Añadimos las 3 variables nuevas al final de la fila
                writer.writerow([timestamp, algoritmo, brazo, resultado, rmse, tjv, jerk, min_dist, final_err, time_factor, t_cpu_mean, t_cpu_max, t_col_mean])
                
            self.get_logger().info(f"📊 Métricas Guardadas -> {algoritmo} | {brazo} | Veredicto Final: {resultado}")

    def cargar_rutas_yaml(self, filepath):
        try:
            with open(filepath, 'r') as f:
                datos = yaml.safe_load(f)
                return datos.get('trayectorias', [])
        except Exception as e:
            return []

    def cb_status(self, msg: String):
        if msg.data == "REACHED" and self.estado == "EJECUTANDO":
            self.get_logger().info(f"✅ Trayectoria #{self.trayectoria_actual} completada.")
            
            # Guardar datos en el logger obsoleto (Opcional, lo puedes borrar luego si quieres)
            cmd_msg = String()
            cmd_msg.data = f"SAVE_{self.trayectoria_actual}"
            self.pub_logger_cmd.publish(cmd_msg)
            
            # Comandar Retorno Seguro a Home (Enviando bandera t_total = -1.0)
            self.get_logger().info("🏠 Comandando retorno suave a Home...")
            msg_home = BezierTrajectory()
            msg_home.t_total = -1.0 
            self.pub_traj_L.publish(msg_home)
            self.pub_traj_R.publish(msg_home)
            self.estado = "RETORNANDO_HOME"
            
        elif msg.data == "HOME_REACHED" and self.estado == "RETORNANDO_HOME":
            self.get_logger().info("🤖 Brazos en Home. Estabilizando vibraciones (1.5s)...")
            self.tiempo_llegada_home = self.get_clock().now().nanoseconds / 1e9
            self.estado = "ESPERANDO"

    def maquina_estados(self):
        if self.estado == "INICIO":
            if self.trayectoria_actual < self.total_trayectorias:
                self.enviar_nueva_trayectoria(self.trayectoria_actual)
                self.estado = "EJECUTANDO"
            else:
                self.get_logger().info("🎉 BENCHMARK FINALIZADO EXITOSAMENTE.")
                self.estado = "FIN"
                self.timer.cancel()
                
        elif self.estado == "ESPERANDO":
            tiempo_actual = self.get_clock().now().nanoseconds / 1e9
            if (tiempo_actual - self.tiempo_llegada_home) >= 1.5:
                # Ya pasó el tiempo de gracia, iniciar siguiente experimento
                self.trayectoria_actual += 1
                self.estado = "INICIO"

    def enviar_nueva_trayectoria(self, indice):
        # 🛑 RESET DEL ÁRBITRO PARA AMBOS BRAZOS EN EL NUEVO EXPERIMENTO
        self.colision_L = False 
        self.colision_R = False 
        
        datos = self.rutas[indice]
        msg_L, msg_R = BezierTrajectory(), BezierTrajectory()
        
        msg_L.t_total = datos['t_total']; msg_R.t_total = datos['t_total']
        msg_L.p0 = datos['left_arm']['p0']; msg_R.p0 = datos['right_arm']['p0']
        msg_L.p1 = datos['left_arm']['p1']; msg_R.p1 = datos['right_arm']['p1']
        msg_L.p2 = datos['left_arm']['p2']; msg_R.p2 = datos['right_arm']['p2']
        msg_L.p3 = datos['left_arm']['p3']; msg_R.p3 = datos['right_arm']['p3']
        
        self.pub_traj_L.publish(msg_L); self.pub_traj_R.publish(msg_R)
        
        cmd_msg = String()
        cmd_msg.data = "START"
        self.pub_logger_cmd.publish(cmd_msg)
        self.get_logger().info(f"▶️ Lanzando Experimento #{indice} (Duración: {msg_L.t_total}s)")

def main(args=None):
    rclpy.init(args=args)
    nodo = BenchmarkDirector()
    try: rclpy.spin(nodo)
    except KeyboardInterrupt: pass
    finally: nodo.destroy_node(); rclpy.try_shutdown()

if __name__ == '__main__':
    main()