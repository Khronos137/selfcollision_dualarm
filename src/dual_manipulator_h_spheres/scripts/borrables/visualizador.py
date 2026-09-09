import numpy as np
import math
from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion

class VisualizadorRViz:
    def __init__(self, nodo, funcion_trayectoria, To=0.1, tfin=60.0):
        self.pub_ml_des = nodo.create_publisher(Marker, '/left/trajectory/desired_marker', 10)
        self.pub_mr_des = nodo.create_publisher(Marker, '/right/trajectory/desired_marker', 10)
        self.pub_colisiones = nodo.create_publisher(MarkerArray, '/colisiones/cilindros', 10)

        self.ml_des = self._crear_marcador_linea('left_link1', 'left_desired', 0, 1.0, 0.0, 0.0)
        self.mr_des = self._crear_marcador_linea('right_link1', 'right_desired', 1, 1.0, 0.0, 1.0)

        pasos = int(tfin / To) + 1
        for i in range(pasos):
            hd_l = funcion_trayectoria(i * To, brazo='left')
            self.ml_des.points.append(Point(x=float(hd_l[0]), y=float(hd_l[1]), z=float(hd_l[2])))
            
            hd_r = funcion_trayectoria(i * To, brazo='right')
            self.mr_des.points.append(Point(x=float(hd_r[0]), y=float(hd_r[1]), z=float(hd_r[2])))

    def _crear_marcador_linea(self, frame, ns, id_num, r, g, b):
        m = Marker()
        m.header.frame_id = frame
        m.ns = ns; m.id = id_num; m.type = Marker.LINE_STRIP; m.action = Marker.ADD
        m.scale.x = 0.005; m.color.r = r; m.color.g = g; m.color.b = b; m.color.a = 1.0
        return m

    def publicar(self, stamp):
        self.ml_des.header.stamp = stamp
        self.mr_des.header.stamp = stamp
        self.pub_ml_des.publish(self.ml_des)
        self.pub_mr_des.publish(self.mr_des)

    # =========================================================
    # MATEMÁTICA PARA CILINDROS Y CÁPSULAS 3D REALES
    # =========================================================
    def _crear_capsula(self, p1, p2, radio, id_num, ns, r, g, b, a=0.4):
        """Genera un cilindro 3D real entre p1 y p2 usando cuaterniones."""
        m = Marker()
        m.header.frame_id = 'base_link'
        m.ns = ns
        m.id = id_num
        m.type = Marker.CYLINDER
        m.action = Marker.ADD
        
        v = p2 - p1
        dist = np.linalg.norm(v)
        mid = p1 + v / 2.0
        
        # Posición (siempre en el centro del segmento)
        m.pose.position.x = float(mid[0])
        m.pose.position.y = float(mid[1])
        m.pose.position.z = float(mid[2])
        
        # Orientación (Rotar el eje Z del cilindro para que apunte hacia el vector v)
        if dist > 1e-5:
            v_dir = v / dist
            z_axis = np.array([0.0, 0.0, 1.0])
            dot = np.dot(z_axis, v_dir)
            if dot > 0.9999:
                m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
            elif dot < -0.9999:
                m.pose.orientation = Quaternion(x=1.0, y=0.0, z=0.0, w=0.0)
            else:
                cross = np.cross(z_axis, v_dir)
                qw = 1.0 + dot
                norm = math.sqrt(qw**2 + cross[0]**2 + cross[1]**2 + cross[2]**2)
                m.pose.orientation = Quaternion(x=float(cross[0]/norm), y=float(cross[1]/norm), z=float(cross[2]/norm), w=float(qw/norm))
        else:
            m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)

        # Dimensiones (X e Y son el diámetro, Z es el largo)
        m.scale.x = float(radio * 2.0)
        m.scale.y = float(radio * 2.0)
        m.scale.z = float(dist)
        
        m.color.r = float(r); m.color.g = float(g); m.color.b = float(b); m.color.a = float(a)
        return m

    def _crear_esfera_articulacion(self, p, radio, id_num, ns, r, g, b, a=0.4):
        """Genera las esferas en las articulaciones para cerrar la cápsula perfecta."""
        m = Marker()
        m.header.frame_id = 'base_link'
        m.ns = ns + "_esferas"
        m.id = id_num
        m.type = Marker.SPHERE
        m.action = Marker.ADD
        m.pose.position.x = float(p[0]); m.pose.position.y = float(p[1]); m.pose.position.z = float(p[2])
        m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
        m.scale.x = m.scale.y = m.scale.z = float(radio * 2.0)
        m.color.r = float(r); m.color.g = float(g); m.color.b = float(b); m.color.a = float(a)
        return m

    def publicar_cilindros_colision(self, stamp, puntos_globales, radio_eslabon=0.03, radio_torso=0.15):
        arreglo = MarkerArray()
        identificador = 0
        
        # 1. Torso (Cilindro + Esferas) -> Azul
        tb = puntos_globales['torso_base']
        tt = puntos_globales['torso_top']
        arreglo.markers.append(self._crear_capsula(tb, tt, radio_torso, identificador, 'torso', 0.0, 0.5, 1.0))
        identificador += 1
        arreglo.markers.append(self._crear_esfera_articulacion(tb, radio_torso, identificador, 'torso', 0.0, 0.5, 1.0)); identificador += 1
        arreglo.markers.append(self._crear_esfera_articulacion(tt, radio_torso, identificador, 'torso', 0.0, 0.5, 1.0)); identificador += 1

        # 2. Brazos (Cilindros + Esferas) -> Naranja
        # orden = ['j0', 'j1', 'j3', 'j4', 'j5', 'j6']
        orden = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6']
        for prefix in ['left', 'right']:
            # Dibujamos los huesos (cilindros)
            for i in range(1, len(orden)):
                p1 = puntos_globales[f'{prefix}_{orden[i-1]}']
                p2 = puntos_globales[f'{prefix}_{orden[i]}']
                arreglo.markers.append(self._crear_capsula(p1, p2, radio_eslabon, identificador, f'{prefix}_huesos', 1.0, 0.5, 0.0))
                identificador += 1
            
            # Dibujamos los codos/muñecas (esferas) para redondear las puntas
            for i in range(len(orden)):
                p = puntos_globales[f'{prefix}_{orden[i]}']
                arreglo.markers.append(self._crear_esfera_articulacion(p, radio_eslabon, identificador, f'{prefix}_huesos', 1.0, 0.5, 0.0))
                identificador += 1

        # Enviamos todas las formas geométricas a RViz
        for m in arreglo.markers:
            m.header.stamp = stamp
            
        self.pub_colisiones.publish(arreglo)