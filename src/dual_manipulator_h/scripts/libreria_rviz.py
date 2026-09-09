from visualization_msgs.msg import Marker, MarkerArray
from geometry_msgs.msg import Point, Quaternion
import numpy as np
import math

# =====================================================================
# 1. GENERADOR DE TRAYECTORIAS (Anillos estáticos)
# =====================================================================
def crear_marcador_trayectoria(puntos, frame_id, ns, marker_id, r, g, b):
    """
    Genera un Marker LINE_STRIP genérico a partir de una lista de puntos XYZ.
    """
    marcador = Marker()
    marcador.header.frame_id = frame_id
    marcador.ns = ns
    marcador.id = marker_id
    
    marcador.type = Marker.LINE_STRIP
    marcador.action = Marker.ADD
    
    marcador.scale.x = 0.01  # Grosor de la línea
    
    marcador.color.r = float(r)
    marcador.color.g = float(g)
    marcador.color.b = float(b)
    marcador.color.a = 0.8 
    
    marcador.lifetime.sec = 0 
    marcador.lifetime.nanosec = 0
    
    for pt in puntos:
        p = Point()
        p.x = float(pt[0])
        p.y = float(pt[1])
        p.z = float(pt[2])
        marcador.points.append(p)
        
    return marcador


# =====================================================================
# 2. GENERADORES DE CÁPSULAS 3D (Colisiones avanzadas)
# =====================================================================
def _crear_capsula(p1, p2, radio, id_num, ns, r, g, b, frame_id):
    """Genera un cilindro 3D real entre p1 y p2 usando cuaterniones."""
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns
    m.id = id_num
    m.type = Marker.CYLINDER
    m.action = Marker.ADD
    
    v = p2 - p1
    dist = np.linalg.norm(v)
    mid = p1 + v / 2.0
    
    m.pose.position.x = float(mid[0])
    m.pose.position.y = float(mid[1])
    m.pose.position.z = float(mid[2])
    
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

    m.scale.x = float(radio * 2.0)
    m.scale.y = float(radio * 2.0)
    m.scale.z = float(dist)
    m.color.r = float(r); m.color.g = float(g); m.color.b = float(b); m.color.a = 0.6 
    return m

def _crear_esfera_articulacion(p, radio, id_num, ns, r, g, b, frame_id):
    """Genera esferas en las articulaciones para cerrar la cápsula perfecta."""
    m = Marker()
    m.header.frame_id = frame_id
    m.ns = ns + "_esferas"
    m.id = id_num
    m.type = Marker.SPHERE
    m.action = Marker.ADD
    m.pose.position.x = float(p[0]); m.pose.position.y = float(p[1]); m.pose.position.z = float(p[2])
    m.pose.orientation = Quaternion(x=0.0, y=0.0, z=0.0, w=1.0)
    m.scale.x = m.scale.y = m.scale.z = float(radio * 2.0)
    m.color.r = float(r); m.color.g = float(g); m.color.b = float(b); m.color.a = 0.6
    return m

def generar_esqueleto_colisiones(puntos_globales, radios_eslabones, radio_torso, frame_id):
    arreglo = MarkerArray()
    identificador = 0
    
    # 1. Torso
    tb = puntos_globales['torso_base']
    tt = puntos_globales['torso_top']
    arreglo.markers.append(_crear_capsula(tb, tt, radio_torso, identificador, 'torso', 0.0, 0.5, 1.0, frame_id))
    identificador += 1
    # arreglo.markers.append(_crear_esfera_articulacion(tb, radio_torso, identificador, 'torso', 0.0, 0.5, 1.0, frame_id)); identificador += 1
    # arreglo.markers.append(_crear_esfera_articulacion(tt, radio_torso, identificador, 'torso', 0.0, 0.5, 1.0, frame_id)); identificador += 1

    # 2. Brazos
    # 2. Brazos -> Naranja
    orden = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9', 'j10'] # Abarcando los 11 puntos
    for prefix in ['left', 'right']:
        for i in range(1, len(orden)):
            p1 = puntos_globales[f'{prefix}_{orden[i-1]}']
            p2 = puntos_globales[f'{prefix}_{orden[i]}']
            
            # Extraemos el radio escalable de este eslabón
            r_actual = radios_eslabones[orden[i]]
            
            arreglo.markers.append(_crear_capsula(p1, p2, r_actual, identificador, f'{prefix}_huesos', 1.0, 0.5, 0.0, frame_id))
            identificador += 1
            
        for i in range(len(orden)):
            p = puntos_globales[f'{prefix}_{orden[i]}']
            
            # Para la primera articulación (j0), copiamos el radio de j1 por estética
            r_actual = radios_eslabones.get(orden[i], radios_eslabones['j1'])
            
            # arreglo.markers.append(_crear_esfera_articulacion(p, r_actual, identificador, f'{prefix}_huesos', 1.0, 0.5, 0.0, frame_id))
            identificador += 1

    return arreglo

def generar_flecha_fuerza(p_inicio, vector_fuerza, id_flecha=0, factor_escala=0.1, borrar=False):
    """
    Genera un Marker tipo ARROW para visualizar el vector de fuerza de repulsión en RViz.
    """
    flecha = Marker()
    flecha.header.frame_id = "world"
    flecha.ns = "fuerza"
    flecha.id = id_flecha
    flecha.type = Marker.ARROW
    
    # Si pedimos borrarla (porque ya no hay peligro), mandamos la acción DELETE
    if borrar:
        flecha.action = Marker.DELETE
        return flecha
        
    flecha.action = Marker.ADD
    
    # Escala de la flecha (x: grosor tallo, y: ancho cabeza, z: alto cabeza)
    flecha.scale.x = 0.01
    flecha.scale.y = 0.02
    flecha.scale.z = 0.02
    
    # Color (Amarillo/Naranja para peligro)
    flecha.color.a = 1.0
    flecha.color.r = 1.0
    flecha.color.g = 0.0
    flecha.color.b = 1.0
    
    # Puntos: Inicio (Codo/Muñeca) y Fin (Dirección escalada de la fuerza)
    p1 = Point(x=p_inicio[0], y=p_inicio[1], z=p_inicio[2])
    p2 = Point(
        x=p_inicio[0] + (vector_fuerza[0] * factor_escala), 
        y=p_inicio[1] + (vector_fuerza[1] * factor_escala), 
        z=p_inicio[2] + (vector_fuerza[2] * factor_escala)
    )
    flecha.points = [p1, p2]
    
    return flecha