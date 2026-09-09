#!/usr/bin/env python3
import yaml
import random
import os
import math

def generar_punto_en_esfera(centro, radio):
    """Genera un punto aleatorio dentro de una esfera tridimensional."""
    u = random.random()
    v = random.random()
    theta = u * 2.0 * math.pi
    phi = math.acos(2.0 * v - 1.0)
    
    # CORRECCIÓN PARA PYTHON 3.10 (ROS 2 Humble): 
    # Usar ** (1.0/3.0) en lugar de math.cbrt()
    r = radio * (random.random() ** (1.0 / 3.0))
    
    x = centro[0] + r * math.sin(phi) * math.cos(theta)
    y = centro[1] + r * math.sin(phi) * math.sin(theta)
    z = centro[2] + r * math.cos(phi)
    
    # Redondear a 3 decimales para un YAML limpio
    return [round(x, 3), round(y, 3), round(z, 3)]

def generar_punto_intermedio(p_inicio, p_fin, varianza):
    """Genera un punto intermedio P1 o P2 añadiendo ruido a la interpolación lineal."""
    return [
        round(p_inicio[0] + (p_fin[0] - p_inicio[0]) * random.random() + random.uniform(-varianza, varianza), 3),
        round(p_inicio[1] + (p_fin[1] - p_inicio[1]) * random.random() + random.uniform(-varianza, varianza), 3),
        round(p_inicio[2] + (p_fin[2] - p_inicio[2]) * random.random() + random.uniform(-varianza, varianza), 3)
    ]

def main():
    num_trayectorias = 50
    datos = {'trayectorias': []}
    
    # Puntos Home estáticos (Libres de colisión)
    home_L = [0.42, 0.55, 0.65]
    home_R = [0.42, -0.55, 0.65]

    # Zonas de meta cruzadas (Para forzar colisiones)
    # Brazo Izquierdo invade el lado derecho
    centro_meta_L = [0.2, -0.05, 0.7]
    radio_meta_L = 0.05
    
    # Brazo Derecho invade el lado izquierdo (Esfera más grande)
    centro_meta_R = [0.2, 0.05, 0.7]
    radio_meta_R = 0.15

    print(f"Generando {num_trayectorias} trayectorias de Bézier al azar...")

    for i in range(num_trayectorias):
        # 1. Definir tiempos aleatorios
        t_total = round(random.uniform(2.5, 5.0), 2)
        
        # 2. Generar P3 (Goals cruzados)
        p3_L = generar_punto_en_esfera(centro_meta_L, radio_meta_L)
        p3_R = generar_punto_en_esfera(centro_meta_R, radio_meta_R)
        
        # 3. Generar P1 y P2 (Añadiendo curvatura aleatoria)
        p1_L = generar_punto_intermedio(home_L, p3_L, 0.3)
        p2_L = generar_punto_intermedio(home_L, p3_L, 0.3)
        
        p1_R = generar_punto_intermedio(home_R, p3_R, 0.3)
        p2_R = generar_punto_intermedio(home_R, p3_R, 0.3)

        trayectoria = {
            'id': i,
            't_total': float(t_total),
            'left_arm': {
                'p0': home_L,
                'p1': p1_L,
                'p2': p2_L,
                'p3': p3_L
            },
            'right_arm': {
                'p0': home_R,
                'p1': p1_R,
                'p2': p2_R,
                'p3': p3_R
            }
        }
        datos['trayectorias'].append(trayectoria)

    # Definir ruta de guardado
    ruta_script = os.path.dirname(os.path.abspath(__file__))
    ruta_config = os.path.join(ruta_script, '..', 'config', 'trayectorias_50.yaml')
    
    # Crear carpeta config si no existe
    os.makedirs(os.path.dirname(ruta_config), exist_ok=True)

    with open(ruta_config, 'w') as file:
        yaml.dump(datos, file, default_flow_style=None, sort_keys=False)

    print(f"¡Éxito! Archivo YAML generado en: {ruta_config}")

if __name__ == '__main__':
    main()