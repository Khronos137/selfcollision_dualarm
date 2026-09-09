import numpy as np

def distancia_segmento_segmento(p1, p2, p3, p4):
    """Algoritmo de Lumelsky para distancia mínima en 3D."""
    u = p2 - p1; v = p4 - p3; w = p1 - p3
    a = np.dot(u, u); b = np.dot(u, v); c = np.dot(v, v)
    d = np.dot(u, w); e = np.dot(v, w)
    D = a * c - b * b
    
    if D < 1e-8:
        sN = 0.0; sD = 1.0; tN = e; tD = c
    else:
        sN = (b * e - c * d); tN = (a * e - b * d); sD = D; tD = D
        if sN < 0.0: sN = 0.0; tN = e; tD = c
        elif sN > sD: sN = sD; tN = e + b; tD = c

    if tN < 0.0:
        tN = 0.0
        if -d < 0.0: sN = 0.0
        elif -d > a: sN = sD
        else: sN = -d; sD = a
    elif tN > tD:
        tN = tD
        if (-d + b) < 0.0: sN = 0.0
        elif (-d + b) > a: sN = sD
        else: sN = (-d + b); sD = a

    sc = 0.0 if abs(sN) < 1e-8 else sN / sD
    tc = 0.0 if abs(tN) < 1e-8 else tN / tD
    
    dP = w + (sc * u) - (tc * v)
    return np.linalg.norm(dP)

def revisar_estado_seguridad(puntos_globales, radio_eslabon=0.03, radio_torso=0.15):
    """
    Evalúa colisiones Suelo, Torso y Brazo-Brazo usando la constelación de puntos.
    Devuelve un booleano (True = Hay colisión) y una lista de alertas.
    """
    alertas = []
    
    # Secuencia de eslabones a evaluar (j0 a j1, j1 a j3, etc.)
    orden = ['j0', 'j1', 'j3', 'j4', 'j5', 'j6']
    
    # 1. Torso y Suelo
    torso_b = puntos_globales['torso_base']
    torso_t = puntos_globales['torso_top']
    
    for prefix in ['left', 'right']:
        for i in range(1, len(orden)):
            p1 = puntos_globales[f'{prefix}_{orden[i-1]}']
            p2 = puntos_globales[f'{prefix}_{orden[i]}']
            
            # Suelo (Z = 0)
            if min(p1[2], p2[2]) < radio_eslabon:
                alertas.append(f"Colisión Suelo: {prefix} {orden[i]}")
                
            # Torso (Pase libre para j0 y j1 porque son el hombro)
            if i > 1: 
                dist_torso = distancia_segmento_segmento(p1, p2, torso_b, torso_t)
                if dist_torso < (radio_eslabon + radio_torso):
                    alertas.append(f"Colisión Torso: {prefix} {orden[i]}")

    # 2. Brazo vs Brazo
    for i in range(1, len(orden)):
        for j in range(1, len(orden)):
            L1 = puntos_globales[f'left_{orden[i-1]}']; L2 = puntos_globales[f'left_{orden[i]}']
            R1 = puntos_globales[f'right_{orden[j-1]}']; R2 = puntos_globales[f'right_{orden[j]}']
            
            dist_brazos = distancia_segmento_segmento(L1, L2, R1, R2)
            if dist_brazos < (radio_eslabon * 2):
                alertas.append(f"Auto-Colisión: left_{orden[i]} con right_{orden[j]}")

    hay_colision = len(alertas) > 0
    return hay_colision, alertas