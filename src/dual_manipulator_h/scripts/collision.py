import numpy as np

# =====================================================================
# 1. GENERACIÓN DE PUNTOS CINEMÁTICOS (9 PUNTOS)
# =====================================================================

def puntos_articulaciones_der(L, q):
    h, b, l1, l1b, l2, l3 = L[0], L[1], L[2], L[3], L[4], L[5]
    l1a = (l1**2 + l1b**2)**0.5
    
    q1, q2, q3, q4, q5, q6 = q[0], q[1], q[2] + 0.7854, q[3], q[4], q[5]

    # 1. Puntos Puros
    p0 = np.array([0.0, 0.0, h])
    p1 = np.array([0.0, -b, h])
    
    p2 = np.array([
        l1 * np.sin(q2) * np.cos(q1),
        -b - l1 * np.cos(q2),
        h + l1 * np.sin(q2) * np.sin(q1)
    ])
    
    p3 = np.array([
        l1a * np.sin(q2 + 0.113) * np.cos(q1),
        -b - l1a * np.cos(q2 + 0.113),
        h + l1a * np.sin(q2 + 0.113) * np.sin(q1)
    ])
    
    p4 = np.array([
        p3[0] - l1b * np.cos(q3 + 0.67) * np.cos(q2 + 0.113) * np.cos(q1) + l1b * np.sin(q3 + 0.67) * np.sin(q2 + 0.113) * np.cos(q1),
        p3[1] - l1b * np.cos(q3 + 0.67) * np.sin(q2 + 0.113) - l1b * np.cos(q2 + 0.113) * np.sin(q3 + 0.67),
        p3[2] - l1b * np.cos(q3 + 0.67) * np.cos(q2 + 0.113) * np.sin(q1) + l1b * np.sin(q3 + 0.67) * np.sin(q2 + 0.113) * np.sin(q1)
    ])
    
    p5 = np.array([
        p4[0] + l2 * (np.cos(q3 + 0.67) * np.sin(q2 + 0.113) * np.cos(q1) + np.cos(q2 + 0.113) * np.sin(q3 + 0.67) * np.cos(q1)),
        p4[1] - l2 * (np.cos(q3 + 0.67) * np.cos(q2 + 0.113) - np.sin(q3 + 0.67) * np.sin(q2 + 0.113)),
        p4[2] + l2 * (np.cos(q3 + 0.67) * np.sin(q2 + 0.113) * np.sin(q1) + np.cos(q2 + 0.113) * np.sin(q3 + 0.67) * np.sin(q1))
    ])
    
    dx6 = - l3 * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q4) * (np.cos(q3 + 0.67) * np.cos(q2 + 0.113) * np.cos(q1) - np.sin(q3 + 0.67) * np.sin(q2 + 0.113) * np.cos(q1))) - np.cos(q5) * (np.cos(q3 + 0.67) * np.sin(q2 + 0.113) * np.cos(q1) + np.cos(q2 + 0.113) * np.sin(q3 + 0.67) * np.cos(q1)))
    dy6 = - l3 * (np.cos(q5) * (np.cos(q3 + 0.67) * np.cos(q2 + 0.113) - np.sin(q3 + 0.67) * np.sin(q2 + 0.113)) - np.cos(q4) * np.sin(q5) * (np.cos(q3 + 0.67) * np.sin(q2 + 0.113) + np.cos(q2 + 0.113) * np.sin(q3 + 0.67)))
    dz6 = + l3 * (np.cos(q5) * (np.cos(q3 + 0.67) * np.sin(q2 + 0.113) * np.sin(q1) + np.cos(q2 + 0.113) * np.sin(q3 + 0.67) * np.sin(q1)) + np.sin(q5) * (np.cos(q4) * (np.cos(q3 + 0.67) * np.cos(q2 + 0.113) * np.sin(q1) - np.sin(q3 + 0.67) * np.sin(q2 + 0.113) * np.sin(q1)) + np.cos(q1) * np.sin(q4)))
    p6 = np.array([p5[0] + dx6, p5[1] + dy6, p5[2] + dz6])

    p1_in = p1 + 0.15 * (p0 - p1)  
    p1_out = p1 + 0.2 * (p2 - p1) 
    p4_out = p4 + 0.2 * (p5 - p4) 
    p5_in = p5 - 0.10 * (p5 - p4)
    p5_out = p5 + 0.35 * (p6 - p5)

    pts = [p0, p1_in, p1, p1_out, p2, p4, p4_out, p5_in, p5, p5_out, p6]
    return [list(pt) for pt in pts]



def puntos_articulaciones_izq(L, q):
    h, b, l1, l1b, l2, l3 = L[0], L[1], L[2], L[3], L[4], L[5]
    l1a = (l1**2 + l1b**2)**0.5
    
    q1, q2, q3, q4, q5, q6 = q[0], q[1], q[2] + 0.7854, q[3], q[4], q[5] 

    p0 = np.array([0.0, 0.0, h])
    p1 = np.array([0.0, b, h])
    
    p2 = np.array([
        l1 * np.sin(q2) * np.cos(q1),
        b + l1 * np.cos(q2),
        h - l1 * np.sin(q2) * np.sin(q1)
    ])
    
    p3 = np.array([
        l1a * np.sin(q2 + 0.113) * np.cos(q1),
        b + l1a * np.cos(q2 + 0.113),
        h - l1a * np.sin(q2 + 0.113) * np.sin(q1)
    ])
    
    p4 = np.array([
        p3[0] - l1b * np.cos(q3 + 0.67) * np.cos(q2 + 0.113) * np.cos(q1) + l1b * np.sin(q3 + 0.67) * np.sin(q2 + 0.113) * np.cos(q1),
        p3[1] + l1b * np.cos(q3 + 0.67) * np.sin(q2 + 0.113) + l1b * np.cos(q2 + 0.113) * np.sin(q3 + 0.67),
        p3[2] + l1b * np.cos(q3 + 0.67) * np.cos(q2 + 0.113) * np.sin(q1) - l1b * np.sin(q3 + 0.67) * np.sin(q2 + 0.113) * np.sin(q1)
    ])
    
    p5 = np.array([
        p4[0] + l2 * (np.cos(q3 + 0.67) * np.sin(q2 + 0.113) * np.cos(q1) + np.cos(q2 + 0.113) * np.sin(q3 + 0.67) * np.cos(q1)),
        p4[1] + l2 * (np.cos(q3 + 0.67) * np.cos(q2 + 0.113) - np.sin(q3 + 0.67) * np.sin(q2 + 0.113)),
        p4[2] - l2 * (np.cos(q3 + 0.67) * np.sin(q2 + 0.113) * np.sin(q1) + np.cos(q2 + 0.113) * np.sin(q3 + 0.67) * np.sin(q1))
    ])
    
    dx6 = - l3 * (np.sin(q5) * (np.sin(q1) * np.sin(q4) - np.cos(q4) * (np.cos(q3 + 0.67) * np.cos(q2 + 0.113) * np.cos(q1) - np.sin(q3 + 0.67) * np.sin(q2 + 0.113) * np.cos(q1))) - np.cos(q5) * (np.cos(q3 + 0.67) * np.sin(q2 + 0.113) * np.cos(q1) + np.cos(q2 + 0.113) * np.sin(q3 + 0.67) * np.cos(q1)))
    dy6 = + l3 * (np.cos(q5) * (np.cos(q3 + 0.67) * np.cos(q2 + 0.113) - np.sin(q3 + 0.67) * np.sin(q2 + 0.113)) - np.cos(q4) * np.sin(q5) * (np.cos(q3 + 0.67) * np.sin(q2 + 0.113) + np.cos(q2 + 0.113) * np.sin(q3 + 0.67)))
    dz6 = - l3 * (np.cos(q5) * (np.cos(q3 + 0.67) * np.sin(q2 + 0.113) * np.sin(q1) + np.cos(q2 + 0.113) * np.sin(q3 + 0.67) * np.sin(q1)) + np.sin(q5) * (np.cos(q4) * (np.cos(q3 + 0.67) * np.cos(q2 + 0.113) * np.sin(q1) - np.sin(q3 + 0.67) * np.sin(q2 + 0.113) * np.sin(q1)) + np.cos(q1) * np.sin(q4)))
    p6 = np.array([p5[0] + dx6, p5[1] + dy6, p5[2] + dz6])

    # Envoltorio del motor (Misma interpolación, cero problemas de signos)
    p1_in = p1 + 0.15 * (p0 - p1)  
    p1_out = p1 + 0.2 * (p2 - p1) 
    p4_out = p4 + 0.2 * (p5 - p4) 
    p5_in = p5 - 0.10 * (p5 - p4)
    p5_out = p5 + 0.35 * (p6 - p5)

    pts = [p0, p1_in, p1, p1_out, p2, p4, p4_out, p5_in, p5, p5_out, p6]
    return [list(pt) for pt in pts]


# =====================================================================
# 2. SISTEMA DE DETECCIÓN DE COLISIONES
# =====================================================================

def distancia_segmento_segmento(p1, p2, p3, p4):
    """Algoritmo de Lumelsky (Se queda igual)"""
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

def revisar_estado_seguridad(puntos_globales, radios_eslabones, radio_torso=0.15):
    alertas = []
    orden = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9', 'j10'] # ¡11 Elementos!
    
    # MATRIZ 9x9
    matriz_distancias = np.zeros((11, 11))
    
    torso_b = puntos_globales['torso_base']
    torso_t = puntos_globales['torso_top']
    
    for prefix in ['left', 'right']:
        for i in range(1, len(orden)):
            p1 = puntos_globales[f'{prefix}_{orden[i-1]}']
            p2 = puntos_globales[f'{prefix}_{orden[i]}']
            
            r_actual = radios_eslabones[orden[i]]
            
            if min(p1[2], p2[2]) < r_actual:
                alertas.append(f"Colisión Suelo: {prefix} {orden[i]}")
                
            dist_torso = distancia_segmento_segmento(p1, p2, torso_b, torso_t)
            
            if prefix == 'left': matriz_distancias[i, 0] = dist_torso 
            else: matriz_distancias[0, i] = dist_torso
            
            # El torso ignora las colisiones con los primeros 4 eslabones que están empotrados en él
            if i > 4: 
                if dist_torso < (r_actual + radio_torso):
                    alertas.append(f"Colisión Torso: {prefix} {orden[i]}")

    for i in range(1, len(orden)):
        for j in range(1, len(orden)):
            L1 = puntos_globales[f'left_{orden[i-1]}']
            L2 = puntos_globales[f'left_{orden[i]}']
            R1 = puntos_globales[f'right_{orden[j-1]}']
            R2 = puntos_globales[f'right_{orden[j]}']
            
            dist_brazos = distancia_segmento_segmento(L1, L2, R1, R2)
            matriz_distancias[i, j] = dist_brazos
            
            # Filtro de imposibilidad física ajustado a la nueva cadena (ignorar choques internos de hombros)
            if i <= 4 and j <= 4:
                continue
            
            r_L = radios_eslabones[orden[i]]
            r_R = radios_eslabones[orden[j]]
            
            if dist_brazos < (r_L + r_R):
                alertas.append(f"Auto-Colisión: left_{orden[i]} con right_{orden[j]}")

    hay_colision = len(alertas) > 0
    return hay_colision, alertas, matriz_distancias

def obtener_puntos_actuales(q_left, q_right, L):
    pts_L = puntos_articulaciones_izq(L, q_left)
    pts_R = puntos_articulaciones_der(L, q_right)
    
    puntos = {}
    for i in range(11): # ¡Cambio vital a 11!
        puntos[f'left_j{i}'] = np.array(pts_L[i])
        puntos[f'right_j{i}'] = np.array(pts_R[i])
        
    puntos['torso_base'] = np.array([0.0, 0.0, 0.0])
    puntos['torso_top']  = np.array([0.0, 0.0, 0.7]) 
    return puntos