import numpy as np

# IMPORTAMOS LAS JACOBIANAS EXACTAS
from libreria_cinematica_der import derJac6ManH
from libreria_cinematica_izq import derJac6IzqManH

# =====================================================================
# 1. MAPA TOPOLÓGICO DE LA CADENA CINEMÁTICA
# =====================================================================
MAPA_INFLUENCIA = {
    0: 0, 1: 0, 2: 0,   # Base y Hombro estático
    3: 2, 4: 2,         # Brazo superior (q1, q2)
    5: 3, 6: 3, 7: 3, 8: 3, # Codo y Antebrazo (q1, q2, q3)
    9: 5, 10: 5         # Muñeca y Mano (q1, q2, q3, q4, q5)
}

# =====================================================================
# 2. CÁLCULO DE LA MATRIZ DE PESOS DISTRIBUIDA
# =====================================================================
def calcular_matriz_W_multipunto(radares, q_actual, q_min, q_max, d_safe, d_crit, rho, margen_limite):
    """
    Evalúa el peligro de TODOS los huesos y le asigna al motor el peso 
    del hueso más crítico que ese motor debe mover.
    """
    W = np.eye(6)
    
    for motor_idx in range(6):
        # 1. Límites Articulares
        peso_limite = 1.0
        rango = q_max[motor_idx] - q_min[motor_idx]
        margen = margen_limite * rango 
        
        if q_actual[motor_idx] > (q_max[motor_idx] - margen):
            peso_limite += ((q_actual[motor_idx] - (q_max[motor_idx] - margen)) / margen) ** 2 * 100
        elif q_actual[motor_idx] < (q_min[motor_idx] + margen):
            peso_limite += (((q_min[motor_idx] + margen) - q_actual[motor_idx]) / margen) ** 2 * 100
            
        # 2. Evaluación Multi-Punto
        peso_max_colision = 1.0
        
        for punto_idx, datos in radares.items():
            motores_activos = MAPA_INFLUENCIA.get(punto_idx, 0)
            
            if motor_idx < motores_activos:
                d = datos['dist']
                if d_crit < d < d_safe:
                    peso_local = 1.0 + rho * ((d_safe - d) / (d - d_crit)) ** 2
                    if peso_local > peso_max_colision: peso_max_colision = peso_local
                elif d <= d_crit:
                    peso_max_colision = 1000.0
                    
        W[motor_idx, motor_idx] = max(peso_limite, peso_max_colision)
        
    return W

# =====================================================================
# 3. JACOBIANA SIMBÓLICA EXACTA
# =====================================================================
def calcular_Jc_simbolica(q, brazo, idx_punto, L):
    """
    Construye la Jacobiana exacta para CADA hueso individual.
    """
    h, b, l1, l1b, l2, l3 = L
    if idx_punto <= 2:   L_mod = [h, b, 0.0, 0.0, 0.0, 0.0]
    elif idx_punto == 3: L_mod = [h, b, l1*0.2, 0.0, 0.0, 0.0]
    elif idx_punto == 4: L_mod = [h, b, l1, 0.0, 0.0, 0.0]
    elif idx_punto == 5: L_mod = [h, b, l1, l1b, 0.0, 0.0]
    elif idx_punto == 6: L_mod = [h, b, l1, l1b, l2*0.2, 0.0]
    elif idx_punto == 7: L_mod = [h, b, l1, l1b, l2*0.9, 0.0]
    elif idx_punto == 8: L_mod = [h, b, l1, l1b, l2, 0.0]
    elif idx_punto == 9: L_mod = [h, b, l1, l1b, l2, l3*0.35]
    else:                L_mod = [h, b, l1, l1b, l2, l3]

    if brazo == 'left': return derJac6IzqManH(L_mod, q)
    else:               return derJac6ManH(L_mod, q)

# =====================================================================
# 4. CAMPOS POTENCIALES DISTRIBUIDOS (SUPERPOSICIÓN ANALÍTICA)
# =====================================================================
def calcular_fuerzas_multipunto(radares, puntos_3d, prefijo_brazo, L_params, d_safe, q_actual, q_min, q_max, eta_col, eta_lim, margen_limite, eta_vort, v_des):
    """
    Suma las fuerzas de repulsión de TODOS los huesos en peligro simultáneamente.
    """
    q_rep_total = np.zeros(6)
    fuerzas_3d_rviz = {i: np.zeros(3) for i in range(1, 11)} 
    
    for punto_idx, datos in radares.items():
        d_actual = datos['dist']
        
        if d_actual < d_safe:
            p_hueso = puntos_3d[f'{prefijo_brazo}_j{punto_idx}']
            p_obstaculo = datos['pos']
            
            d_efectiva = max(d_actual, 1e-4) 
            vector_escape = p_hueso - p_obstaculo
            norma = np.linalg.norm(vector_escape)
            
            if norma > 1e-6:
                u = vector_escape / norma
                fuerza_base = ((1.0 / d_efectiva) - (1.0 / d_safe)) * (1.0 / (d_efectiva**2))
                
                # ¡DESATAMOS A LA BESTIA! 5.0 N maximos para no quedarse trabados.
                V_max_escape = 1.0 
                
                # Repulsión
                F_rep = V_max_escape * np.tanh((eta_col * fuerza_base) / V_max_escape) * u
                
                # Vórtice Orientado a la Tarea
                v_tangencial = v_des - np.dot(v_des, u) * u
                norma_tang = np.linalg.norm(v_tangencial)
                if norma_tang > 1e-6:
                    u_tang = v_tangencial / norma_tang
                    F_vort = V_max_escape * np.tanh((eta_vort * fuerza_base) / V_max_escape) * u_tang
                else:
                    F_vort = np.zeros(3)
                    
                F_total = F_rep + F_vort
                fuerzas_3d_rviz[punto_idx] = F_total
                
                # Superposición: Sumamos la fuerza usando la Jacobiana local del hueso
                Jc_local = calcular_Jc_simbolica(q_actual, prefijo_brazo, punto_idx, L_params)
                q_rep_total += Jc_local.T @ F_total

    # Límites Articulares (Solo 1 vez al final de la sumatoria)
    for i in range(6):
        margen = margen_limite * (q_max[i] - q_min[i])
        if q_actual[i] > (q_max[i] - margen): 
            dist_lim = max(q_max[i] - q_actual[i], 1e-4)
            q_rep_total[i] -= 2.0 * np.tanh((eta_lim * (1.0 / dist_lim)**2) / 2.0)
        elif q_actual[i] < (q_min[i] + margen): 
            dist_lim = max(q_actual[i] - q_min[i], 1e-4)
            q_rep_total[i] += 2.0 * np.tanh((eta_lim * (1.0 / dist_lim)**2) / 2.0)

    # ==========================================================
    # NORMALIZACIÓN DE SUPERPOSICIÓN (Anti-Torsión y Vibraciones)
    # ==========================================================
    # Si la suma de las fuerzas de todos los huesos pide una velocidad 
    # articular suicida, escalamos el vector completo para mantener la 
    # dirección de escape, pero a una velocidad máxima segura.
    norma_q_rep = np.linalg.norm(q_rep_total)
    max_vel_articular = 2.5 # Radianes por segundo máximos permitidos en el espacio nulo
    
    if norma_q_rep > max_vel_articular:
        q_rep_total = (q_rep_total / norma_q_rep) * max_vel_articular
            
    return q_rep_total, fuerzas_3d_rviz