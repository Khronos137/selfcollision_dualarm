import numpy as np
import math

# =====================================================================
# PARÁMETROS GLOBALES DE COLISIÓN (Sugerencia: Migrar a config.yaml)
# =====================================================================
N_POINTS          = 3    # Puntos de control interpolados por eslabón

# Grosores Volumétricos (10 Sub-segmentos para precisión milimétrica)
RADIOS_BRAZOS = {
    1: 0.045,  # j1: Clavícula (Dentro del torso)
    2: 0.045,  # j2: Mitad interna del motor del hombro
    3: 0.075,  # j3: Mitad externa del motor del hombro
    4: 0.050,  # j4: Hueso superior
    5: 0.065,  # j5: Zona pre-antebrazo
    6: 0.065,  # j6: Antebrazo
    7: 0.05,  # j7: Antebrazo delgado
    8: 0.05,  # j8: Pre-muñeca
    9: 0.055,  # j9: Muñeca centro
    10: 0.030  # j10: Efector final
}

CELL_SIZE = max(RADIOS_BRAZOS.values()) * 2

RADIO_TORSO       = 0.12  # Radio de cada esfera del torso [m]
DIST_SEG_TORSO    = 0.05  # Distancia mínima de seguridad brazo-torso [m]
TORSO_N_ESFERAS   = 6     # Número de esferas a lo largo del torso
TORSO_BASE        = np.array([0.0, 0.0, 0.05])
TORSO_TOP         = np.array([0.0, 0.0, 0.75])

# =====================================================================
# 1. GENERACIÓN DE PUNTOS CINEMÁTICOS (11 PUNTOS -> 10 SEGMENTOS)
# =====================================================================
def puntos_articulaciones_der(L, q):
    h, b, l1, l1b, l2, l3 = L[0], L[1], L[2], L[3], L[4], L[5]
    l1a = (l1**2 + l1b**2)**0.5
    q1, q2, q3, q4, q5, q6 = q[0], q[1], q[2] + 0.7854, q[3], q[4], q[5]

    # Puntos Base
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

    # Puntos Intermedios (Para envoltorio volumétrico)
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

    # Puntos Base
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

    # Puntos Intermedios (Para envoltorio volumétrico)
    p1_in = p1 + 0.15 * (p0 - p1)  
    p1_out = p1 + 0.2 * (p2 - p1) 
    p4_out = p4 + 0.2 * (p5 - p4) 
    p5_in = p5 - 0.10 * (p5 - p4)
    p5_out = p5 + 0.35 * (p6 - p5)

    pts = [p0, p1_in, p1, p1_out, p2, p4, p4_out, p5_in, p5, p5_out, p6]
    return [list(pt) for pt in pts]

def obtener_puntos_actuales(q_left, q_right, L):
    pts_L = puntos_articulaciones_izq(L, q_left)
    pts_R = puntos_articulaciones_der(L, q_right)
    puntos = {}
    
    # 11 puntos cinemáticos
    for i in range(11):
        puntos[f'left_j{i}'] = np.array(pts_L[i])
        puntos[f'right_j{i}'] = np.array(pts_R[i])
        
    puntos['torso_base'] = np.array([0.0, 0.0, 0.0])
    puntos['torso_top']  = np.array([0.0, 0.0, 0.7])
    return puntos

# =====================================================================
# 2. UTILIDADES DE GEOMETRÍA Y HASHING ESPACIAL
# =====================================================================
def puntos_control(link_pos, n):
    pts = []
    for i in range(len(link_pos) - 1):
        p0 = np.array(link_pos[i])
        p1 = np.array(link_pos[i + 1])
        for k in range(n):
            t = (k + 0.5) / n
            pts.append((i + 1, p0 + t * (p1 - p0)))  # (seg_idx 1-based, punto)
    return pts

def spatial_hash(pc_L, pc_R, cell_size, radios):
    grid = {}
    for idx, (seg, p) in enumerate(pc_L):
        key = (math.floor(p[0] / cell_size),
               math.floor(p[1] / cell_size),
               math.floor(p[2] / cell_size))
        grid.setdefault(key, {'L': [], 'R': []})['L'].append((idx, seg, p))
        
    for idx, (seg, p) in enumerate(pc_R):
        key = (math.floor(p[0] / cell_size),
               math.floor(p[1] / cell_size),
               math.floor(p[2] / cell_size))
        grid.setdefault(key, {'L': [], 'R': []})['R'].append((idx, seg, p))
        
    colisiones = []
    for cell in grid.values():
        if not cell['L'] or not cell['R']:
            continue
        for idx_l, seg_l, pl in cell['L']:
            r_l = radios.get(seg_l, 0.05)
            for idx_r, seg_r, pr in cell['R']:
                r_r = radios.get(seg_r, 0.05)
                umbral_sq = (r_l + r_r) ** 2
                if np.sum((pl - pr) ** 2) < umbral_sq:
                    colisiones.append((idx_l, pl, idx_r, pr))
    return colisiones

def generar_centros_torso():
    centros = []
    for k in range(TORSO_N_ESFERAS):
        t = (k + 0.5) / TORSO_N_ESFERAS
        centros.append(TORSO_BASE + t * (TORSO_TOP - TORSO_BASE))
    return centros

def check_torso_esferas(pc_brazo, centros_torso, radio_torso, dist_seg, radios):
    result = []
    for idx, (seg, p) in enumerate(pc_brazo):
        r_brazo = radios.get(seg, 0.05)
        umbral = radio_torso + r_brazo + dist_seg
        umbral_sq = umbral ** 2
        for c in centros_torso:
            diff = p - c
            dist_sq_val = float(np.dot(diff, diff))
            if dist_sq_val < umbral_sq:
                d_sup = max(0.0, np.sqrt(dist_sq_val) - radio_torso - r_brazo)
                result.append((idx, p, c, d_sup))
                break
    return result

# =====================================================================
# 3. NÚCLEO DE EVALUACIÓN MULTI-PUNTO (MATRIZ 11x11)
# =====================================================================
def revisar_estado_seguridad(puntos_globales, radios_eslabones):
    alertas = []
    orden = ['j0', 'j1', 'j2', 'j3', 'j4', 'j5', 'j6', 'j7', 'j8', 'j9', 'j10']

    pts_L_joints = [puntos_globales[f'left_{j}']  for j in orden]
    pts_R_joints = [puntos_globales[f'right_{j}'] for j in orden]

    pc_L = puntos_control(pts_L_joints, N_POINTS)
    pc_R = puntos_control(pts_R_joints, N_POINTS)
    pts_L_flat = [p for (_, p) in pc_L]
    pts_R_flat = [p for (_, p) in pc_R]

    seg_pts_L, seg_pts_R = {}, {}
    for seg, pt in pc_L:
        seg_pts_L.setdefault(seg, []).append(pt)
    for seg, pt in pc_R:
        seg_pts_R.setdefault(seg, []).append(pt)

    # Matriz 11x11: Índice 0 = Torso, Índices 1-10 = Eslabones
    matriz_distancias = np.full((11, 11), np.inf)

    # ── Torso esferas ─────────────────────────────────────────────────
    centros_torso = generar_centros_torso()
    torso_col_L = check_torso_esferas(pc_L, centros_torso, RADIO_TORSO, DIST_SEG_TORSO, RADIOS_BRAZOS)
    torso_col_R = check_torso_esferas(pc_R, centros_torso, RADIO_TORSO, DIST_SEG_TORSO, RADIOS_BRAZOS)
    torso_idx_L = {idx for (idx, _, _, _) in torso_col_L}
    torso_idx_R = {idx for (idx, _, _, _) in torso_col_R}

    # Llenado de matriz torso-brazo (1 a 10)
    for seg_i in range(1, 11):
        r_i = RADIOS_BRAZOS.get(seg_i, 0.05)
        if seg_i in seg_pts_L:
            d = min(
                max(0.0, np.linalg.norm(p - c) - RADIO_TORSO - r_i)
                for p in seg_pts_L[seg_i]
                for c in centros_torso
            )
            matriz_distancias[seg_i, 0] = d
            
        if seg_i in seg_pts_R:
            d = min(
                max(0.0, np.linalg.norm(p - c) - RADIO_TORSO - r_i)
                for p in seg_pts_R[seg_i]
                for c in centros_torso
            )
            matriz_distancias[0, seg_i] = d

    # Generación de Alertas Torso
    seen_torso_L, seen_torso_R = set(), set()
    for idx, p_arm, c_torso, d_sup in torso_col_L:
        seg = pc_L[idx][0]
        # Poda: Ignoramos empotramientos base (<= 4)
        if seg > 4 and seg not in seen_torso_L:
            alertas.append(
                f"Colisión Torso: left {orden[seg]} | "
                f"brazo=({p_arm[0]:.3f},{p_arm[1]:.3f},{p_arm[2]:.3f}) | "
                f"dist={d_sup:.3f}m"
            )
            seen_torso_L.add(seg)
            
    for idx, p_arm, c_torso, d_sup in torso_col_R:
        seg = pc_R[idx][0]
        if seg > 4 and seg not in seen_torso_R:
            alertas.append(
                f"Colisión Torso: right {orden[seg]} | "
                f"brazo=({p_arm[0]:.3f},{p_arm[1]:.3f},{p_arm[2]:.3f}) | "
                f"dist={d_sup:.3f}m"
            )
            seen_torso_R.add(seg)

    # ── Suelo ─────────────────────────────────────────────────────────
    for prefix, pts_j in [('left', pts_L_joints), ('right', pts_R_joints)]:
        for i in range(1, len(orden)):
            r_actual = radios_eslabones.get(orden[i], 0.06)
            z_min = min(pts_j[i-1][2], pts_j[i][2])
            if z_min < r_actual:
                alertas.append(
                    f"Colisión Suelo: {prefix} {orden[i]} | z={z_min:.3f}m"
                )

    # ── Brazo-Brazo (Spatial Hashing) ─────────────────────────────────
    col_brazos = spatial_hash(pc_L, pc_R, CELL_SIZE, RADIOS_BRAZOS)
    col_idx_L = {idx_l for (idx_l, _, _, _) in col_brazos}
    col_idx_R = {idx_r for (_, _, idx_r, _) in col_brazos}

    # Llenado de matriz brazo-brazo (11x11)
    for seg_i in range(1, 11):
        r_i = RADIOS_BRAZOS.get(seg_i, 0.05)
        for seg_j in range(1, 11):
            r_j = RADIOS_BRAZOS.get(seg_j, 0.05)
            if seg_i in seg_pts_L and seg_j in seg_pts_R:
                d = min(
                    max(0.0, np.linalg.norm(pl - pr) - r_i - r_j)
                    for pl in seg_pts_L[seg_i]
                    for pr in seg_pts_R[seg_j]
                )
                matriz_distancias[seg_i, seg_j] = d

    # Generación de Alertas Brazo-Brazo
    seen_pairs = set()
    for idx_l, pl, idx_r, pr in col_brazos:
        seg_l = pc_L[idx_l][0]
        seg_r = pc_R[idx_r][0]
        pair = (seg_l, seg_r)
        
        # Poda: Ignorar choques entre partes empotradas
        if pair not in seen_pairs and (seg_l > 4 or seg_r > 4):
            dist = np.linalg.norm(pl - pr)
            pmid = (pl + pr) / 2
            alertas.append(
                f"Auto-Colisión: left_{orden[seg_l]} ↔ right_{orden[seg_r]} | "
                f"dist={dist:.3f}m | "
                f"punto_medio=({pmid[0]:.3f},{pmid[1]:.3f},{pmid[2]:.3f})"
            )
            seen_pairs.add(pair)

    hay_colision = len(alertas) > 0

    viz_data = {
        'pc_L':           pc_L,
        'pc_R':           pc_R,
        'pts_L':          pts_L_flat,
        'pts_R':          pts_R_flat,
        'col_idx_L':      col_idx_L,
        'col_idx_R':      col_idx_R,
        'torso_idx_L':    torso_idx_L,
        'torso_idx_R':    torso_idx_R,
        'col_brazos':     col_brazos,
        'torso_col_L':    torso_col_L,
        'torso_col_R':    torso_col_R,
        'pts_L_joints':   pts_L_joints,
        'pts_R_joints':   pts_R_joints,
        'centros_torso':  centros_torso,
    }

    return hay_colision, alertas, matriz_distancias, viz_data