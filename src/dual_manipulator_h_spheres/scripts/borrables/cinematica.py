import numpy as np
import math

JOINTS = [
    ([0.000,  0.000,  0.126], 'z'),   # j1
    ([0.000,  0.069,  0.033], 'y'),   # j2
    ([0.030, -0.0115, 0.264], 'y'),   # j3
    ([0.195, -0.0575, 0.030], 'x'),   # j4
    ([0.063,  0.045,  0.000], 'y'),   # j5
    ([0.123, -0.045,  0.000], 'x'),   # j6
]

# ¡TUS MEDIDAS EXACTAS! Offsets locales [X, Y, Z] en metros
# Aquí empujamos el centro del volumen hacia atrás en el eje Y (el verde)
OFFSETS_COLISION = [
    [0.0,  0.0,   0.0],  # j1: Sin cambios
    [0.0, -0.06,  0.0],  # j2: Restar 5 cm en Y
    [0.0, -0.06,  0.0],  # j3: Restar 5 cm en Y
    [0.0,  0.0,   0.0],  # j4: Sin cambios
    [0.0, -0.04,  0.0],  # j5: Restar 3 cm en Y
    [0.0,  0.0,   0.0],  # j6: Sin cambios
]

TORSO_HEIGHT = 0.8

def _rpy_to_R(roll, pitch, yaw):
    Rx = np.array([[1, 0, 0], [0, math.cos(roll), -math.sin(roll)], [0, math.sin(roll), math.cos(roll)]])
    Ry = np.array([[math.cos(pitch), 0, math.sin(pitch)], [0, 1, 0], [-math.sin(pitch), 0, math.cos(pitch)]])
    Rz = np.array([[math.cos(yaw), -math.sin(yaw), 0], [math.sin(yaw), math.cos(yaw), 0], [0, 0, 1]])
    return Rz @ Ry @ Rx

_R_left  = _rpy_to_R(-math.pi / 2, 0,  math.pi)
_t_left  = np.array([0.0, -0.1, 0.7])
_R_right = _rpy_to_R( math.pi / 2, 0, -math.pi)
_t_right = np.array([0.0,  0.1, 0.7])

def to_base(pos: np.ndarray, arm: str) -> np.ndarray:
    R, t = (_R_left, _t_left) if arm == 'left' else (_R_right, _t_right)
    return R @ pos + t

def fk_matrices(q: np.ndarray):
    """Calcula la Matriz T completa para saber la rotación exacta del motor."""
    T = np.eye(4)
    matrices = []
    for i, (xyz, axis) in enumerate(JOINTS):
        c, s = math.cos(q[i]), math.sin(q[i])
        Ti = np.eye(4)
        Ti[:3, 3] = xyz
        if axis == 'z':   Ti[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        elif axis == 'y': Ti[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        elif axis == 'x': Ti[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        T = T @ Ti
        matrices.append(T.copy())
    return matrices

def obtener_puntos_globales(q_left: np.ndarray, q_right: np.ndarray) -> dict:
    puntos = {}
    for prefix, q, arm in [('left', q_left, 'left'), ('right', q_right, 'right')]:
        matrices_T = fk_matrices(q)
        
        locales = {f'{prefix}_j0': np.zeros(3)}
        
        for i in range(6):
            T = matrices_T[i]
            # Convertimos tu offset a un vector 3D de coordenadas homogéneas
            v_offset = np.array([OFFSETS_COLISION[i][0], 
                                 OFFSETS_COLISION[i][1], 
                                 OFFSETS_COLISION[i][2], 
                                 1.0])
            
            # ¡ALGEBRA LINEAL AL RESCATE! Desplazamos y rotamos el punto en 1 paso
            punto_final = T @ v_offset
            locales[f'{prefix}_j{i+1}'] = punto_final[:3]
            
        for key, pos_local in locales.items():
            puntos[key] = to_base(pos_local, arm)
            
    puntos['torso_base'] = np.array([0.0, 0.0, 0.0])
    puntos['torso_top']  = np.array([0.0, 0.0, TORSO_HEIGHT])
    return puntos