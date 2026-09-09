import numpy as np
import math
from cinematica import JOINTS  # Importamos la geometría del otro archivo

def fk_efector(q):
    """Cinemática directa específica para la punta del efector (link 6)."""
    T = np.eye(4)
    for i, (xyz, axis) in enumerate(JOINTS):
        c, s = math.cos(q[i]), math.sin(q[i])
        Ti = np.eye(4)
        Ti[:3, 3] = xyz
        if axis == 'z':   Ti[:3, :3] = [[c, -s, 0], [s, c, 0], [0, 0, 1]]
        elif axis == 'y': Ti[:3, :3] = [[c, 0, s], [0, 1, 0], [-s, 0, c]]
        elif axis == 'x': Ti[:3, :3] = [[1, 0, 0], [0, c, -s], [0, s, c]]
        T = T @ Ti
    return T[:3, 3].copy()

def jacobiano_numerico(q):
    """Calcula la matriz Jacobiana 3x6 mediante diferencias finitas."""
    h0 = fk_efector(q)
    J = np.zeros((3, 6))
    delta = 1e-6
    for i in range(6):
        qp = q.copy()
        qp[i] += delta
        J[:, i] = (fk_efector(qp) - h0) / delta
    return J

# def trayectoria_deseada(t):
#     """Genera el punto XYZ objetivo en el instante t."""
#     r = 0.2      # Radio del círculo
#     omega = 0.2  # Velocidad angular
#     x = r * math.cos(omega * t) + 0.38
#     y = r * math.sin(omega * t)
#     z = 0.42
#     return np.array([x, y, z])

def calcular_velocidad_articular(q_actual, hd_actual, hd_siguiente, To, W):
    """
    Ahora es una función pura: tú le das los puntos exactos que debe seguir 
    (hd_actual y hd_siguiente) y ella solo devuelve las velocidades de los motores.
    """
    h_real = fk_efector(q_actual)
    error = hd_actual - h_real
    
    J = jacobiano_numerico(q_actual)
    J_pinv = np.linalg.pinv(J)
    
    v_articular = J_pinv @ (hd_siguiente - W @ error - h_real) / To
    
    return v_articular, error, h_real