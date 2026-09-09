import numpy as np
from math import cos, sin
from scipy.spatial.transform import Rotation as R

class ArmKinematics:
    def __init__(self):
        # Constantes de diseño según tu C++ y XACRO
        self.l1a = 0.2657
        self.l1b = 0.03
        self.l2 = 0.258
        self.l3 = 0.149
        self.cq3 = 0.116
        self.cq2 = 0.113

    def get_jacobian_and_fk_local(self, q):
        """
        Calcula la FK y Jacobiana estrictamente en el marco LOCAL del brazo.
        """
        q1, q2, q3, q4, q5, q6 = q
        
        # Pre-cálculo de senos y cosenos
        s1, c1 = sin(q1), cos(q1)
        s23, c23 = sin(q2 + q3 + self.cq2 - self.cq3), cos(q2 + q3 + self.cq2 - self.cq3)
        s2, c2 = sin(q2 + self.cq2), cos(q2 + self.cq2)
        s4, c4 = sin(q4), cos(q4)
        s5, c5 = sin(q5), cos(q5)

        # --- CINEMÁTICA DIRECTA LOCAL ---
        x_l = (self.l1a * c2 + self.l1b * c23 + self.l2 * c23 + self.l3 * (c5 * c23 + s5 * c4 * s23)) * c1
        y_l = (self.l1a * c2 + self.l1b * c23 + self.l2 * c23 + self.l3 * (c5 * c23 + s5 * c4 * s23)) * s1
        z_l = self.l1a * s2 + (self.l1b + self.l2) * s23 + self.l3 * (c5 * s23 - s5 * c4 * c23)
        p_local = np.array([x_l, y_l, z_l])

        # --- JACOBIANA SIMBÓLICA LOCAL (3x6) ---
        Jl = np.zeros((3, 6))

        # dX
        Jl[0,0] = -y_l
        Jl[0,1] = c1 * (-self.l1a*s2 - (self.l1b + self.l2)*s23 + self.l3*(-c5*s23 + s5*c4*c23))
        Jl[0,2] = c1 * (-(self.l1b + self.l2)*s23 + self.l3*(-c5*s23 + s5*c4*c23))
        Jl[0,3] = c1 * (self.l3 * s5 * s4 * s23)
        Jl[0,4] = c1 * (self.l3 * (-s5*c23 + c5*c4*s23))
        # dY
        Jl[1,0] = x_l
        Jl[1,1] = s1 * (-self.l1a*s2 - (self.l1b + self.l2)*s23 + self.l3*(-c5*s23 + s5*c4*c23))
        Jl[1,2] = s1 * (-(self.l1b + self.l2)*s23 + self.l3*(-c5*s23 + s5*c4*c23))
        Jl[1,3] = s1 * (self.l3 * s5 * s4 * s23)
        Jl[1,4] = s1 * (self.l3 * (-s5*c23 + c5*c4*s23))
        # dZ
        Jl[2,1] = self.l1a*c2 + (self.l1b + self.l2)*c23 + self.l3*(c5*c23 + s5*c4*s23)
        Jl[2,2] = (self.l1b + self.l2)*c23 + self.l3*(c5*c23 + s5*c4*s23)
        Jl[2,3] = self.l3 * s5 * s4 * c23
        Jl[2,4] = self.l3 * (-s5*s23 - c5*c4*c23)

        return Jl, p_local

# Instancia global para evitar NameError
_kin = ArmKinematics()

def calcular_velocidad_articular(q_actual, hd, K_gain, prefix, q_null=None):
    """
    Ley de Control Pura: q_p = J# * K * tanh(hd - h_real) + (I - J#J) * q_null
    """
    # 1. Definir Rotación (Rb) y Traslación (Pb) con SCIPY basándose en tu XACRO
    if prefix == 'left':
        # rpy="${-pi/2} 0 ${pi}" | xyz="0.0 -0.1 0.3"
        Rb = R.from_euler('xyz', [-np.pi/2, 0, np.pi]).as_matrix()
        Pb = np.array([0.0, -0.1, 0.3])
    else:
        # rpy="${pi/2} 0 ${-pi}" | xyz="0.0 0.1 0.3"
        Rb = R.from_euler('xyz', [np.pi/2, 0, -np.pi]).as_matrix()
        Pb = np.array([0.0, 0.1, 0.3])

    # 2. Obtener Cinemática y Jacobiana LOCAL
    Jl, p_local = _kin.get_jacobian_and_fk_local(q_actual)
    
    # 3. TRADUCCIÓN AL MUNDO (La solución al problema)
    J_global = Rb @ Jl
    h_real = (Rb @ p_local) + Pb  # <- ROTACIÓN + TRASLACIÓN DE LA BASE
    
    J_pinv = np.linalg.pinv(J_global)

    # 4. Error y Tarea Primaria (Regulación con Tanh)
    error = hd - h_real
    v_task = K_gain * np.tanh(error)

    # 5. Tarea Secundaria (Espacio Nulo)
    if q_null is None:
        q_null = np.zeros(6)
    
    P_null = np.eye(6) - (J_pinv @ J_global)
    v_articular = (J_pinv @ v_task) + (P_null @ q_null)
    
    return v_articular, error, h_real