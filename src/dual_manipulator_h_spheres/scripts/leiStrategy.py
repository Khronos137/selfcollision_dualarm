"""
Baseline: real-time kinematics-based self-collision avoidance of Lei et al. (2020).
Pure Python (no ROS). Equation numbers in brackets refer to Lei et al.;
the manuscript describes the same steps in Eqs. (32)-(37).
"""
import time
import numpy as np

from cinematica_puntos import puntos_cadena, jacobianos_cadena, jacobiano_en_segmento
from collision import beta_minimo, PHI
from benchmark_common import pinv_dls, escalar_norma_infinito, QDOT_MAX

PARAMETROS_LEI = {
    'V_max': 1.5,        # maximum repulsive speed [m/s]        (Table 2)
    'alpha_sig': 20.0,   # sigmoid shape factor                  (Lei Eq. 8)
    'beta0': 0.7,        # sigmoid midpoint                      (Lei Eq. 8)
    'beta_act': 1.0,     # activation: beta_lrmin <= 1           (Lei Sec. 3.1)
    'Ke': 5.5,           # task tracking gain (same as proposed)
    'lam': 0.05,         # DLS damping (same as proposed)
    'qdot_max': QDOT_MAX,
}


class ControladorLei:

    def __init__(self, parametros=None):
        self.p = dict(PARAMETROS_LEI)
        if parametros:
            self.p.update(parametros)

    def reiniciar(self):
        pass

    def _brazo(self, q, lado, P, pdes, vdes, par, propio, ajeno):
        p = self.p
        JC = jacobianos_cadena(q, lado)
        J = JC[10]
        e = pdes - P[10]
        v_task = vdes + p['Ke'] * e                                  # v_task (Eq. 32 of manuscript)
        J_pinv = pinv_dls(J, p['lam'])

        if par is None or par['beta'] > p['beta_act']:
            return J_pinv @ v_task, False

        # Lei Eq. (8): sigmoid magnitude of the repulsive velocity
        v_mag = p['V_max'] / (1.0 + np.exp(p['alpha_sig'] * (par['beta'] - p['beta0'])))
        # Lei Eqs. (9)-(10): direction from the other sphere centre to the own one
        k, t, c_own = propio
        c_other = ajeno
        diff = c_own - c_other
        n = np.linalg.norm(diff)
        v_rv = (diff / n) * v_mag if n > 1e-9 else np.zeros(3)

        # Lei Eqs. (12)-(13): joint velocity of the joints proximal to the control point,
        # using the Jacobian of the control point (sphere centre); distal joints = 0.
        q_re = np.zeros(6)
        phi = PHI[k]
        if phi > 0:
            Jcp = jacobiano_en_segmento(JC, k, t)[:, :phi]
            q_re[:phi] = pinv_dls(Jcp, p['lam']) @ v_rv
        # Lei Eq. (14): equivalent end-effector repulsive velocity
        v_re_end = J @ q_re
        # Lei Eqs. (15)-(16): projection on the task velocity (magnitude modulation only)
        nt, nr = np.linalg.norm(v_task), np.linalg.norm(v_re_end)
        if nt > 1e-4 and nr > 1e-4:
            cos_g = float(v_re_end @ v_task) / (nr * nt)
            v_pro = nr * cos_g * v_task / nt
        else:
            v_pro = np.zeros(3)
        # Lei Eqs. (17)-(18)
        return J_pinv @ (v_task + v_pro), True

    def calcular(self, q_L, q_R, pdes_L, vdes_L, pdes_R, vdes_R):
        t0 = time.perf_counter()
        P_L = puntos_cadena(q_L, 'left')
        P_R = puntos_cadena(q_R, 'right')
        par = beta_minimo(P_L, P_R)
        t_col = (time.perf_counter() - t0) * 1e3

        qL, actL = self._brazo(q_L, 'left', P_L, pdes_L, vdes_L, par,
                               (par['kL'], par['tL'], par['cL']), par['cR'])
        qR, actR = self._brazo(q_R, 'right', P_R, pdes_R, vdes_R, par,
                               (par['kR'], par['tR'], par['cR']), par['cL'])
        qL, sL = escalar_norma_infinito(qL, self.p['qdot_max'])
        qR, sR = escalar_norma_infinito(qR, self.p['qdot_max'])
        info = {'collision_ms': t_col, 'clear_L': par['clear_fisico'], 'clear_R': par['clear_fisico'],
                'scale_L': sL, 'scale_R': sR, 'beta_min': par['beta'], 'activo': actL or actR,
                'P': {'L': P_L, 'R': P_R}}
        return qL, qR, info
