"""
Proposed controller: dynamically weighted null-space self-collision avoidance.
Pure Python (no ROS) so it can be unit-tested and simulated offline.

Equation numbers refer to the manuscript (revision 2).
"""
import time
import numpy as np

from cinematica_puntos import puntos_cadena, jacobianos_cadena, jacobiano_en_segmento
from collision import estado_geometrico, PHI, K
from benchmark_common import pinv_dls, escalar_norma_infinito, QDOT_MAX, Q_MIN, Q_MAX

# =====================================================================
# PARAMETERS (Table 2)
# =====================================================================
PARAMETROS_POR_DEFECTO = {
    'd_safe': 0.06,        # safe clearance threshold [m] (pilot study, Phase 2)
    'd_crit': 0.02,        # critical clearance threshold [m] (pilot study, Phase 2)
    'Ke': 5.5,             # task tracking gain
    'lam': 0.05,           # DLS damping factor
    'alpha': 0.15,         # null-space low-pass filter factor
    'rho_L': 30.0,         # collision-weight growth rate, left (primary) arm
    'rho_R': 500.0,        # collision-weight growth rate, right (subordinate) arm
    'Xi': 1000.0,          # maximum collision weight (Eq. 12)
    'eta_col': 2.0,        # repulsive scaling gain (Eq. 17)
    'eta_vort_L': 0.0,     # vortex scaling gain, left arm (Eq. 23)
    'eta_vort_R': 8.0,     # vortex scaling gain, right arm (Eq. 23)
    'V_max': 1.0,          # maximum escape velocity [m/s] (Eqs. 17, 23)
    'K_f': 1.0,            # admittance gain (Eq. 15)
    'gamma': 0.1,          # joint-limit warning margin (Eq. 10)
    'sigma': 100.0,        # joint-limit stiffness (Eq. 10)
    'eps_d': 1e-4,         # regularization of f(d) (Eq. 18) [m]
    'eps_p': 1e-6,         # regularization of u_k (Eq. 19) [m]
    'eps_u': 1e-6,         # regularization of u_tang (Eq. 22) [m/s]
    'v_tang_min': 1e-3,    # below this tangential speed the vortex keeps its last direction [m/s]
    'qdot_max': QDOT_MAX,  # joint velocity limit [rad/s]
    'usar_ponderacion': True,   # ablation switch: collision-dependent weighting H_col
}

# Presets for the ablation and priority studies (R1-C5, R1-C19, R2-C7)
VARIANTES = {
    'full': {},
    'no_weighting': {'usar_ponderacion': False},
    'no_vortex': {'eta_vort_L': 0.0, 'eta_vort_R': 0.0},
    'no_weighting_no_vortex': {'usar_ponderacion': False, 'eta_vort_L': 0.0, 'eta_vort_R': 0.0},
    'swapped': {'rho_L': 500.0, 'rho_R': 30.0, 'eta_vort_L': 8.0, 'eta_vort_R': 0.0},
    'symmetric': {'rho_L': 500.0, 'rho_R': 500.0, 'eta_vort_L': 8.0, 'eta_vort_R': 8.0},
}


# =====================================================================
# 1. MULTI-POINT DYNAMIC WEIGHTING MATRIX W (Eqs. 9-12)
# =====================================================================
def H_lim(q, i, p):
    dq = p['gamma'] * (Q_MAX[i] - Q_MIN[i])
    if q[i] > Q_MAX[i] - dq:
        return 1.0 + p['sigma'] * ((q[i] - (Q_MAX[i] - dq)) / dq) ** 2
    if q[i] < Q_MIN[i] + dq:
        return 1.0 + p['sigma'] * (((Q_MIN[i] + dq) - q[i]) / dq) ** 2
    return 1.0


def H_local(d, rho, p):
    """Eq. (12): continuous and bounded collision weight (saturated at Xi)."""
    if d <= p['d_crit']:
        return p['Xi']
    if d >= p['d_safe']:
        return 1.0
    return min(p['Xi'], 1.0 + rho * ((p['d_safe'] - d) / (d - p['d_crit'])) ** 2)


def matriz_W(amenazas, q, rho, p):
    W = np.ones(6)
    for i in range(6):                          # joint index i (0-based) -> joint i+1
        h_col = 1.0
        if p['usar_ponderacion']:
            for k, a in amenazas.items():
                if i < PHI[k]:                      # k in S_i  (Eq. 11)
                    h_col = max(h_col, H_local(a['d'], rho, p))
        W[i] = max(H_lim(q, i, p), h_col)       # Eq. (9)
    return W


# =====================================================================
# 2. SECONDARY TASK: REPULSIVE + TASK-ORIENTED VORTEX FIELD (Eqs. 15-23)
# =====================================================================
def _f(d, p):
    de = d + p['eps_d']
    return (1.0 / de - 1.0 / p['d_safe']) / de ** 2          # Eq. (18)


def _direccion_escape(a, memoria_k):
    """Eq. (19) with fallbacks for coincident closest points (R1-C12, R2-C6)."""
    # Fallback 1: torso -> radial direction from the torso axis
    if a['tipo'] == 'torso':
        r = np.array([a['p'][0], a['p'][1], 0.0])
        if np.linalg.norm(r) > 1e-6:
            return r / np.linalg.norm(r)
    # Fallback 2: last valid escape direction of this capsule
    if memoria_k.get('u') is not None:
        return memoria_k['u']
    # Fallback 3: any direction orthogonal to the obstacle axis
    dobs = a['dir_obs'] / (np.linalg.norm(a['dir_obs']) + 1e-12)
    aux = np.array([0.0, 0.0, 1.0]) if abs(dobs[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    n2 = np.cross(dobs, aux)
    return n2 / np.linalg.norm(n2)


def campo_secundario(amenazas, JC, J_pinv, xdot_des, eta_vort, p, memoria):
    """Returns qdot0 (Eq. 15), the Cartesian fields (for RViz) and the active set size."""
    qdot0 = np.zeros(6)
    fuerzas = {}
    v_task_ee = J_pinv @ xdot_des                       # J^+ xdot_des (Eq. 20)
    for k in range(1, K + 1):
        a = amenazas.get(k)
        mem = memoria.setdefault(k, {'u': None, 'u_tang': None})
        if a is None or a['d'] >= p['d_safe'] or PHI[k] == 0:
            memoria[k] = {'u': None, 'u_tang': None}    # leaving the active set resets memory
            continue

        Jc = jacobiano_en_segmento(JC, k, a['s'])        # closest point on capsule k (Eq. 14)
        if a['tipo'] == 'intra':                         # relative Jacobian of both closest points
            _, k_obs, t_obs = a['obs']
            Jc_map = Jc - jacobiano_en_segmento(JC, k_obs, t_obs)
        else:
            Jc_map = Jc

        diff = a['p'] - a['p_obs']
        n = np.linalg.norm(diff)
        u = diff / (n + p['eps_p']) if n > 1e-6 else _direccion_escape(a, mem)   # Eq. (19)
        u_unit = u / (np.linalg.norm(u) + 1e-12)
        mem['u'] = u_unit

        fd = _f(a['d'], p)
        F_rep = p['V_max'] * np.tanh(p['eta_col'] * fd / p['V_max']) * u          # Eq. (17)

        F_vort = np.zeros(3)
        if eta_vort > 0.0:
            v_task = Jc @ v_task_ee                                                  # Eq. (20)
            v_tang = v_task - (v_task @ u_unit) * u_unit                             # Eq. (21)
            nt = np.linalg.norm(v_tang)
            if nt >= p['v_tang_min']:
                u_t = v_tang / (nt + p['eps_u'])                                     # Eq. (22)
                mem['u_tang'] = u_t / (np.linalg.norm(u_t) + 1e-12)
            elif mem.get('u_tang') is not None:
                # Fallback: desired motion vanished -> keep the last contouring direction,
                # re-orthogonalised with respect to the current escape direction.
                u_t = mem['u_tang'] - (mem['u_tang'] @ u_unit) * u_unit
                u_t = u_t / (np.linalg.norm(u_t) + 1e-12)
            else:
                u_t = np.zeros(3)
            F_vort = p['V_max'] * np.tanh(eta_vort * fd / p['V_max']) * u_t          # Eq. (23)

        F_total = F_rep + F_vort                                                     # Eq. (16)
        qdot0 += p['K_f'] * (Jc_map.T @ F_total)                                     # Eq. (15)
        fuerzas[k] = (a['p'], F_total)
    return qdot0, fuerzas


# =====================================================================
# 3. CONTROLLER
# =====================================================================
class ControladorPropuesto:

    def __init__(self, parametros=None):
        self.p = dict(PARAMETROS_POR_DEFECTO)
        if parametros:
            self.p.update(parametros)
        self.reiniciar()

    def reiniciar(self):
        self.q0f = {'L': np.zeros(6), 'R': np.zeros(6)}       # filter state (Eq. 24)
        self.memoria = {'L': {}, 'R': {}}

    def calcular(self, q_L, q_R, pdes_L, vdes_L, pdes_R, vdes_R):
        p = self.p
        t0 = time.perf_counter()
        P = {'L': puntos_cadena(q_L, 'left'), 'R': puntos_cadena(q_R, 'right')}
        geo = estado_geometrico(P['L'], P['R'])                  # Section III (Algorithm 1, I)
        t_col = (time.perf_counter() - t0) * 1e3

        salida, info = {}, {'collision_ms': t_col, 'clear_L': geo['clear_L'],
                            'clear_R': geo['clear_R'], 'fuerzas': {}, 'P': P}
        datos = {'L': (q_L, pdes_L, vdes_L, 'left', p['rho_L'], p['eta_vort_L']),
                 'R': (q_R, pdes_R, vdes_R, 'right', p['rho_R'], p['eta_vort_R'])}
        for b, (q, pdes, vdes, lado, rho, eta_vort) in datos.items():
            amenazas = geo['amenazas_' + b]
            JC = jacobianos_cadena(q, lado)
            J = JC[10]                                           # end-effector Jacobian
            e = pdes - P[b][10]                                  # tracking error

            W = matriz_W(amenazas, q, rho, p)                    # Algorithm 1, II
            J_pinv = pinv_dls(J, p['lam'])
            qdot0, fuerzas = campo_secundario(amenazas, JC, J_pinv, vdes,
                                              eta_vort, p, self.memoria[b])   # Algorithm 1, III

            # Eq. (24): first-order filter applied to the secondary command BEFORE the
            # current null-space operator, so the projected term always uses N_W(t).
            self.q0f[b] = p['alpha'] * qdot0 + (1.0 - p['alpha']) * self.q0f[b]
            J_W = pinv_dls(J, p['lam'], 1.0 / W)                 # Eq. (6)
            N_W = np.eye(6) - J_W @ J                            # Eq. (7)
            qc = J_W @ (vdes + p['Ke'] * e) + N_W @ self.q0f[b]  # Eq. (25)
            qs, s = escalar_norma_infinito(qc, p['qdot_max'])    # Algorithm 1, lines 28-29
            salida[b] = qs
            info['scale_' + b] = s
            info['fuerzas'][b] = fuerzas
            info['W_' + b] = W
        return salida['L'], salida['R'], info