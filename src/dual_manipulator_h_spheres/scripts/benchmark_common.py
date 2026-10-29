"""
Pure-Python utilities shared by the proposed controller and the baseline.

SHARED MODULE: this file must be identical in
    dual_manipulator_h/scripts/ and dual_manipulator_h_spheres/scripts/

No ROS imports here, so the control laws and metrics can be unit-tested offline.
"""
import os
import numpy as np

DT = 0.01            # control period [s] (100 Hz)
DEADLINE_MS = 10.0   # real-time deadline [ms]
QDOT_MAX = 3.14      # joint velocity limit used by BOTH controllers [rad/s]

# URDF joint limits (manipulator_h_macro.xacro): all joints in [-pi/2, pi/2]
Q_MIN = np.full(6, -np.pi / 2)
Q_MAX = np.full(6, np.pi / 2)

TOL_BAJA = 0.02      # low terminal tolerance [m]
TOL_MEDIA = 0.05     # medium terminal tolerance [m] (predefined, Section VI.E)


# =====================================================================
# TRAJECTORY
# =====================================================================
def evaluar_bezier(t, T_total, P):
    """Cubic Bezier (Eq. 26) with s = t / T_total. Returns (p_des, v_des).
    For t >= T_total the reference stays at P3 with zero velocity."""
    if T_total <= 1e-3:
        return P[0].copy(), np.zeros(3)
    if t >= T_total:
        return P[3].copy(), np.zeros(3)
    s = t / T_total
    p = ((1 - s)**3 * P[0] + 3 * (1 - s)**2 * s * P[1]
         + 3 * (1 - s) * s**2 * P[2] + s**3 * P[3])
    v = (1.0 / T_total) * (3 * (1 - s)**2 * (P[1] - P[0])
                           + 6 * (1 - s) * s * (P[2] - P[1])
                           + 3 * s**2 * (P[3] - P[2]))
    return p, v


# =====================================================================
# LINEAR ALGEBRA
# =====================================================================
def pinv_dls(J, lam, W_inv_diag=None):
    """(Weighted) damped least-squares pseudoinverse, Eq. (6):
       J_W^+ = W^-1 J^T (J W^-1 J^T + lam^2 I)^-1     (W diagonal)."""
    if W_inv_diag is None:
        JWinv = J.T
    else:
        JWinv = W_inv_diag[:, None] * J.T
    A = J @ JWinv + (lam**2) * np.eye(J.shape[0])
    return JWinv @ np.linalg.inv(A)


def escalar_norma_infinito(qdot, qdot_max=QDOT_MAX):
    """Uniform scaling of the whole command (Algorithm 1, lines 28-29)."""
    m = np.max(np.abs(qdot))
    s = 1.0 if m <= qdot_max else qdot_max / m
    return s * qdot, s


# =====================================================================
# PER-ARM METRICS (Table 3)
# =====================================================================
class MetricasBrazo:
    """Accumulates the operational metrics of one arm during one trial.
       RMSE : sqrt(mean ||p_des - p||^2)
       CE   : sum_{i>=2} ||qdot(i) - qdot(i-1)||_2      (commanded joint velocity)
       JPL  : sum_i ||qdot(i)||_2 * dt                 (commanded joint velocity)
    Measured joint velocities are also accumulated (JPL_meas) for cross-checking."""

    def __init__(self, T_total):
        self.T_total = T_total
        self.sum_e2 = 0.0
        self.n = 0
        self.ce = 0.0
        self.jpl = 0.0
        self.jpl_meas = 0.0
        self.qdot_prev = None
        self.min_clear = np.inf
        self.t_reach = {TOL_BAJA: None, TOL_MEDIA: None}

    def paso(self, t, err_track, err_goal, qdot_cmd, qdot_meas, clearance):
        self.sum_e2 += float(err_track @ err_track)
        self.n += 1
        if self.qdot_prev is not None:
            self.ce += float(np.linalg.norm(qdot_cmd - self.qdot_prev))
        self.qdot_prev = qdot_cmd.copy()
        self.jpl += float(np.linalg.norm(qdot_cmd)) * DT
        self.jpl_meas += float(np.linalg.norm(qdot_meas)) * DT
        if clearance < self.min_clear:
            self.min_clear = float(clearance)
        for tol in self.t_reach:
            if self.t_reach[tol] is None and err_goal < tol:
                self.t_reach[tol] = t

    def resumen(self, t_final, err_final):
        rmse = np.sqrt(self.sum_e2 / max(1, self.n))
        return {
            'rmse_m': rmse,
            'ce': self.ce,
            'jpl_rad': self.jpl,
            'jpl_meas_rad': self.jpl_meas,
            'min_clearance_m': self.min_clear if np.isfinite(self.min_clear) else None,
            'final_err_m': float(err_final),
            'time_factor': t_final / self.T_total,
            't_total_s': self.T_total,
            't_reach_002_s': self.t_reach[TOL_BAJA],
            't_reach_005_s': self.t_reach[TOL_MEDIA],
        }


# =====================================================================
# TIMING STATISTICS (R1-C20 / R2-C4)
# =====================================================================
def estadisticas_tiempo(pipeline_ms, callback_ms, period_ms, state_age_ms, col_ms):
    def st(x):
        x = np.asarray(x, dtype=float)
        if x.size == 0:
            return None, None, None
        return float(np.mean(x)), float(np.percentile(x, 99)), float(np.max(x))
    pm, p99, pmax = st(pipeline_ms)
    cm, c99, cmax = st(callback_ms)
    per = np.asarray(period_ms[1:], dtype=float)   # first period spans the trial start
    return {
        'pipeline_mean_ms': pm, 'pipeline_p99_ms': p99, 'pipeline_max_ms': pmax,
        'collision_mean_ms': float(np.mean(col_ms)) if len(col_ms) else None,
        'callback_mean_ms': cm, 'callback_p99_ms': c99, 'callback_max_ms': cmax,
        'deadline_misses': int(np.sum(np.asarray(callback_ms) > DEADLINE_MS)),
        'n_cycles': int(len(callback_ms)),
        'period_mean_ms': float(np.mean(per)) if per.size else None,
        'period_std_ms': float(np.std(per)) if per.size else None,
        'period_max_ms': float(np.max(per)) if per.size else None,
        'state_age_max_ms': float(np.max(state_age_ms)) if len(state_age_ms) else None,
    }


# =====================================================================
# PER-CYCLE LOG (one compressed .npz per trial)
# =====================================================================
class RegistroEnsayo:
    """Stores per-cycle signals so every metric (path deviation, convergence
    histories, latency percentiles, offline mesh checks) can be recomputed."""

    CAMPOS = ['t', 'q_L', 'q_R', 'qdot_meas_L', 'qdot_meas_R', 'qdot_cmd_L', 'qdot_cmd_R',
              'x_L', 'x_R', 'xdes_L', 'xdes_R', 'clear_L', 'clear_R', 'scale_L', 'scale_R',
              'pipeline_ms', 'collision_ms', 'callback_ms', 'period_ms', 'state_age_ms',
              'js_stamp']

    def __init__(self):
        self.datos = {c: [] for c in self.CAMPOS}

    def agregar(self, **kw):
        for c in self.CAMPOS:
            self.datos[c].append(kw.get(c, np.nan))

    def guardar(self, ruta, meta):
        os.makedirs(os.path.dirname(ruta), exist_ok=True)
        arr = {c: np.asarray(v, dtype=float) for c, v in self.datos.items()}
        np.savez_compressed(ruta, **arr, **{f'meta_{k}': np.asarray(v) for k, v in meta.items()})
