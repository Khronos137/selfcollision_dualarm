"""
Capsule-based self-collision detector of the proposed controller (Section III).

* Each arm is covered by K = 10 capsules (segment k joins chain points P[k-1], P[k]).
* The torso is capsule index 0.
* Global index: 0 = torso, 1..K = left arm, K+1..2K = right arm (Section III.D).
* M[i, j] : distance between capsule AXES (Eq. 2)  -> computed with Lumelsky's
            segment-segment algorithm.
* C[i, j] : SURFACE CLEARANCE  C = max(0, M - r_i - r_j). Every threshold of the
            controller (d_safe, d_crit, Eq. 3, Eq. 12, Eq. 18) is applied to C.

Structural pruning (Section III.E), applied per arm with LOCAL indices k = 1..K:
  1) Contiguous-link exclusion: capsules of the same arm are only compared when
     they belong to rigid bodies that are not connected by a single joint
     (|body(k) - body(l)| >= 2). Bodies follow the influence map PHI.
  2) Base-invariant exclusion: capsules k <= K_BASE are embedded in the shoulder;
     they are not compared with the torso, and two such capsules of different
     arms are not compared with each other.
  3) Kinematic-impossibility filter: pairs that cannot collide anywhere in the
     admissible joint range (EXCLUIDOS_*, verified by tools/verificar_poda.py).
"""
import numpy as np

K = 10
K_BASE = 3

# Capsule radius of segment k (physical envelope of the Manipulator-H links) [m]
RADIOS = {1: 0.035, 2: 0.075, 3: 0.075, 4: 0.050, 5: 0.065,
          6: 0.065, 7: 0.045, 8: 0.055, 9: 0.055, 10: 0.030}

# Torso: URDF cylinder of radius 0.10 m and height 0.80 m (z in [0, 0.8]),
# enclosed by a capsule whose axis spans z in [0.10, 0.80].
TORSO_BASE = np.array([0.0, 0.0, 0.10])
TORSO_TOP = np.array([0.0, 0.0, 0.80])
RADIO_TORSO = 0.10

# Kinematic influence map PHI(k): index of the most distal joint that moves capsule k
PHI = {1: 0, 2: 0, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 5, 10: 5}
# Rigid bodies of the chain (consecutive values of PHI)
_niveles = sorted(set(PHI.values()))
CUERPO = {k: _niveles.index(PHI[k]) for k in PHI}


# Kinematic-impossibility filter: pairs whose minimum clearance over the whole
# admissible joint range is positive (verified with tools/verificar_poda.py,
# 1e5 samples per arm, margin 0.02 m). They can never collide.
EXCLUIDOS_INTRA = {(5, 1), (5, 2), (6, 1), (6, 2)}     # (distal, proximal)
EXCLUIDOS_TORSO = {4, 5}


def indice_global(brazo, k):
    return k if brazo == 'L' else K + k


# =====================================================================
# 1. SEGMENT-SEGMENT CLOSEST POINTS (Lumelsky), Section III.B-C
# =====================================================================
def segmento_segmento(p1, p2, p3, p4, eps=1e-12):
    """Closest points between segments [p1,p2] and [p3,p4].
    Returns (distance, s, t, c1, c2) with c1 = p1 + s(p2-p1), c2 = p3 + t(p4-p3).
    Near-parallel segments: the midpoint of the overlapping parametric interval
    is used (no arbitrary zero-anchoring), Section III.C."""
    u = p2 - p1
    v = p4 - p3
    w = p1 - p3
    a = u @ u; b = u @ v; c = v @ v; d = u @ w; e = v @ w
    D = a * c - b * b

    if a < eps and c < eps:
        s, t = 0.0, 0.0
    elif a < eps:
        s, t = 0.0, min(max(e / c, 0.0), 1.0)
    elif c < eps:
        s, t = min(max(-d / a, 0.0), 1.0), 0.0
    elif D <= 1e-9 * a * c:
        # (near) parallel: project the second segment on the first one
        s3 = ((p3 - p1) @ u) / a
        s4 = ((p4 - p1) @ u) / a
        lo, hi = max(0.0, min(s3, s4)), min(1.0, max(s3, s4))
        if lo <= hi:
            s = 0.5 * (lo + hi)                      # midpoint of the overlap
        else:
            s = 0.0 if max(s3, s4) < 0.0 else 1.0    # disjoint: nearest endpoint
        t = min(max(((p1 + s * u - p3) @ v) / c, 0.0), 1.0)
    else:
        s = min(max((b * e - c * d) / D, 0.0), 1.0)
        t = (b * s + e) / c
        if t < 0.0:
            t = 0.0
            s = min(max(-d / a, 0.0), 1.0)
        elif t > 1.0:
            t = 1.0
            s = min(max((b - d) / a, 0.0), 1.0)
    c1 = p1 + s * u
    c2 = p3 + t * v
    return float(np.linalg.norm(c1 - c2)), s, t, c1, c2


# =====================================================================
# 2. VALID PAIRS (pruned topological matrix), Section III.E
# =====================================================================
def _construir_pares():
    pares = []   # (tipo, (brazo_a, k_a), (brazo_b, k_b))
    for brazo in ('L', 'R'):
        for k in range(K_BASE + 1, K + 1):
            if k not in EXCLUIDOS_TORSO:
                pares.append(('torso', (brazo, k), ('T', 0)))
    for k in range(1, K + 1):
        for l in range(1, K + 1):
            if k <= K_BASE and l <= K_BASE:
                continue
            pares.append(('cross', ('L', k), ('R', l)))
    for brazo in ('L', 'R'):
        for k in range(1, K + 1):
            for l in range(k + 1, K + 1):
                if abs(CUERPO[k] - CUERPO[l]) >= 2 and (l, k) not in EXCLUIDOS_INTRA:
                    pares.append(('intra', (brazo, l), (brazo, k)))   # (distal, proximal)
    return pares


PARES = _construir_pares()


# =====================================================================
# 3. GEOMETRIC STATE OF THE ROBOT
# =====================================================================
def estado_geometrico(P_L, P_R):
    """P_L, P_R: (11, 3) chain points.
    Returns dict with:
      M, C          : (2K+1, 2K+1) axis distances and surface clearances (inf if pruned)
      amenazas_L/R  : dict k -> closest threat of capsule k
                      {'d', 'p', 'p_obs', 's', 'tipo', 'obs': (brazo, k_obs, t_obs), 'dir_obs'}
      clear_L/R     : minimum clearance of each arm (metric)
    For intra-arm pairs the threat is assigned to the distal capsule only, which
    uses the relative Jacobian of both closest points (Section IV.C)."""
    n = 2 * K + 1
    M = np.full((n, n), np.inf)
    C = np.full((n, n), np.inf)
    P = {'L': P_L, 'R': P_R}
    amenazas = {'L': {}, 'R': {}}

    def registrar(brazo, k, d, s, p, p_obs, tipo, obs, dir_obs):
        actual = amenazas[brazo].get(k)
        if actual is None or d < actual['d']:
            amenazas[brazo][k] = {'d': d, 's': s, 'p': p, 'p_obs': p_obs, 'tipo': tipo,
                                  'obs': obs, 'dir_obs': dir_obs}

    for tipo, (ba, ka), (bb, kb) in PARES:
        a1, a2 = P[ba][ka - 1], P[ba][ka]
        if tipo == 'torso':
            b1, b2, rb = TORSO_BASE, TORSO_TOP, RADIO_TORSO
        else:
            b1, b2, rb = P[bb][kb - 1], P[bb][kb], RADIOS[kb]
        dist, s, t, ca, cb = segmento_segmento(a1, a2, b1, b2)
        clr = max(0.0, dist - RADIOS[ka] - rb)
        gi = indice_global(ba, ka)
        gj = 0 if tipo == 'torso' else indice_global(bb, kb)
        M[gi, gj] = M[gj, gi] = dist
        C[gi, gj] = C[gj, gi] = clr

        registrar(ba, ka, clr, s, ca, cb, tipo, (bb, kb, t), b2 - b1)
        if tipo == 'cross':
            registrar(bb, kb, clr, t, cb, ca, tipo, (ba, ka, s), a2 - a1)

    clear_L = min((a['d'] for a in amenazas['L'].values()), default=np.inf)
    clear_R = min((a['d'] for a in amenazas['R'].values()), default=np.inf)
    return {'M': M, 'C': C, 'amenazas_L': amenazas['L'], 'amenazas_R': amenazas['R'],
            'clear_L': clear_L, 'clear_R': clear_R}


# =====================================================================
# 4. COMPATIBILITY WITH THE RVIZ HELPERS
# =====================================================================
def puntos_dict(P_L, P_R):
    """Point dictionary used by libreria_rviz.generar_esqueleto_colisiones."""
    d = {f'left_j{i}': P_L[i] for i in range(11)}
    d.update({f'right_j{i}': P_R[i] for i in range(11)})
    d['torso_base'] = TORSO_BASE
    d['torso_top'] = TORSO_TOP
    return d


RADIOS_DICT = {f'j{k}': r for k, r in RADIOS.items()}
