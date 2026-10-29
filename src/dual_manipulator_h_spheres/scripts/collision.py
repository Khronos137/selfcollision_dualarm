"""
Discrete-sphere collision model of the BASELINE (Lei et al., Appl. Sci. 2020, Sec. 2).

* N_ESFERAS control points (sphere centres) are placed on every segment k of the
  same 11-point chain used by the proposed controller.
* Matched geometry: the PHYSICAL radius of every sphere equals the capsule radius
  of the proposed controller for that segment (RADIOS_FISICOS).
* Following Lei et al., the bounding spheres are INFLATED so that
      r_cy / r_cm = 0.4      (Sec. 3.1 of Lei et al.)
  i.e. physical contact between two links corresponds to beta = 0.4, and the
  avoidance is activated at beta <= 1 (centre distance = 2.5 x physical contact).
* Sensitivity index (Lei Eq. 4):  beta_ab = d_ab / (r_a + r_b),
  with d_ab the distance between sphere CENTRES and r the inflated radii.
* Lei et al. evaluate arm-arm pairs only (matrix beta_lr, Eq. 6). Pairs of
  shoulder-embedded segments (k <= K_BASE on both arms) are skipped, exactly as
  in the proposed detector.
"""
import numpy as np

K = 10
K_BASE = 3
N_ESFERAS = 3                    # control points per segment
RAZON_LEI = 0.4                  # r_cy / r_cm

# Physical radius per segment (identical to the proposed capsule model) [m]
RADIOS_FISICOS = {1: 0.035, 2: 0.075, 3: 0.075, 4: 0.050, 5: 0.065,
                  6: 0.065, 7: 0.045, 8: 0.055, 9: 0.055, 10: 0.030}
RADIOS_INFLADOS = {k: r / RAZON_LEI for k, r in RADIOS_FISICOS.items()}

# Influence map: most distal joint that moves segment k (same as proposed)
PHI = {1: 0, 2: 0, 3: 2, 4: 2, 5: 3, 6: 3, 7: 3, 8: 3, 9: 5, 10: 5}

_T_ESFERAS = [(m + 0.5) / N_ESFERAS for m in range(N_ESFERAS)]
PARES_SEG = [(k, l) for k in range(1, K + 1) for l in range(1, K + 1)
             if not (k <= K_BASE and l <= K_BASE)]


def esferas(P):
    """Returns list of (k, t, centre) for one arm (P: chain points (11, 3))."""
    out = []
    for k in range(1, K + 1):
        for t in _T_ESFERAS:
            out.append((k, t, P[k - 1] + t * (P[k] - P[k - 1])))
    return out


def beta_minimo(P_L, P_R):
    """Minimal sensitivity index beta_lrmin (Lei Eqs. 4-7) and the pair that attains it.
    Also returns the minimum PHYSICAL clearance between spheres (for logging)."""
    eL = {}
    for k, t, c in esferas(P_L):
        eL.setdefault(k, []).append((t, c))
    eR = {}
    for k, t, c in esferas(P_R):
        eR.setdefault(k, []).append((t, c))

    mejor = None
    clear_fis = np.inf
    for k, l in PARES_SEG:
        Rs = RADIOS_INFLADOS[k] + RADIOS_INFLADOS[l]
        rs = RADIOS_FISICOS[k] + RADIOS_FISICOS[l]
        for tL, cL in eL[k]:
            for tR, cR in eR[l]:
                d = float(np.linalg.norm(cL - cR))           # Lei Eq. (3)
                beta = d / Rs                                 # Lei Eq. (4)
                if mejor is None or beta < mejor['beta']:
                    mejor = {'beta': beta, 'kL': k, 'tL': tL, 'cL': cL,
                             'kR': l, 'tR': tR, 'cR': cR}
                if d - rs < clear_fis:
                    clear_fis = d - rs
    mejor['clear_fisico'] = max(0.0, clear_fis)
    return mejor
