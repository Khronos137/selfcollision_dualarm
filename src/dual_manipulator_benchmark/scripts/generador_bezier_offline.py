#!/usr/bin/env python3
"""
Stochastic X-crossing trajectory generator (Section V), revision 2.

  * Fixed random seed (reproducible dataset).
  * p0 = exact end-effector position at the home posture q = 0.
  * Intermediate points (Eq. 30) with SCALAR interpolation weights eps_1 < eps_2.
  * Terminal feasibility by rejection sampling:
      (a) Cartesian clearance between end-effector targets (Eq. 31);
      (b) configuration-space feasibility: a collision-free inverse-kinematics
          solution of BOTH arms must exist at the targets (joint limits + full
          capsule clearance: arm-arm, arm-torso and intra-arm) (R1-C14, R2-C18).
  * Every rejected sample and its reason is counted and stored in the YAML header.

Run from the source tree:
    python3 generador_bezier_offline.py [--seed 2026] [--n 50] [--swap-zones] [--out FILE]
"""
import os
import sys
import math
import argparse
import numpy as np
import yaml

AQUI = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(AQUI, '..', '..', 'dual_manipulator_h', 'scripts'))
from cinematica_puntos import puntos_cadena, jacobianos_cadena   # noqa: E402
from collision import estado_geometrico, RADIOS                   # noqa: E402
from benchmark_common import Q_MIN, Q_MAX, pinv_dls              # noqa: E402

CENTRO_L, RADIO_L = np.array([0.20, -0.05, 0.70]), 0.05   # left arm goals (right hemisphere)
CENTRO_R, RADIO_R = np.array([0.20, 0.05, 0.70]), 0.15    # right arm goals (left hemisphere)
DELTA_MAX = 0.30            # intermediate point deviation bound [m]
T_MIN, T_MAX = 2.5, 5.0     # execution time range [s]
R_END = RADIOS[10]          # end-effector capsule radius [m]


def punto_en_esfera(rng, centro, radio):
    u, v, w = rng.random(3)
    theta = 2.0 * math.pi * u
    phi = math.acos(2.0 * v - 1.0)
    r = radio * w ** (1.0 / 3.0)
    return centro + r * np.array([math.sin(phi) * math.cos(theta),
                                  math.sin(phi) * math.sin(theta), math.cos(phi)])


def puntos_intermedios(rng, p0, p3):
    eps = np.sort(rng.random(2))                         # eps_1 < eps_2
    return [p0 + e * (p3 - p0) + rng.uniform(-DELTA_MAX, DELTA_MAX, 3) for e in eps]


def ik_soluciones(rng, objetivo, lado, n_inicios=24, max_sol=8, tol=1e-3):
    """Distinct position-IK solutions inside the joint limits (DLS with random restarts)."""
    sols = []
    for _ in range(n_inicios):
        q = rng.uniform(Q_MIN, Q_MAX)
        for _ in range(200):
            x = puntos_cadena(q, lado)[10]
            e = objetivo - x
            if np.linalg.norm(e) < tol:
                break
            J = jacobianos_cadena(q, lado)[10]
            q = np.clip(q + pinv_dls(J, 0.05) @ (0.5 * e), Q_MIN, Q_MAX)
        if np.linalg.norm(objetivo - puntos_cadena(q, lado)[10]) < tol:
            if all(np.linalg.norm(q[:5] - s[:5]) > 0.1 for s in sols):
                sols.append(q.copy())
                if len(sols) >= max_sol:
                    break
    return sols


def factible_configuracion(rng, p3L, p3R, margen=0.0):
    """True if a collision-free terminal configuration of both arms exists."""
    solL = ik_soluciones(rng, p3L, 'left')
    if not solL:
        return False, 'ik_left', None
    solR = ik_soluciones(rng, p3R, 'right')
    if not solR:
        return False, 'ik_right', None
    mejor = -np.inf
    for qL in solL:
        PL = puntos_cadena(qL, 'left')
        for qR in solR:
            g = estado_geometrico(PL, puntos_cadena(qR, 'right'))
            mejor = max(mejor, min(g['clear_L'], g['clear_R']))
    if mejor > margen:
        return True, 'ok', mejor
    return False, 'no_collision_free_ik', mejor


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--seed', type=int, default=2026)
    ap.add_argument('--n', type=int, default=50)
    ap.add_argument('--d-terminal', type=float, default=0.10, help='d_terminal_safe [m] (Eq. 31)')
    ap.add_argument('--swap-zones', action='store_true',
                    help='exchange the target spheres of the two arms (R1-C19)')
    ap.add_argument('--out', default=os.path.join(AQUI, '..', 'config', 'trayectorias_50.yaml'))
    args = ap.parse_args()

    rng = np.random.default_rng(args.seed)
    cL, rL, cR, rR = CENTRO_L, RADIO_L, CENTRO_R, RADIO_R
    if args.swap_zones:
        rL, rR = rR, rL
    p0L = puntos_cadena(np.zeros(6), 'left')[10]
    p0R = puntos_cadena(np.zeros(6), 'right')[10]

    rechazos = {'terminal_distance': 0, 'ik_left': 0, 'ik_right': 0, 'no_collision_free_ik': 0}
    trayectorias = []
    while len(trayectorias) < args.n:
        T = round(float(rng.uniform(T_MIN, T_MAX)), 2)
        p3L = punto_en_esfera(rng, cL, rL)
        p3R = punto_en_esfera(rng, cR, rR)
        p1L, p2L = puntos_intermedios(rng, p0L, p3L)
        p1R, p2R = puntos_intermedios(rng, p0R, p3R)

        if np.linalg.norm(p3L - p3R) < 2 * R_END + args.d_terminal:           # Eq. (31)
            rechazos['terminal_distance'] += 1
            continue
        ok, motivo, margen = factible_configuracion(rng, p3L, p3R)
        if not ok:
            rechazos[motivo] += 1
            continue

        r = lambda v: [round(float(x), 4) for x in v]   # noqa: E731
        trayectorias.append({
            'id': len(trayectorias),
            't_total': T,
            'terminal_clearance_m': round(float(margen), 4),
            'left_arm': {'p0': r(p0L), 'p1': r(p1L), 'p2': r(p2L), 'p3': r(p3L)},
            'right_arm': {'p0': r(p0R), 'p1': r(p1R), 'p2': r(p2R), 'p3': r(p3R)},
        })
        print(f'trajectory {len(trayectorias)}/{args.n} accepted (terminal clearance {margen:.3f} m)')

    datos = {'generator': {'seed': args.seed, 'n': args.n, 'd_terminal_safe': args.d_terminal,
                           'swap_zones': args.swap_zones, 'delta_max': DELTA_MAX,
                           'T_range': [T_MIN, T_MAX], 'rejections': rechazos},
             'trayectorias': trayectorias}
    with open(args.out, 'w') as f:
        yaml.safe_dump(datos, f, default_flow_style=None, sort_keys=False)
    print(f'Saved {args.out}\nRejected samples: {rechazos}')


if __name__ == '__main__':
    main()
