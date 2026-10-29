#!/usr/bin/env python3
"""
Explicit tests of the degenerate cases of the secondary field (R1-C12, R2-C6).

Each test builds a threat by hand and evaluates the secondary command of the proposed
controller (selfCollisionStrategy.campo_secundario):

  1. Coincident closest points between two arms (d = 0, p = p_obs), no previous direction.
  2. Coincident closest points with a previous valid escape direction in memory.
  3. Coincident closest point with the torso axis.
  4. Desired velocity equal to zero (no tangential direction), no previous vortex direction.
  5. Desired velocity equal to zero with a previous vortex direction in memory (the
     vortex keeps contouring in its last direction instead of stopping).

For every case it checks that the escape direction is a finite unit vector, that the
secondary joint command is finite and nonzero, and prints the fallback that was used.

Usage (from the repository root):  python3 src/dual_manipulator_h/tools/prueba_degenerados.py
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from cinematica_puntos import puntos_cadena, jacobianos_cadena          # noqa: E402
from benchmark_common import pinv_dls                                   # noqa: E402
import selfCollisionStrategy as S                                       # noqa: E402

P = dict(S.PARAMETROS_POR_DEFECTO)
q = np.array([0.3, 0.4, 0.5, 0.2, 0.3, 0.0])
PL = puntos_cadena(q, 'left')
JC = jacobianos_cadena(q, 'left')
J_pinv = pinv_dls(JC[10], P['lam'])
k = 9                                     # wrist capsule
p_k = PL[k]


def amenaza(tipo, p_obs, dir_obs, d=0.0, obs=('R', 9, 0.5)):
    return {k: {'d': d, 's': 1.0, 'p': p_k.copy(), 'p_obs': p_obs, 'tipo': tipo,
                'obs': obs, 'dir_obs': dir_obs}}


def evaluar(nombre, amenazas, vdes, eta_vort, memoria):
    qdot0, fuerzas = S.campo_secundario(amenazas, JC, J_pinv, vdes, eta_vort, P, memoria)
    F = fuerzas[k][1]
    u = memoria[k]['u']
    ok = (np.all(np.isfinite(qdot0)) and np.linalg.norm(qdot0) > 1e-9
          and abs(np.linalg.norm(u) - 1) < 1e-9 and np.all(np.isfinite(F)))
    print(f'{"PASS" if ok else "FAIL"}  {nombre}')
    print(f'      escape direction u = {np.round(u, 3)} | |F| = {np.linalg.norm(F):.3f} m/s | '
          f'|qdot0| = {np.linalg.norm(qdot0):.3f} rad/s')
    return ok


res = []
eje = np.array([0.0, 1.0, 0.0])

# 1. coincident points, arm-arm, empty memory -> direction orthogonal to the obstacle axis
res.append(evaluar('1. coincident points (arm-arm), no memory',
                   amenaza('cross', p_k.copy(), eje), np.array([0.1, 0, 0]), 0.0, {}))

# 2. coincident points with a previous valid direction -> keeps it
mem = {k: {'u': np.array([0.0, 0.0, 1.0]), 'u_tang': None}}
res.append(evaluar('2. coincident points (arm-arm), previous direction in memory',
                   amenaza('cross', p_k.copy(), eje), np.array([0.1, 0, 0]), 0.0, mem))
assert np.allclose(mem[k]['u'], [0, 0, 1])

# 3. coincident with the torso axis -> radial direction from the torso axis
p_axis = np.array([0.0, 0.0, p_k[2]])
res.append(evaluar('3. coincident point with the torso (radial fallback)',
                   amenaza('torso', p_k.copy(), np.array([0, 0, 1.0])), np.array([0.1, 0, 0]), 0.0, {}))

# 4. zero desired velocity, active vortex, no memory -> repulsion only (nonzero)
p_obs = p_k + np.array([0.0, 0.03, 0.0])
res.append(evaluar('4. zero desired velocity, no previous vortex direction',
                   amenaza('cross', p_obs, eje, d=0.01), np.zeros(3), P['eta_vort_R'], {}))

# 5. desired motion vanishes while contouring: previous vortex direction is kept
mem = {k: {'u': None, 'u_tang': np.array([1.0, 0.0, 0.0])}}
u_esc = (p_k - p_obs) / np.linalg.norm(p_k - p_obs)
res.append(evaluar('5. zero desired velocity, previous vortex direction in memory',
                   amenaza('cross', p_obs, eje, d=0.01), np.zeros(3), P['eta_vort_R'], mem))
ut = mem[k]['u_tang']
print(f'      vortex direction kept: {np.round(ut, 3)}, orthogonal to u: {abs(ut @ u_esc) < 1e-9}')

print(f'\n{sum(res)}/{len(res)} tests passed')
sys.exit(0 if all(res) else 1)