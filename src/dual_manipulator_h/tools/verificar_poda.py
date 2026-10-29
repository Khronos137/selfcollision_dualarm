#!/usr/bin/env python3
"""
Verifies the kinematic-impossibility filter of Section III.E (R1-C18).

Samples the admissible joint space (URDF limits, all joints in [-pi/2, pi/2]) and
reports, for every intra-arm and arm-torso capsule pair, the minimum signed
surface clearance found. Pairs whose minimum clearance stays positive (with a
safety margin) can never collide and are excluded from the online matrix.

Usage (from the package root):  python3 tools/verificar_poda.py [n_muestras]
"""
import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from cinematica_puntos import puntos_cadena                       # noqa: E402
from collision import (segmento_segmento, RADIOS, RADIO_TORSO, TORSO_BASE, TORSO_TOP,
                       K, K_BASE, CUERPO)                          # noqa: E402

MARGEN = 0.02   # [m] pairs whose sampled minimum clearance exceeds this are excluded


def main(n):
    rng = np.random.default_rng(12345)
    intra = [(l, k) for k in range(1, K + 1) for l in range(k + 1, K + 1)
             if abs(CUERPO[k] - CUERPO[l]) >= 2]
    torso = list(range(K_BASE + 1, K + 1))
    mi = {p: np.inf for p in intra}
    mt = {k: np.inf for k in torso}
    for i in range(n):
        q = rng.uniform(-np.pi / 2, np.pi / 2, 6)
        if i % 10 == 0:                      # also sample the corners of the box
            q = rng.choice([-np.pi / 2, np.pi / 2], 6)
        for lado in ('left', 'right'):
            P = puntos_cadena(q, lado)
            for (l, k) in intra:
                d = segmento_segmento(P[l - 1], P[l], P[k - 1], P[k])[0] - RADIOS[l] - RADIOS[k]
                mi[(l, k)] = min(mi[(l, k)], d)
            for k in torso:
                d = segmento_segmento(P[k - 1], P[k], TORSO_BASE, TORSO_TOP)[0] - RADIOS[k] - RADIO_TORSO
                mt[k] = min(mt[k], d)
    print(f'{n} configurations per arm. Minimum signed clearance [m]:')
    print('intra-arm (distal, proximal):')
    for p, v in sorted(mi.items()):
        print(f'  {p}: {v:+.3f}  {"EXCLUDE" if v > MARGEN else "keep"}')
    print('arm-torso (capsule k):')
    for k, v in sorted(mt.items()):
        print(f'  {k}: {v:+.3f}  {"EXCLUDE" if v > MARGEN else "keep"}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100000)
