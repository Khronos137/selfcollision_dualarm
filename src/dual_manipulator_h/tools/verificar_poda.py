#!/usr/bin/env python3

import os
import sys
import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'scripts'))
from cinematica_puntos import puntos_cadena                       # noqa: E402
from collision import (segmento_segmento, RADIOS, RADIO_TORSO, TORSO_BASE, TORSO_TOP,
                       K, K_BASE, CUERPO, PHI)                     # noqa: E402

MARGEN = 0.02   # [m] pairs whose sampled minimum clearance exceeds this are excluded


def main(n):
    rng = np.random.default_rng(12345)
    intra = [(l, k) for k in range(1, K + 1) for l in range(k + 1, K + 1)
             if abs(CUERPO[k] - CUERPO[l]) >= 2]
    torso = list(range(K_BASE + 1, K + 1))
    base = list(range(1, K_BASE + 1))
    mi = {p: np.inf for p in intra}
    mt = {k: np.inf for k in torso}
    mbt = {k: np.inf for k in base}                       # base capsule vs torso
    mbc = {(k, l): np.inf for k in base for l in base}    # base capsule L vs base capsule R
    for i in range(n):
        q = rng.uniform(-np.pi / 2, np.pi / 2, 6)
        q2 = rng.uniform(-np.pi / 2, np.pi / 2, 6)
        if i % 10 == 0:                      # also sample the corners of the box
            q = rng.choice([-np.pi / 2, np.pi / 2], 6)
            q2 = rng.choice([-np.pi / 2, np.pi / 2], 6)
        PL, PR = puntos_cadena(q, 'left'), puntos_cadena(q2, 'right')
        for (k, l) in mbc:
            d = segmento_segmento(PL[k - 1], PL[k], PR[l - 1], PR[l])[0] - RADIOS[k] - RADIOS[l]
            mbc[(k, l)] = min(mbc[(k, l)], d)
        for lado in ('left', 'right'):
            P = puntos_cadena(q, lado)
            for k in base:
                d = segmento_segmento(P[k - 1], P[k], TORSO_BASE, TORSO_TOP)[0] - RADIOS[k] - RADIO_TORSO
                mbt[k] = min(mbt[k], d)
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
    print(f'base-invariant rule (K_BASE = {K_BASE}, same local indices for both arms):')
    for k, v in sorted(mbt.items()):
        estado = 'static (attached to the torso)' if PHI[k] == 0 else ('OK' if v > MARGEN else 'CHECK')
        print(f'  capsule {k} vs torso: {v:+.3f}  {estado}')
    for (k, l), v in sorted(mbc.items()):
        estado = 'static (both attached to the torso)' if PHI[k] == 0 and PHI[l] == 0 else \
            ('OK' if v > MARGEN else 'CHECK')
        print(f'  left {k} vs right {l}: {v:+.3f}  {estado}')


if __name__ == '__main__':
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 100000)