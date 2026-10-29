#!/usr/bin/env python3
"""
Per-cycle analysis of the benchmark logs (.npz), revision 2.

For each algorithm folder it reports:
  * timing over ALL control cycles: pipeline and full-callback duration (mean, p99, max),
    cycles above the 10 ms deadline, timer period (mean, std, p99, max);
  * velocity scaling: fraction of cycles in which the infinity-norm scaling was active;
  * joint limits: executions in which a joint (1-5) entered the warning band or reached
    the URDF limit;
  * Cartesian speeds: maximum speed of any chain point, and maximum closing speed of the
    minimum arm-to-arm surface clearance (capsule model), overall and below d_safe;
  * tracking decomposition: geometric deviation from the reference path (distance to the
    closest point of the Bezier curve) versus tracking error w.r.t. the time-parametrized
    reference (RMSE).

Usage:
  python3 analisis_logs.py <logs_dir> [--algs LUMELSKY LEI] [--repo <path to repository>]
"""
import os
import sys
import glob
import argparse
import numpy as np


def bezier(P, s):
    s = s[:, None]
    return ((1 - s)**3 * P[0] + 3 * (1 - s)**2 * s * P[1] + 3 * (1 - s) * s**2 * P[2] + s**3 * P[3])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('logs')
    ap.add_argument('--algs', nargs='+', default=['LUMELSKY', 'LEI'])
    ap.add_argument('--repo', default=os.path.join(os.path.dirname(os.path.abspath(__file__)), '..'))
    ap.add_argument('--d-safe', type=float, default=0.06)
    ap.add_argument('--gamma', type=float, default=0.1)
    ap.add_argument('--sin-geometria', action='store_true', help='skip the (slower) capsule analysis')
    a = ap.parse_args()

    sys.path.insert(0, os.path.join(a.repo, 'src', 'dual_manipulator_h', 'scripts'))
    from cinematica_puntos import puntos_cadena
    import collision as cap

    q_lim = np.pi / 2
    banda = a.gamma * np.pi          # gamma * (q_max - q_min)
    pares_cruzados = [p for p in cap.PARES if p[0] == 'cross']

    def clear_brazos(PL, PR):
        m = np.inf
        for _, (ba, ka), (bb, kb) in pares_cruzados:
            d = cap.segmento_segmento(PL[ka - 1], PL[ka], PR[kb - 1], PR[kb])[0]
            m = min(m, d - cap.RADIOS[ka] - cap.RADIOS[kb])
        return m

    for alg in a.algs:
        archivos = sorted(glob.glob(os.path.join(a.logs, alg, 'trial_*.npz')))
        if not archivos:
            print(f'{alg}: no logs'); continue
        pipe, cb, per, age, esc = [], [], [], [], []
        lim_banda = {'L': 0, 'R': 0}; lim_tope = {'L': 0, 'R': 0}
        v_pt, v_cierre, v_cierre_cerca, clr_min = [], [], [], []
        geo = {'L': [], 'R': []}; rmse = {'L': [], 'R': []}
        for f in archivos:
            d = np.load(f)
            pipe += list(d['pipeline_ms']); cb += list(d['callback_ms'])
            per += list(d['period_ms'][1:]); age += list(d['state_age_ms'])
            esc += list(np.minimum(d['scale_L'], d['scale_R']))
            t = d['t']
            for b in ('L', 'R'):
                q = d[f'q_{b}'][:, :5]
                if np.any(np.abs(q) > q_lim - banda):
                    lim_banda[b] += 1
                if np.any(np.abs(q) > q_lim - 0.01):
                    lim_tope[b] += 1
                P = d[f'meta_P_{b}']
                curva = bezier(P, np.linspace(0, 1, 2001))
                x = d[f'x_{b}']
                dg = np.min(np.linalg.norm(x[:, None, :] - curva[None, :, :], axis=2), axis=1)
                geo[b].append(np.sqrt(np.mean(dg**2)))
                rmse[b].append(np.sqrt(np.mean(np.sum((x - d[f'xdes_{b}'])**2, axis=1))))
            if a.sin_geometria:
                continue
            paso = 2                                     # every 2nd cycle (20 ms) for speed
            idx = np.arange(0, len(t), paso)
            PLs = [puntos_cadena(d['q_L'][k], 'left') for k in idx]
            PRs = [puntos_cadena(d['q_R'][k], 'right') for k in idx]
            dt = np.diff(t[idx])
            for PS in (PLs, PRs):
                v = np.linalg.norm(np.diff(np.array(PS), axis=0), axis=2) / dt[:, None]
                v_pt.append(np.max(v))
            c = np.array([clear_brazos(pl, pr) for pl, pr in zip(PLs, PRs)])
            vc = -np.diff(c) / dt
            clr_min.append(np.min(c))
            v_cierre.append(np.max(vc))
            cerca = c[:-1] < a.d_safe
            v_cierre_cerca.append(np.max(vc[cerca]) if np.any(cerca) else np.nan)

        cb = np.array(cb); per = np.array(per); per = per[np.isfinite(per)]
        pipe = np.array(pipe); age = np.array(age); age = age[np.isfinite(age)]
        esc = np.array(esc)
        print(f'\n===== {alg}  ({len(archivos)} trials, {len(cb)} control cycles)')
        print(f'pipeline [ms]: mean {pipe.mean():.2f}  p99 {np.percentile(pipe, 99):.2f}  max {pipe.max():.2f}')
        print(f'callback [ms]: mean {cb.mean():.2f}  p99 {np.percentile(cb, 99):.2f}  max {cb.max():.2f}  '
              f'cycles > 10 ms: {int(np.sum(cb > 10))}')
        print(f'timer period [ms]: mean {per.mean():.3f}  std {per.std():.3f}  p99 {np.percentile(per, 99):.2f}  '
              f'max {per.max():.2f}  periods > 10.5 ms: {int(np.sum(per > 10.5))}')
        print(f'joint-state age at callback [ms]: p99 {np.percentile(age, 99):.2f}  max {age.max():.2f}')
        print(f'velocity scaling active (s < 1): {100 * np.mean(esc < 1):.2f}% of cycles; min s = {esc.min():.3f}')
        print(f'joint limits (joints 1-5), executions entering the warning band: L {lim_banda["L"]}  R {lim_banda["R"]}; '
              f'reaching the limit (within 0.01 rad): L {lim_tope["L"]}  R {lim_tope["R"]}')
        for b in ('L', 'R'):
            g, r = np.array(geo[b]), np.array(rmse[b])
            print(f'arm {b}: path deviation RMS median {np.median(g):.4f} m (IQR {np.percentile(g, 25):.4f}-'
                  f'{np.percentile(g, 75):.4f}) | tracking RMSE median {np.median(r):.4f} m | '
                  f'median share of RMSE due to path deviation {np.median(g / r):.2f}')
        if not a.sin_geometria:
            v_pt = np.array(v_pt); v_cierre = np.array(v_cierre); vcc = np.array(v_cierre_cerca)
            print(f'max chain-point speed [m/s]: median over trials {np.median(v_pt):.2f}, max {v_pt.max():.2f}')
            print(f'min arm-arm capsule clearance [m]: median {np.median(clr_min):.3f}, min {np.min(clr_min):.3f}')
            print(f'max closing speed [m/s]: overall max {np.nanmax(v_cierre):.2f}; below d_safe: median '
                  f'{np.nanmedian(vcc):.2f}, max {np.nanmax(vcc):.2f}  (trials with clearance < d_safe: '
                  f'{int(np.sum(np.isfinite(vcc)))})')


if __name__ == '__main__':
    main()