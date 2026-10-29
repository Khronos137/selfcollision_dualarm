#!/usr/bin/env python3
"""
Diagnostic of the per-cycle logs (.npz) written by the controllers.

For every trial it reports:
  * initial configuration: max |q| of each arm at the first cycle (homing accuracy)
  * initial end-effector error with respect to p0
  * sign agreement between commanded and executed joint velocity, per joint
    (correlation between qdot_cmd(k) and the finite difference of q, k -> k+1)
  * real-time factor estimated from the joint_states stamps (simulation time)
    and the measured controller periods (wall time)

Usage:  python3 revisar_logs.py <log_dir>      (e.g. ~/Maestrias/resultados_rev2/fase1/logs)
"""
import os
import sys
import glob
import numpy as np


def main(raiz):
    archivos = sorted(glob.glob(os.path.join(os.path.expanduser(raiz), '*', 'trial_*.npz')))
    if not archivos:
        print('No .npz files found under', raiz)
        return
    for f in archivos:
        d = np.load(f)
        alg = os.path.basename(os.path.dirname(f))
        print(f'\n=== {alg} / {os.path.basename(f)}  ({len(d["t"])} cycles)')
        for b in ('L', 'R'):
            q = d[f'q_{b}']
            cmd = d[f'qdot_cmd_{b}']
            ejec = np.diff(q, axis=0) / 0.01           # executed velocity between cycles
            corr = []
            for j in range(6):
                a, c = cmd[:-1, j], ejec[:, j]
                if np.std(a) < 1e-6 or np.std(c) < 1e-6:
                    corr.append(np.nan)
                else:
                    corr.append(np.corrcoef(a, c)[0, 1])
            e0 = np.linalg.norm(d[f'x_{b}'][0] - d[f'xdes_{b}'][0])
            print(f'  arm {b}: start max|q| = {np.max(np.abs(q[0])):.3f} rad | '
                  f'start EE error = {e0:.3f} m | corr(cmd, executed) per joint = '
                  + ' '.join('  nan' if np.isnan(c) else f'{c:+.2f}' for c in corr))
        stamps = d['js_stamp']
        per = d['period_ms'][1:]
        per = per[np.isfinite(per)]
        if len(per) and np.all(np.isfinite(stamps)):
            rtf = (stamps[-1] - stamps[0]) / (np.sum(per) / 1000.0)
            print(f'  real-time factor ~ {rtf:.3f} | callback max {np.nanmax(d["callback_ms"]):.2f} ms')


if __name__ == '__main__':
    main(sys.argv[1] if len(sys.argv) > 1 else '~/benchmark_logs')