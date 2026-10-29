#!/usr/bin/env python3
"""
Figures of the revised manuscript (revision 2).

  fig_outcomes.pdf     outcome per arm: success / safe timeout / collision (replaces Fig. 5)
  fig_convergence.pdf  (a) fraction of executions that reached the tolerance vs normalized
                       time (Kaplan-Meier, unfinished executions censored) and
                       (b) median and interquartile band of the distance to the target
                       vs normalized time, from the per-cycle logs (replaces Fig. 9)

Usage:
  python3 figuras_rev2.py principal.csv --logs logs --out figuras [--tol 0.05]
                          [--algs LUMELSKY LEI]
"""
import os
import glob
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt   # noqa: E402

# Colors: algorithm identity = categorical slots (blue, orange, aqua, violet);
# outcome = fixed status colors, always paired with labels/counts and hatching.
COL_ALG = ['#2a78d6', '#eb6834', '#1baf7a', '#6250d6']
COL_OUT = {'SUCCESS': '#0ca30c', 'SAFE_TIMEOUT': '#fab219', 'COLLISION': '#d03b3b'}
NOMBRE_OUT = {'SUCCESS': 'Success', 'SAFE_TIMEOUT': 'Safe timeout', 'COLLISION': 'Collision'}
NOMBRE_ALG = {'LUMELSKY': 'Proposed', 'LEI': 'Lei et al.',
              'LUMELSKY_no_vortex': 'No vortex', 'LUMELSKY_no_weighting': 'No weighting',
              'LUMELSKY_no_weighting_no_vortex': 'No weighting, no vortex'}
TEXTO = '#2b2b2b'
SECUNDARIO = '#6b6b6b'
GRID = '#e4e4e0'

plt.rcParams.update({
    'font.family': 'serif', 'font.size': 8, 'axes.titlesize': 8, 'axes.labelsize': 8,
    'xtick.labelsize': 7, 'ytick.labelsize': 7, 'legend.fontsize': 7,
    'axes.edgecolor': SECUNDARIO, 'axes.labelcolor': TEXTO, 'xtick.color': SECUNDARIO,
    'ytick.color': SECUNDARIO, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42,
})


def clasificar(df, tol):
    col = df.result.eq('COLLISION')
    df['outcome'] = np.where(col, 'COLLISION', np.where(df.final_err_m < tol, 'SUCCESS', 'SAFE_TIMEOUT'))
    return df


# Style of the original manuscript figure (Fig. 5 of the first revision)
ESTILO_ORIG = {
    'SUCCESS': ('#38833c', 'Success (< 0.05 m)'),
    'SAFE_TIMEOUT': ('#ffb74d', 'Safe timeout'),
    'COLLISION': ('#d53939', 'Collision'),
}
BORDE_ORIG = '#24313b'


def fig_outcomes(df, algs, out, por_brazo=False):
    """Stacked vertical bars in the style of the original Fig. 5.
    Default: one bar per algorithm with the 100 arm executions (both arms), as before.
    por_brazo=True: one bar per algorithm and arm (50 executions each)."""
    rc = {'font.family': 'sans-serif',
          'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
          'font.size': 12, 'font.weight': 'bold', 'axes.labelweight': 'bold',
          'axes.labelsize': 13, 'axes.titlesize': 13, 'xtick.labelsize': 13, 'ytick.labelsize': 12,
          'mathtext.default': 'regular',
          'axes.titleweight': 'bold', 'axes.spines.top': True, 'axes.spines.right': True,
          'axes.edgecolor': '#cccccc', 'xtick.color': '#262626', 'ytick.color': '#262626',
          'axes.labelcolor': '#262626', 'pdf.fonttype': 42}
    with plt.rc_context(rc):
        grupos = ([(a, arm) for a in algs for arm in ('LEFT', 'RIGHT')] if por_brazo
                  else [(a, None) for a in algs])
        fig, ax = plt.subplots(figsize=(6.5, 4.0))
        ancho = 0.7 if por_brazo else 0.58
        fs = 9 if por_brazo else 12
        if por_brazo:
            ax.tick_params(axis='x', labelsize=11)
        x = np.arange(len(grupos))
        total_max = 0
        for i, (a, arm) in enumerate(grupos):
            s = df[(df.algorithm == a) & ((df.arm == arm) if arm else True)]
            base = 0
            for o in ('SUCCESS', 'SAFE_TIMEOUT', 'COLLISION'):
                sel = s[s.outcome == o]
                k = len(sel)
                if k == 0:
                    continue
                ax.bar(x[i], k, bottom=base, width=ancho, color=ESTILO_ORIG[o][0],
                       edgecolor=BORDE_ORIG, linewidth=1.2, zorder=3)
                if o == 'COLLISION':
                    txt = f'{k}'
                else:
                    e = sel.final_err_m
                    txt = f'{k}\n({e.mean():.3f} $\\pm$ {e.std(ddof=1) if k > 1 else 0:.3f} m)'
                if k >= 8:
                    ax.text(x[i], base + k / 2, txt, ha='center', va='center', fontsize=fs,
                            fontweight='bold', color='black', zorder=4)
                else:
                    ax.text(x[i], base + k + 0.005 * (1.25 * (100 if not por_brazo else 50)),
                            txt.split('\n')[0], ha='center', va='bottom',
                            fontsize=fs, fontweight='bold', color='black', zorder=4)
                base += k
            total_max = max(total_max, base)
        etiquetas = [NOMBRE_ALG.get(a, a) + (f'\n{arm.lower()} arm' if arm else '') for a, arm in grupos]
        ax.set_xticks(x)
        ax.set_xticklabels(etiquetas, fontweight='bold')
        ax.set_ylim(0, 1.25 * total_max)
        ax.set_ylabel('Number of Executions' if por_brazo else 'Number of Arm Executions')
        ax.set_xlabel('Algorithm')
        ax.yaxis.grid(True, linestyle='--', color='#cccccc', zorder=0)
        ax.xaxis.grid(True, linestyle='-', color='#e6e6e6', zorder=0)
        ax.set_axisbelow(True)
        manijas = [plt.Rectangle((0, 0), 1, 1, fc=ESTILO_ORIG[o][0], ec=BORDE_ORIG, lw=1.2)
                   for o in ESTILO_ORIG]
        leg = ax.legend(manijas, [ESTILO_ORIG[o][1] for o in ESTILO_ORIG], ncol=3, loc='upper center',
                        frameon=True, fancybox=True, edgecolor=BORDE_ORIG, prop={'size': 12 if not por_brazo else 11, 'weight': 'normal'})
        leg.get_frame().set_linewidth(1.0)
        fig.tight_layout()
        nombre = 'fig_outcomes_por_brazo' if por_brazo else 'fig_outcomes'
        fig.savefig(os.path.join(out, nombre + '.pdf'))
        fig.savefig(os.path.join(out, nombre + '.png'), dpi=160)
        plt.close(fig)


def km(tiempos, eventos, grilla):
    t = np.asarray(tiempos, float); e = np.asarray(eventos, bool)
    orden = np.argsort(t); t, e = t[orden], e[orden]
    s, n, curva, i = 1.0, len(t), [], 0
    for g in grilla:
        while i < len(t) and t[i] <= g:
            if e[i]:
                s *= 1 - 1.0 / (n - i)
            i += 1
        curva.append(1 - s)
    return np.array(curva)


def historias(logs, alg, arm, grilla):
    """Distance to the target vs t/T for every trial of one algorithm and arm."""
    b = 'L' if arm == 'LEFT' else 'R'
    curvas = []
    for f in sorted(glob.glob(os.path.join(logs, alg, 'trial_*.npz'))):
        d = np.load(f)
        T = float(d[f'meta_T_{b}'])
        P3 = d[f'meta_P_{b}'][3]
        e = np.linalg.norm(d[f'x_{b}'] - P3, axis=1)
        tn = d['t'] / T
        c = np.interp(grilla, tn, e, right=e[-1])     # hold the last value after the trial ends
        curvas.append(c)
    return np.array(curvas)


def fig_convergence(df, algs, logs, tol, out):
    grilla = np.linspace(0, 2.0, 201)
    con_logs = logs is not None and os.path.isdir(logs)
    filas = 2 if con_logs else 1
    fig, axs = plt.subplots(filas, 2, figsize=(7.16, 2.3 * filas), sharex=True, squeeze=False)
    for j, arm in enumerate(('LEFT', 'RIGHT')):
        ax = axs[0, j]
        for k, a in enumerate(algs):
            s = df[(df.algorithm == a) & (df.arm == arm)]
            tr = pd.to_numeric(s.t_reach_005_s if abs(tol - 0.05) < 1e-9 else s.t_reach_002_s,
                               errors='coerce')
            ev = tr.notna() & (s.result != 'COLLISION')
            t = np.where(ev, tr / s.t_total_s, s.time_factor)
            ax.step(grilla, km(t, ev, grilla), where='post', color=COL_ALG[k], linewidth=1.6,
                    label=NOMBRE_ALG.get(a, a))
        ax.axvline(1.0, color=SECUNDARIO, linewidth=0.8, linestyle=':')
        ax.set_ylim(0, 1.02)
        ax.set_title(f'({"a" if j == 0 else "b"}) {arm.lower()} arm: reached {tol:.2f} m', loc='left')
        ax.set_ylabel('Fraction of executions')
        ax.yaxis.grid(True, color=GRID, linewidth=0.6)
        if j == 0:
            ax.legend(frameon=False, loc='lower right')
        if con_logs:
            ax2 = axs[1, j]
            for k, a in enumerate(algs):
                h = historias(logs, a, arm, grilla)
                if len(h) == 0:
                    continue
                q1, q2, q3 = np.percentile(h, [25, 50, 75], axis=0)
                ax2.fill_between(grilla, q1, q3, color=COL_ALG[k], alpha=0.18, linewidth=0)
                ax2.plot(grilla, q2, color=COL_ALG[k], linewidth=1.6, label=NOMBRE_ALG.get(a, a))
            ax2.axhline(tol, color=SECUNDARIO, linewidth=0.8, linestyle='--')
            ax2.axvline(1.0, color=SECUNDARIO, linewidth=0.8, linestyle=':')
            ax2.set_ylabel('Distance to target [m]')
            ax2.set_title(f'({"c" if j == 0 else "d"}) {arm.lower()} arm: median and IQR', loc='left')
            ax2.yaxis.grid(True, color=GRID, linewidth=0.6)
    for ax in axs[-1]:
        ax.set_xlabel(r'Normalized time $t/T$')
    fig.tight_layout()
    fig.savefig(os.path.join(out, 'fig_convergence.pdf'))
    fig.savefig(os.path.join(out, 'fig_convergence.png'), dpi=200)
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--logs', default=None)
    ap.add_argument('--out', default='figuras')
    ap.add_argument('--tol', type=float, default=0.05)
    ap.add_argument('--algs', nargs='+', default=['LUMELSKY', 'LEI'])
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    df = pd.read_csv(a.csv)
    df = df[~df.duplicated(['algorithm', 'trial_id', 'arm'], keep='last')]
    df = clasificar(df, a.tol)
    algs = [x for x in a.algs if x in set(df.algorithm)]
    fig_outcomes(df, algs, a.out)
    fig_outcomes(df, algs, a.out, por_brazo=True)
    fig_convergence(df, algs, a.logs, a.tol, a.out)
    print('Saved:', sorted(os.listdir(a.out)))


if __name__ == '__main__':
    main()