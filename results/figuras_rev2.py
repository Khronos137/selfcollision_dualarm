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



def _rc_orig():
    return {'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Liberation Sans', 'DejaVu Sans'],
            'font.size': 11, 'axes.labelweight': 'bold', 'axes.labelsize': 11,
            'xtick.labelsize': 11, 'ytick.labelsize': 10, 'axes.spines.top': True,
            'axes.spines.right': True, 'axes.edgecolor': '#cccccc', 'axes.labelcolor': '#262626',
            'xtick.color': '#262626', 'ytick.color': '#262626', 'pdf.fonttype': 42}


def fig_tiempos(df, algs, out):
    """Timing figures in the style of the original manuscript (no in-figure title).
    collision_time: mean collision-detection time per trial (bar = mean over trials, dots = trials).
    controlTime: per-trial mean and maximum callback time (box plots), with the 10 ms deadline."""
    s = df[df.arm == 'LEFT']                      # timing columns are identical for both arm rows
    nombres = [NOMBRE_ALG.get(a, a) for a in algs]
    rng = np.random.default_rng(0)
    with plt.rc_context(_rc_orig()):
        # ---- collision detection time ----
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        colores = ['#add8e6', '#e6a23c']
        for i, a in enumerate(algs):
            v = s[s.algorithm == a].collision_mean_ms.dropna().values
            ax.bar(i, v.mean(), width=0.8, color=colores[i % 2], edgecolor='#1a237e', linewidth=2, zorder=2)
            ax.errorbar(i, v.mean(), yerr=v.std(ddof=1), color='#1a237e', capsize=6, lw=2, zorder=3)
            ax.scatter(i + rng.uniform(-0.12, 0.12, len(v)), v, s=14, color='#333333', alpha=0.6, zorder=4)
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels(nombres, fontweight='bold')
        ax.set_xlabel('Algorithm'); ax.set_ylabel('Execution Time (ms)')
        ax.set_ylim(0, 1.3 * s[s.algorithm.isin(algs)].collision_mean_ms.max())
        ax.grid(True, linestyle='--', color='#c5d1f5', zorder=0)
        ax.set_axisbelow(True)
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'collision_time.pdf'))
        fig.savefig(os.path.join(out, 'collision_time.png'), dpi=150)
        plt.close(fig)

        # ---- complete callback time: mean vs maximum per trial ----
        fig, ax = plt.subplots(figsize=(5.5, 4.0))
        estilos = {'callback_mean_ms': ('#add8e6', 'Mean Time'), 'callback_max_ms': ('#f0b04f', 'Maximum Time')}
        ymax = 0
        for i, a in enumerate(algs):
            for j, (col, (color, _)) in enumerate(estilos.items()):
                v = s[s.algorithm == a][col].dropna().values
                ymax = max(ymax, v.max())
                pos = i + (-0.2 if j == 0 else 0.2)
                ax.boxplot(v, positions=[pos], widths=0.3, patch_artist=True, showfliers=False,
                           boxprops=dict(facecolor=color, edgecolor='#1a237e', linewidth=2),
                           whiskerprops=dict(color='#1a237e', linewidth=1.5),
                           capprops=dict(color='#1a237e', linewidth=1.5),
                           medianprops=dict(color='#dc143c', linewidth=2.5), zorder=2)
                ax.scatter(pos + rng.uniform(-0.08, 0.08, len(v)), v, s=14, color='#333333', alpha=0.5, zorder=3)
        ax.axhline(10.0, color='black', linestyle=':', linewidth=2.5, zorder=1)
        ax.set_xticks(range(len(algs)))
        ax.set_xticklabels(nombres, fontweight='bold')
        ax.set_xlabel('Algorithm'); ax.set_ylabel('Execution Time (ms)')
        ax.set_ylim(0, max(14.5, 1.15 * ymax))
        ax.grid(True, linestyle='--', color='#cccccc', zorder=0)
        ax.set_axisbelow(True)
        manijas = [plt.Rectangle((0, 0), 1, 1, fc=c, ec='#1a237e', lw=2) for c, _ in estilos.values()]
        manijas.append(plt.Line2D([0], [0], color='black', linestyle=':', linewidth=2.5))
        ax.legend(manijas, [n for _, n in estilos.values()] + ['Control Deadline (10.0 ms)'],
                  title='Metric', title_fontproperties={'weight': 'bold', 'size': 11}, loc='upper right',
                  frameon=False, ncol=1, prop={'weight': 'bold', 'size': 9})
        fig.tight_layout()
        fig.savefig(os.path.join(out, 'controlTime.pdf'))
        fig.savefig(os.path.join(out, 'controlTime.png'), dpi=150)
        plt.close(fig)

def fig_metricas(df, algs, out):
    """RMSE, Control Effort and Joint Path Length in the style of the original manuscript figures
    (box plots per algorithm and arm, no in-figure title). Only the executions used in the paired
    tests are shown: for each arm, trials in which neither method registered a collision."""
    archivos = {'rmse_m': ('rmse_tracking', 'RMSE (m)'),
                'ce': ('CONTROL_EFFORT', 'Control Effort (rad/s) [Log Scale]'),
                'jpl_rad': ('JOINT_PATH_LENGTH', 'Joint Path Length (rad)')}
    brazos = {'LEFT': ('#add8e6', 'Left'), 'RIGHT': ('#e6a23c', 'Right')}
    validos = {}
    for arm in brazos:
        s = df[(df.arm == arm) & df.algorithm.isin(algs)]
        col = s[s.outcome == 'COLLISION'].trial_id.unique()
        validos[arm] = set(s.trial_id.unique()) - set(col)
    rng = np.random.default_rng(0)
    with plt.rc_context(_rc_orig()):
        for metrica, (nombre, etiqueta) in archivos.items():
            fig, ax = plt.subplots(figsize=(5.5, 4.0))
            for i, a in enumerate(algs):
                for j, (arm, (color, _)) in enumerate(brazos.items()):
                    s = df[(df.algorithm == a) & (df.arm == arm) & df.trial_id.isin(validos[arm])]
                    v = s[metrica].dropna().values
                    pos = i + (-0.2 if j == 0 else 0.2)
                    ax.boxplot(v, positions=[pos], widths=0.36, patch_artist=True, showfliers=False,
                               whis=(0, 100),
                               boxprops=dict(facecolor=color, edgecolor='#1a237e', linewidth=1.5),
                               whiskerprops=dict(color='#1a237e', linewidth=1.5),
                               capprops=dict(color='#1a237e', linewidth=1.5),
                               medianprops=dict(color='#dc143c', linewidth=2.0), zorder=2)
                    ax.scatter(pos + rng.uniform(-0.1, 0.1, len(v)), v, s=12, color='#333333',
                               alpha=0.5, zorder=3)
            ax.set_xticks(range(len(algs)))
            ax.set_xticklabels([NOMBRE_ALG.get(a, a) for a in algs], fontweight='bold')
            ax.set_xlabel('Algorithm'); ax.set_ylabel(etiqueta)
            if metrica == 'ce':
                ax.set_yscale('log')       # as in the original figure (wide range of the baseline)
            else:
                ax.set_ylim(bottom=0)
            ax.grid(True, linestyle='--', color='#cccccc', zorder=0)
            ax.set_axisbelow(True)
            manijas = [plt.Rectangle((0, 0), 1, 1, fc=c, ec='#1a237e', lw=1.5) for c, _ in brazos.values()]
            ax.legend(manijas, [n for _, n in brazos.values()], title='Manipulator',
                      title_fontproperties={'weight': 'bold', 'size': 11}, loc='best',
                      frameon=False, prop={'weight': 'bold', 'size': 10})
            fig.tight_layout()
            fig.savefig(os.path.join(out, nombre + '.pdf'))
            fig.savefig(os.path.join(out, nombre + '.png'), dpi=150)
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
    """Convergence figure (replaces the previous Fig. 9), in the style of the original manuscript
    figures and without in-figure titles; panels are identified by a letter only.
    (a)-(b) fraction of executions that reached the tolerance vs t/T (Kaplan-Meier: safe timeouts
    and collisions are censored at their final time); (c)-(d) median and interquartile band of the
    distance to the target vs t/T, from the per-cycle logs."""
    grilla = np.linspace(0, 2.0, 201)
    con_logs = logs is not None and os.path.isdir(logs)
    filas = 2 if con_logs else 1
    col = ['#1f5fa8', '#e08a1e']                    # proposed (blue), baseline (orange)
    rc = _rc_orig()
    rc.update({'axes.titlesize': 11})
    with plt.rc_context(rc):
        fig, axs = plt.subplots(filas, 2, figsize=(10.0, 3.4 * filas), sharex=True, squeeze=False)
        letras = iter('abcd')
        for j, arm in enumerate(('LEFT', 'RIGHT')):
            ax = axs[0, j]
            for k, a in enumerate(algs):
                s = df[(df.algorithm == a) & (df.arm == arm)]
                tr = pd.to_numeric(s.t_reach_005_s if abs(tol - 0.05) < 1e-9 else s.t_reach_002_s,
                                   errors='coerce')
                ev = tr.notna() & (s.result != 'COLLISION')
                t = np.where(ev, tr / s.t_total_s, s.time_factor)
                ax.step(grilla, km(t, ev, grilla), where='post', color=col[k % 2], linewidth=2.2,
                        label=NOMBRE_ALG.get(a, a))
            ax.axvline(1.0, color='black', linewidth=1.5, linestyle=':')
            ax.set_ylim(0, 1.05)
            ax.set_ylabel('Fraction Reached' if j == 0 else '')
            ax.grid(True, linestyle='--', color='#cccccc', zorder=0)
            ax.text(0.02, 0.95, f'({next(letras)}) {arm.capitalize()} arm', transform=ax.transAxes,
                    ha='left', va='top', fontweight='bold', fontsize=11)
        fila_c = []
        if con_logs:
            for j, arm in enumerate(('LEFT', 'RIGHT')):
                ax2 = axs[1, j]
                for k, a in enumerate(algs):
                    h = historias(logs, a, arm, grilla)
                    if len(h) == 0:
                        continue
                    q1, q2, q3 = np.percentile(h, [25, 50, 75], axis=0)
                    ax2.fill_between(grilla, q1, q3, color=col[k % 2], alpha=0.20, linewidth=0)
                    ax2.plot(grilla, q2, color=col[k % 2], linewidth=2.2, label=NOMBRE_ALG.get(a, a))
                ax2.axhline(tol, color='black', linewidth=1.5, linestyle='--')
                ax2.axvline(1.0, color='black', linewidth=1.5, linestyle=':')
                ax2.set_ylabel('Distance to Target (m)' if j == 0 else '')
                ax2.grid(True, linestyle='--', color='#cccccc', zorder=0)
                ax2.text(0.98, 0.95, f'({next(letras)}) {arm.capitalize()} arm', transform=ax2.transAxes,
                         ha='right', va='top', fontweight='bold', fontsize=11)
                fila_c.append(ax2)
            y_max = max(a.get_ylim()[1] for a in fila_c)
            for a in fila_c:
                a.set_ylim(0, y_max)
        for ax in axs[-1]:
            ax.set_xlabel('Normalized Time t/T')
        manijas = [plt.Line2D([0], [0], color=col[k % 2], linewidth=2.2) for k in range(len(algs))]
        nombres = [NOMBRE_ALG.get(a, a) for a in algs]
        manijas += [plt.Line2D([0], [0], color='black', linestyle=':', linewidth=1.5)]
        nombres += ['Nominal Time (t = T)']
        if con_logs:
            manijas += [plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5)]
            nombres += [f'Tolerance ({tol:.2f} m)']
        fig.legend(manijas, nombres, loc='upper center', ncol=len(nombres), frameon=False,
                   prop={'weight': 'bold', 'size': 11}, bbox_to_anchor=(0.5, 1.0))
        fig.tight_layout(rect=(0, 0, 1, 0.94 if con_logs else 0.88))
        fig.savefig(os.path.join(out, 'fig_convergence.pdf'))
        fig.savefig(os.path.join(out, 'fig_convergence.png'), dpi=150)
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
    if 'callback_mean_ms' in df.columns:
        fig_tiempos(df, algs, a.out)
    fig_convergence(df, algs, a.logs, a.tol, a.out)
    fig_metricas(df, algs, a.out)
    print('Saved:', sorted(os.listdir(a.out)))


if __name__ == '__main__':
    main()