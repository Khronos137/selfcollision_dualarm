#!/usr/bin/env python3
"""
Statistical analysis of the dual-arm self-collision benchmark (revision 2).
Replaces wilcoxon.py.

  1. Outcomes per arm with a PREDEFINED terminal tolerance (default 0.05 m):
       SUCCESS (no collision, final error < tol), SAFE_TIMEOUT (no collision,
       final error >= tol), COLLISION.  Also: success within the nominal time T.
  2. Bimanual collisions: paired 2x2 table, exact McNemar test, Clopper-Pearson CIs.
  3. Arm-specific paired Wilcoxon signed-rank tests (pairs = same trial, same arm,
     no collision for either method): n, exact p, matched-pairs rank-biserial r,
     Hodges-Lehmann difference with bootstrap 95% CI.
  4. Time to reach the tolerance as a censored outcome (Kaplan-Meier at 1x and 2x T).
  5. Real-time statistics aggregated over all cycles.
  6. Optional stratification by the terminal clearance stored in the trajectory YAML.

Usage:
  python3 analisis_estadistico.py resultados_benchmark.csv \
      [--proposed LUMELSKY] [--baseline LEI] [--tol 0.05] [--yaml trayectorias_50.yaml]
"""
import argparse
import numpy as np
import pandas as pd
from scipy import stats

METRICAS = {'rmse_m': 'RMSE [m]', 'ce': 'Control Effort [rad/s]', 'jpl_rad': 'Joint Path Length [rad]'}


def cargar(csv, prop, base):
    df = pd.read_csv(csv)
    df = df[df['algorithm'].isin([prop, base])].copy()
    dup = df.duplicated(['algorithm', 'trial_id', 'arm'], keep='last')
    if dup.any():
        print(f'WARNING: {dup.sum()} repeated rows (algorithm, trial, arm); keeping the last one.')
        df = df[~dup]
    return df


def clasificar(df, tol):
    col = df['result'].eq('COLLISION')
    df['outcome'] = np.where(col, 'COLLISION', np.where(df['final_err_m'] < tol, 'SUCCESS', 'SAFE_TIMEOUT'))
    t_reach = df['t_reach_005_s'] if abs(tol - 0.05) < 1e-9 else np.nan
    df['success_within_T'] = (~col) & (pd.to_numeric(t_reach, errors='coerce') <= df['t_total_s'])
    return df


def cp(k, n):
    ci = stats.binomtest(int(k), int(n)).proportion_ci(method='exact')
    return ci.low, ci.high


def rank_biserial(d):
    d = d[d != 0]
    if len(d) == 0:
        return 0.0
    r = stats.rankdata(np.abs(d))
    return (r[d > 0].sum() - r[d < 0].sum()) / r.sum()


def hodges_lehmann(d, n_boot=5000, seed=0):
    def hl(x):
        w = (x[:, None] + x[None, :]) / 2.0
        return np.median(w[np.triu_indices(len(x))])
    rng = np.random.default_rng(seed)
    b = [hl(rng.choice(d, len(d), replace=True)) for _ in range(n_boot)]
    return hl(d), np.percentile(b, 2.5), np.percentile(b, 97.5)


def kaplan_meier_fraccion(tiempos, eventos, t_eval):
    """Fraction that reached the tolerance by t_eval (1 - KM survival)."""
    orden = np.argsort(tiempos)
    t, e = np.asarray(tiempos)[orden], np.asarray(eventos)[orden]
    s, n = 1.0, len(t)
    for i in range(len(t)):
        if t[i] > t_eval:
            break
        if e[i]:
            s *= (1 - 1.0 / (n - i))
    return 1 - s


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('csv')
    ap.add_argument('--proposed', default='LUMELSKY')
    ap.add_argument('--baseline', default='LEI')
    ap.add_argument('--tol', type=float, default=0.05)
    ap.add_argument('--yaml', default=None)
    ap.add_argument('--d-crit', type=float, default=0.02, help='critical clearance [m] (Table 2)')
    ap.add_argument('--d-safe', type=float, default=0.06, help='safe clearance [m] (Table 2)')
    a = ap.parse_args()
    P, B = a.proposed, a.baseline

    df = clasificar(cargar(a.csv, P, B), a.tol)
    trials = sorted(set(df[df.algorithm == P].trial_id) & set(df[df.algorithm == B].trial_id))
    df = df[df.trial_id.isin(trials)]
    n = len(trials)
    print(f'Paired trials: {n}   ({P} vs {B}), tolerance = {a.tol} m\n')

    print('=== 1. Outcomes per arm ===')
    print(df.pivot_table(index=['algorithm', 'arm'], columns='outcome', values='trial_id',
                         aggfunc='count', fill_value=0))
    print('\nReached the tolerance at least once within the nominal time T (first passage, per arm):')
    print(df.groupby(['algorithm', 'arm'])['success_within_T'].sum(), '\n')

    print('Terminal error of SAFE_TIMEOUT executions [m] (median, IQR, max):')
    for (alg, arm), g in df[df.outcome == 'SAFE_TIMEOUT'].groupby(['algorithm', 'arm']):
        e = g.final_err_m
        print(f'  {alg:>12s} {arm:5s}: n={len(e):2d}  median {e.median():.3f}  '
              f'IQR {e.quantile(.25):.3f}-{e.quantile(.75):.3f}  max {e.max():.3f}')
    ambos = (df.assign(s=df.outcome.eq('SUCCESS')).groupby(['algorithm', 'trial_id']).s.all()
             .groupby('algorithm').sum())
    print('Bimanual trials with BOTH arms successful:', ambos.to_dict(), '\n')

    print('=== 2. Bimanual collisions ===')
    c = df.assign(c=df.result.eq('COLLISION')).groupby(['algorithm', 'trial_id']).c.any().unstack(0)
    print(pd.crosstab(c[P], c[B], rownames=[f'{P} collides'], colnames=[f'{B} collides']))
    for alg in (B, P):
        k = int(c[alg].sum()); lo, hi = cp(k, n)
        print(f'{alg:>12s}: {k}/{n} = {100*k/n:.1f}%  (95% CI {100*lo:.1f}-{100*hi:.1f}%)')
    b_ = int((~c[P] & c[B]).sum()); c_ = int((c[P] & ~c[B]).sum())
    p_m = stats.binomtest(c_, b_ + c_, 0.5).pvalue if b_ + c_ else 1.0
    print(f'Discordant pairs b={b_}, c={c_}; exact McNemar p = {p_m:.3g}\n')

    print('=== 3. Arm-specific paired Wilcoxon (no collision for either method) ===')
    filas = []
    for arm in ('LEFT', 'RIGHT'):
        x = df[(df.algorithm == P) & (df.arm == arm)].set_index('trial_id')
        y = df[(df.algorithm == B) & (df.arm == arm)].set_index('trial_id')
        ok = (x.result != 'COLLISION') & (y.result != 'COLLISION')
        for m, nombre in METRICAS.items():
            xv, yv = x.loc[ok, m].values.astype(float), y.loc[ok, m].values.astype(float)
            if len(xv) < 5:
                continue
            d = xv - yv
            w = stats.wilcoxon(xv, yv)
            hl, lo, hi = hodges_lehmann(d)
            filas.append({'arm': arm, 'metric': nombre, 'n_pairs': len(xv),
                          f'median_{P}': np.median(xv), f'median_{B}': np.median(yv),
                          'HL_diff': hl, 'CI95_low': lo, 'CI95_high': hi,
                          'p_value': w.pvalue, 'r_rb': rank_biserial(d)})
    with pd.option_context('display.width', 220, 'display.float_format', '{:.4g}'.format):
        print(pd.DataFrame(filas).to_string(index=False), '\n')

    print('=== 4. Time to reach 0.05 m (censored at the trial end) ===')
    for alg in (P, B):
        for arm in ('LEFT', 'RIGHT'):
            s = df[(df.algorithm == alg) & (df.arm == arm)]
            tr = pd.to_numeric(s.t_reach_005_s, errors='coerce')
            ev = tr.notna() & (s.result != 'COLLISION')
            t = np.where(ev, tr / s.t_total_s, s.time_factor)
            print(f'{alg:>12s} {arm:5s}: reached by 1.0T = {kaplan_meier_fraccion(t, ev, 1.0):.2f}, '
                  f'by 2.0T = {kaplan_meier_fraccion(t, ev, 2.0):.2f}')
    print()

    print('=== 5. Real-time statistics (left-arm rows; one row per trial) ===')
    for alg in (P, B):
        s = df[(df.algorithm == alg) & (df.arm == 'LEFT')]
        print(f"{alg:>12s}: pipeline mean {s.pipeline_mean_ms.mean():.2f} ms, "
              f"max {s.pipeline_max_ms.max():.2f} ms | callback mean {s.callback_mean_ms.mean():.2f} ms, "
              f"p99(max over trials) {s.callback_p99_ms.max():.2f} ms, max {s.callback_max_ms.max():.2f} ms | "
              f"deadline misses {int(s.deadline_misses.sum())}/{int(s.n_cycles.sum())} cycles | "
              f"period std (mean) {s.period_std_ms.mean():.2f} ms, period max {s.period_max_ms.max():.2f} ms")

    if a.yaml:
        import yaml
        with open(a.yaml) as f:
            t = yaml.safe_load(f)['trayectorias']
        clr = {x['id']: x.get('terminal_clearance_m', np.nan) for x in t}
        df['terminal_clearance_m'] = df.trial_id.map(clr)
        print(f'\n=== 6. Outcomes stratified by the best collision-free terminal clearance of the task '
              f'(d_crit = {a.d_crit} m, d_safe = {a.d_safe} m) ===')
        etiquetas = [f'< {a.d_crit} (target inside critical zone)',
                     f'{a.d_crit}-{a.d_safe} (target inside warning zone)',
                     f'>= {a.d_safe} (target outside influence)']
        df['strata'] = pd.cut(df.terminal_clearance_m, [-1, a.d_crit, a.d_safe, 10], labels=etiquetas)
        n_tr = df[df.algorithm == P].drop_duplicates('trial_id').groupby('strata', observed=False).size()
        print('Trials per stratum:'); print(n_tr.to_string())
        with pd.option_context('display.width', 250, 'display.max_columns', 10):
            print(df.pivot_table(index=['algorithm', 'arm', 'strata'], columns='outcome', values='trial_id',
                                 aggfunc='count', fill_value=0, observed=False))

    print('\n=== 7. Collision events by interaction type (from the Gazebo contact pairs) ===')

    def tipo(par):
        a_, b_ = par.split('|')
        if 'base_link' in (a_, b_) or 'torso' in a_ or 'torso' in b_:
            return 'arm-torso'
        return 'intra-arm' if a_.split('_')[0] == b_.split('_')[0] else 'arm-arm'

    for alg in (P, B):
        s = df[(df.algorithm == alg) & (df.result == 'COLLISION')]
        por_ensayo = {}
        for _, fila in s.iterrows():
            pares = str(fila.get('contact_pairs', '') or '')
            for par in filter(None, pares.split(';')):
                if par == 'nan':
                    continue
                por_ensayo.setdefault(int(fila.trial_id), set()).add((tipo(par), par))
        conteo = {}
        for tr_, pares in por_ensayo.items():
            for t_ in {p[0] for p in pares}:
                conteo[t_] = conteo.get(t_, 0) + 1
        print(f'{alg:>12s}: trials with a collision = {len(por_ensayo)} | trials per type = {conteo}')
        for tr_ in sorted(por_ensayo):
            print(f'{"":14s}trial {tr_:2d}: ' + ', '.join(sorted(p[1] for p in por_ensayo[tr_])))


if __name__ == '__main__':
    main()