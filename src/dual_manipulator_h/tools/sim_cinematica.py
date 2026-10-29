#!/usr/bin/env python3
"""
Offline KINEMATIC simulation of both controllers (no ROS, no Gazebo).

Purpose: fast sanity check of the controller code before the Gazebo benchmark.
The joints follow the commanded velocity exactly (q += qdot * dt, clipped to the
URDF limits). "Collision" here means overlap of the capsule model (zero clearance),
which is only a surrogate of the Gazebo mesh contacts used in the paper.

Usage (from the repository root):
    python3 src/dual_manipulator_h/tools/sim_cinematica.py --alg prop  --out sim_prop.csv
    python3 src/dual_manipulator_h/tools/sim_cinematica.py --alg lei   --out sim_lei.csv
Options: --variant no_vortex, --yaml path, --trials 0-9
"""
import os
import sys
import time
import argparse
import importlib.util
import numpy as np
import pandas as pd
import yaml

RAIZ = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..'))
DIR_PROP = os.path.join(RAIZ, 'dual_manipulator_h', 'scripts')
DIR_LEI = os.path.join(RAIZ, 'dual_manipulator_h_spheres', 'scripts')


def cargar_modulo(nombre, ruta):
    spec = importlib.util.spec_from_file_location(nombre, ruta)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--alg', choices=['prop', 'lei'], required=True)
    ap.add_argument('--variant', default='full')
    ap.add_argument('--yaml', default=os.path.join(RAIZ, 'dual_manipulator_benchmark', 'config',
                                                   'trayectorias_50.yaml'))
    ap.add_argument('--trials', default=None, help='e.g. 0-9')
    ap.add_argument('--out', default=None)
    ap.add_argument('--set', action='append', default=[], help='parameter override, e.g. d_safe=0.08')
    a = ap.parse_args()
    extra = {k: float(v) for k, v in (x.split('=') for x in a.set)}

    sys.path.insert(0, DIR_PROP if a.alg == 'prop' else DIR_LEI)
    from benchmark_common import (DT, TOL_BAJA, Q_MIN, Q_MAX, evaluar_bezier, MetricasBrazo)
    from cinematica_puntos import puntos_cadena
    if a.alg == 'prop':
        from selfCollisionStrategy import ControladorPropuesto, VARIANTES
        ctrl = ControladorPropuesto({**VARIANTES[a.variant], **extra})
        etiqueta = 'LUMELSKY' if a.variant == 'full' else f'LUMELSKY_{a.variant}'
    else:
        from leiStrategy import ControladorLei
        ctrl = ControladorLei(extra)
        etiqueta = 'LEI'
    capsulas = cargar_modulo('capsulas_gt', os.path.join(DIR_PROP, 'collision.py'))

    with open(a.yaml) as f:
        tray = yaml.safe_load(f)['trayectorias']
    ids = range(len(tray))
    if a.trials:
        i0, i1 = map(int, a.trials.split('-'))
        ids = range(i0, i1 + 1)

    filas = []
    for i in ids:
        tr = tray[i]
        T = float(tr['t_total'])
        PL = np.array([tr['left_arm'][k] for k in ('p0', 'p1', 'p2', 'p3')], float)
        PR = np.array([tr['right_arm'][k] for k in ('p0', 'p1', 'p2', 'p3')], float)
        qL, qR = np.zeros(6), np.zeros(6)
        ctrl.reiniciar()
        mL, mR = MetricasBrazo(T), MetricasBrazo(T)
        colL = colR = False
        tiempos = []
        t = 0.0
        while True:
            xL = puntos_cadena(qL, 'left')[10]; xR = puntos_cadena(qR, 'right')[10]
            eL, eR = np.linalg.norm(PL[3] - xL), np.linalg.norm(PR[3] - xR)
            if (eL < TOL_BAJA and eR < TOL_BAJA) or t >= 2 * T:
                break
            pdL, vdL = evaluar_bezier(t, T, PL)
            pdR, vdR = evaluar_bezier(t, T, PR)
            t0 = time.perf_counter()
            uL, uR, info = ctrl.calcular(qL.copy(), qR.copy(), pdL, vdL, pdR, vdR)
            tiempos.append((time.perf_counter() - t0) * 1e3)
            g = capsulas.estado_geometrico(puntos_cadena(qL, 'left'), puntos_cadena(qR, 'right'))
            colL |= g['clear_L'] <= 0.0
            colR |= g['clear_R'] <= 0.0
            mL.paso(t, pdL - xL, eL, uL, uL, g['clear_L'])
            mR.paso(t, pdR - xR, eR, uR, uR, g['clear_R'])
            qL = np.clip(qL + uL * DT, Q_MIN, Q_MAX)
            qR = np.clip(qR + uR * DT, Q_MIN, Q_MAX)
            t += DT
        for arm, m, e, col in (('LEFT', mL, eL, colL), ('RIGHT', mR, eR, colR)):
            d = {'trial_id': i, 'algorithm': etiqueta, 'arm': arm,
                 'result': 'COLLISION' if col else ('SUCCESS' if e < TOL_BAJA else 'TIMEOUT')}
            d.update(m.resumen(t, e))
            d.update({'pipeline_mean_ms': np.mean(tiempos), 'pipeline_max_ms': np.max(tiempos)})
            filas.append(d)
        print(f"trial {i:2d}: L {filas[-2]['result']:9s} eL={eL:.3f} | R {filas[-1]['result']:9s} "
              f"eR={eR:.3f} | t={t/T:.2f}T | min clr L={mL.min_clear:.3f} R={mR.min_clear:.3f} "
              f"| {np.mean(tiempos):.2f} ms")
    df = pd.DataFrame(filas)
    if a.out:
        df.to_csv(a.out, index=False)
    print(df.groupby(['arm', 'result']).size())


if __name__ == '__main__':
    main()
