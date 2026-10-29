# Dual-Arm Self-Collision Avoidance: Controllers and Benchmark (ROS 2 Humble)

Source code, datasets, and analysis scripts of the manuscript
*Real-Time Self-Collision Avoidance for Dual-Arm Manipulators via Dynamically Weighted Null-Space Control*.

The repository contains:

- the **proposed controller** (dynamically weighted null-space control with capsule-based detection),
- the **baseline controller** (Lei et al., *Appl. Sci.* 2020),
- the **stochastic benchmark** (trajectory generator, director node, Gazebo world and robot model),
- the **analysis scripts** used to produce every figure, table, and statistic of the manuscript.

## Repository structure

```
src/
  dual_manipulator_h/            Robot model (URDF/xacro, Gazebo world) and PROPOSED controller
    scripts/
      cinematica_puntos.py       Chain points and exact point Jacobians       (shared)
      benchmark_common.py        Bezier reference, DLS, inf-norm scaling, metrics, timing (shared)
      nodo_benchmark.py          ROS node common to both controllers         (shared)
      collision.py               Capsule model, distance matrix, pruning
      selfCollisionStrategy.py   Weighting matrix, repulsive/vortex fields, control law
      mainDualControl_lum.py     ROS entry point of the proposed controller
    tools/
      verificar_poda.py          Verification of the kinematic-impossibility filter
      sim_cinematica.py          Offline kinematic simulation of both controllers
      revisar_logs.py            Diagnostic of the per-cycle logs
      prueba_signos.py           Sign test of the Gazebo velocity interface
      prueba_degenerados.py      Tests of the degenerate cases of the secondary field (fallbacks)
  dual_manipulator_h_spheres/    BASELINE controller (Lei et al.)
    scripts/
      collision.py               Inflated discrete spheres and sensitivity index
      leiStrategy.py             Baseline control law
      mainDualControl_sph.py     ROS entry point of the baseline
  dual_manipulator_benchmark/    Benchmark
    scripts/
      generador_bezier_offline.py  Trajectory generator (fixed seed, feasibility rejection)
      benchmark_director.py        Trial sequencing, contact arbitration, CSV output
    config/                      Trajectory datasets (YAML)
results/
  analisis_estadistico.py        Outcomes, McNemar test, arm-specific Wilcoxon tests
  analisis_logs.py               Per-cycle analysis (timing, closing speed, path deviation)
  figuras_rev2.py                Figures of the manuscript
  principal.csv                  Results on the test benchmark (both methods, ablation, gain configurations)
  zonas.csv                      Results with the radii of the target spheres exchanged
  piloto.csv                     Pilot study for the clearance thresholds
```

The files marked *shared* are identical in both controller packages, so that trial termination,
metrics, timing, and logging are the same for both methods.

## Correspondence between the manuscript and the code

| Manuscript | Code |
|---|---|
| Kinematic chain points, capsule segments (Sec. III.A) | `cinematica_puntos.puntos_cadena` |
| Closest points between segments, parallel case (Sec. III.B-C) | `collision.segmento_segmento` |
| Topological distance matrix and surface clearance (Sec. III.D) | `collision.estado_geometrico` |
| Structural pruning and kinematic-impossibility filter (Sec. III.E) | `collision._construir_pares`, `tools/verificar_poda.py` |
| Weighted damped pseudoinverse and null-space operator (Sec. IV.A) | `benchmark_common.pinv_dls`, `ControladorPropuesto.calcular` |
| Joint-limit and collision weights, matrix W (Sec. IV.B) | `selfCollisionStrategy.H_lim`, `H_local`, `matriz_W` |
| Closest-point and relative Jacobians (Sec. IV.C) | `cinematica_puntos.jacobianos_cadena`, `jacobiano_en_segmento` |
| Repulsive field, vortex field, fallbacks (Sec. IV.D) | `selfCollisionStrategy.campo_secundario`, `_direccion_escape`, `tools/prueba_degenerados.py` |
| Low-pass filter and infinity-norm scaling (Sec. IV.E) | `ControladorPropuesto.calcular`, `benchmark_common.escalar_norma_infinito` |
| Baseline sensitivity index and repulsive law (Sec. VI.C) | `dual_manipulator_h_spheres/scripts/collision.beta_minimo`, `leiStrategy.ControladorLei` |
| Trajectory generation and feasibility rejection (Sec. V) | `generador_bezier_offline.py` |
| Metrics (Table 3) | `benchmark_common.MetricasBrazo` |

All numerical parameters of Table 2 are defined in one place for each controller:
`PARAMETROS_POR_DEFECTO` in `selfCollisionStrategy.py` and `PARAMETROS_LEI` in `leiStrategy.py`.

## Running the benchmark

```bash
colcon build && source install/setup.bash

# Terminal 1: simulation
ros2 launch dual_manipulator_h dual_manipulator.launch.py

# Terminal 2: ONE controller
ros2 launch dual_manipulator_h control_lum.launch.py            # proposed
ros2 launch dual_manipulator_h_spheres control_sph.launch.py    # baseline

# Terminal 3: benchmark director (run from the folder where the CSV should be written)
ros2 launch dual_manipulator_benchmark director.launch.py csv_file:=results.csv
```

Variants of the proposed controller used in the ablation and priority studies:
`control_lum.launch.py variant:=no_vortex | no_weighting | no_weighting_no_vortex | swapped | symmetric`.

Per-cycle logs are written to `~/benchmark_logs/<ALGORITHM>/trial_XXX.npz` (argument `log_dir`).

## Datasets

| File | Seed | Use |
|---|---|---|
| `trayectorias_50.yaml` | 2026 | Test benchmark (all reported results) |
| `trayectorias_piloto.yaml` | 1 | Pilot study for the selection of the clearance thresholds |
| `trayectorias_zonas_invertidas.yaml` | 2026, `--swap-zones` | Radii of the target spheres exchanged between the arms |

Regenerating a dataset with the same seed reproduces the file exactly:
`python3 generador_bezier_offline.py --seed 2026 --n 50`.

## Results and per-cycle logs

The `algorithm` column of `results/principal.csv` identifies the method or variant:
`LUMELSKY` (proposed), `LEI` (baseline), `LUMELSKY_no_vortex`, `LUMELSKY_no_weighting`,
`LUMELSKY_no_weighting_no_vortex`, `LUMELSKY_swapped` (gains exchanged between the arms) and
`LUMELSKY_symmetric`. If an execution appears more than once, the analysis scripts keep the last row.

The per-cycle logs of every trial (`.npz`, about 100 MB) are attached to the `v2.0` release of this
repository. Extract them and pass the folder that contains `LUMELSKY/`, `LEI/`, ... as `<log_dir>`.

## Reproducing the analysis

```bash
python3 results/analisis_estadistico.py results/principal.csv --yaml src/dual_manipulator_benchmark/config/trayectorias_50.yaml
python3 results/analisis_estadistico.py results/zonas.csv --yaml src/dual_manipulator_benchmark/config/trayectorias_zonas_invertidas.yaml
python3 results/analisis_logs.py <log_dir>
python3 results/figuras_rev2.py results/principal.csv --logs <log_dir> --out figures
python3 src/dual_manipulator_h/tools/verificar_poda.py
python3 src/dual_manipulator_h/tools/prueba_degenerados.py
```