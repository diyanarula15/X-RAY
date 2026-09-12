# Script Pipeline Order

This file is intentionally gitignored. It is a local map for the numbered script
entry points.

The naming convention is:

- `NN.name.py` for a serial pipeline stage.
- `NN.a_name.py`, `NN.b_name.py` for stages that sit at the same level and can
  be run independently after the previous numbered step is available.
- `99.name.py` for maintenance or audit helpers that are not part of the normal
  production flow.

## Synthetic Pipeline

`scripts/01.run_sim.py`
   Builds and optionally saves synthetic two-car ground truth.

`scripts/02.run_estimator.py`
   Runs the simulator, observes the public trace, estimates hidden energy, and
   scores estimator quality.

`scripts/03.a_validate_decision.py`
   Decision-service diagnostic. It checks one decision point end to end and
   proves future telemetry does not affect the past decision.

`scripts/03.b_run_decision_eval.py`
   Monte Carlo decision evaluation. It compares X-RAY against blind and oracle
   decisions using the estimator output.

`03.a` and `03.b` are parallel siblings after estimator output exists.

`scripts/04.simulate_attack.py`
   Closed-loop attack simulation. It executes the chosen plan in the simulator
   and scores whether the pass actually happens.

`scripts/05.run_ablation.py`
   Telemetry-rate ablation. It writes `out/ablation.json` and the MAPE plot used
   by later summary artifacts.

`scripts/06.a_make_summary.py`
   Builds the one-page summary figure. Requires `out/ablation.json` from step 5.

`scripts/06.b_make_video.py`
   Builds the demo video. It is an artifact-generation sibling of the summary
   figure and can be run independently.

`06.a` and `06.b` are parallel artifact builders.

## Real-Data Branch

`scripts/07.a_feasibility.py`
   Probes one real FastF1 session and writes `out/feasibility/r<round>.json`.

`scripts/07.b_run_real.py`
    Runs the Stage 1 real-session core end to end.

`07.a` and `07.b` are real-data siblings. `08.a` depends on the JSON produced by
`07.a`; `08.b` is the app-data path.

`scripts/08.a_feasibility_report.py`
    Aggregates feasibility JSON files into `out/feasibility.md`.

`scripts/08.b_analyse_race.py`
    Precomputes one real race into app-ready JSON.

## Maintenance And Audit

- `scripts/99.make_golden.py`
  Regenerates golden estimator outputs for tests. Use only when a numerics
  change is intentional.

- `scripts/99.worked_trace.py`
  Prints one worked decision trace and its causality proof.

- `scripts/_common.py`
  Shared CLI plumbing. It stays unnumbered because it is imported by the other
  scripts rather than run as a pipeline stage.
