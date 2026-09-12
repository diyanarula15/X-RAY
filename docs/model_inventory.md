# X-RAY Model Inventory

This inventory labels the main model components so presentation code does not
treat heuristics or placeholder statistical models as physical ground truth.

## Physics

- Aerodynamic drag: `xray.vehicle.drag_force`
- Rolling resistance and grade force: `xray.vehicle.step`
- Power-to-force relationship, `P = Fv`: `xray.vehicle.step`
- Battery store integration and bounds: `xray.vehicle.step`
- Longitudinal timestep simulation: `xray.vehicle.step`
- Energy-to-braking-point speed maps: `xray.decision.calibrate_zone`

## Regulation

- 2026 normal MGU-K power curve: `xray.constants.mguk_power_limit_normal`
- 2026 Overtake-active MGU-K power curve: `xray.constants.mguk_power_limit_overtake`
- Store, recovery, and Overtake allocation values: `xray.constants.POWER_UNIT_2026`

## Reduced-Order Physics

- Synthetic and payload-backed track interfaces reduce full circuit dynamics to
  straights, corner entries, apex speeds, grade, and zone severity.
- Braking remains a reduced force envelope rather than a full brake/tyre
  thermal model.

## Heuristics

- Synthetic zone braking severity: `xray.track.Track._build_zones`
- Real-data zone extraction from longest straights: `xray.data.circuits.RealTrack`
- Low-confidence gap fallback when trace/lap timing is unavailable:
  `xray.decision_service.gap_at_position`
- Current gap from public traces: same-time observation when available, otherwise
  a causal constant-velocity projection from the rival's latest already-observed
  state. This reports `gap_age_s` and lowers confidence as the sample gets stale.
- Current dirty-air modelling remains an exponential aerodynamic surrogate in
  `xray.sim` and `xray.policy`.
- Estimator dry-event likelihood scale: `xray.estimator.RESERVE_SIGMA`
  (`dry_event_e_scale` in config) controls how strongly an observed deployment
  cut-out collapses the usable-energy posterior. It is a synthetic calibration
  parameter for observation noise, not a physical battery parameter.
- Future rival usable-energy forecast: bounded historical deployed/harvested
  lap-rate transition in `xray.decision_service`; causal and store-bounded, but
  explicitly heuristic rather than posterior particle propagation.

## Statistical / Learned

- Overtake probability: `xray.overtake.p_pass`
- Calibration status: placeholder coefficients from synthetic design anchors,
  not empirical race-data calibration.
- Hidden usable-energy belief: `xray.estimator` and `xray.realfit`
- Estimator field semantics:
  - `usable_mean`: current deployable store-energy belief at a telemetry sample, J.
  - `soc_mean`: current raw store-energy belief at a telemetry sample, J.
  - `reserve_mean`: inferred held-back driver buffer, J.
  - `deployed_lap`: point-estimate deployed store energy integrated over a lap, J
    internally and MJ in web JSON.
  - `deployed_lap_posterior_mean`: posterior mean deployed store energy integrated
    over a lap, J internally.
  - `harvested_lap`: point-estimate recovered store energy integrated over a lap,
    J internally and MJ in web JSON.

## Optimization

- Finite-horizon DP: `xray.decision.solve`
- Exogenous rival-energy DP used by the web/API service:
  `xray.decision.solve_exogenous`
- Canonical web/core decision interface:
  `xray.decision_service.evaluate_overtake_decision`
