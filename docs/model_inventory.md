# X-RAY Model Inventory

Labels every model component so presentation code cannot treat a heuristic or a
placeholder statistical model as physical ground truth.

Recreated after the `folder cleanups` commit removed `docs/` entirely. The P0
section is reconstructed from the code as it stands; the P1 sections are new.

## Physics

- Aerodynamic drag: `xray.vehicle.drag_force` (argument is RELATIVE AIRSPEED)
- Rolling resistance and grade force: `xray.vehicle.step`
- Power-to-force, `P = Fv`: `xray.vehicle.step`
- Battery store integration and bounds: `xray.vehicle.step`
- Longitudinal timestep simulation: `xray.vehicle.step` — the single
  authoritative force/energy integrator. No force equations in `decision.py`,
  `decision_service.py`, the API, the frontend, or `scripts/`.
- Energy-to-braking-point speed maps: `xray.decision.calibrate_zone`

## Regulation (single source)

- 2026 normal MGU-K curve: `xray.constants.mguk_power_limit_normal`
- 2026 Overtake curve: `xray.constants.mguk_power_limit_overtake`
- Dispatch on Overtake state: `xray.constants.mguk_power_limit`
- Per-lap legal recharge ceiling: `xray.constants.recharge_allowance_j`
  (7.0 MJ normally, 7.5 MJ on a lap Overtake was granted)
- Values: `xray.constants.POWER_UNIT_2026`
- `xray.regs` DERIVES its taper from those curves and restates no breakpoint.
  `test_one_regulation_curve_and_every_module_uses_it` enforces it, including a
  scan for hard-coded km/h breakpoints outside `constants.py`. It exists because
  `regs.py` once held an independent linear 290->355 km/h ramp while the
  simulator ran the piecewise curve, which made `balance.py` bound deployment
  18 kW below what the car used and excluded the true theta by 91 J.
- Event/session variants: `xray.regs.RegSet` / `PRE_MIAMI` / `POST_MIAMI`.

## Manual Override semantics (kept separate on purpose)

| concept | where |
|---|---|
| stored energy | `CarState.E` |
| deployed this lap | `CarState.deployed_lap` |
| harvested this lap | `CarState.harvested_lap` |
| per-lap legal recharge allowance | `constants.recharge_allowance_j` |
| Overtake granted this lap | `CarState.manual_overtake_allocation_j` |
| Overtake active | `CarState.overtake_active` |
| MGU-K permitted power | `constants.mguk_power_limit` |

The 0.5 MJ allocation raises the legal recharge ceiling and unlocks the Overtake
power curve. It never enters the store. Measured: the recharge half is inert in
Stage 1 (peak harvest 2.25 MJ/lap against a 7.0 MJ cap); only the power curve
currently changes anything.

## P1 environment (implemented)

- `xray.environment`: `EnvironmentalState`, `EnvironmentTimeline`,
  `environment_at_time` — takes the sample at or before the decision time and
  never interpolates through a later one.
- `xray.data.ingest._weather_trace` -> `SessionData.weather_trace`.
  `SessionData.weather` remains the session-mean summary; both are kept.
- FastF1 weather channels VERIFIED against the installed 3.8.3 parser:
  `Time, AirTemp, Humidity, Pressure, Rainfall, TrackTemp, WindDirection,
  WindSpeed`. `Rainfall` exists and is a bool. The feed updates once per minute,
  which is the real resolution of any causal weather claim here.
- Air density: `xray.data.ingest.air_density`, the sole implementation; the
  timeline calls it per sample rather than growing a second formula.
- `xray.data.circuits.RealTrack.tangent` from the real `CircuitGeometry.xy`.
  Circuit Sigma has no tangent, deliberately: its `xy()` is documented cosmetic.
- Wind, two frames kept apart:
  - ENU (east/north) carries true compass bearings.
  - local = `CircuitGeometry.xy`, whatever frame the FastF1 position feed used.
  - `TrackFrame.heading_deg_true_north` is the true bearing of the LOCAL +X
    axis and is the only thing connecting them. It is `null` by default and
    requires a documented `source`; `TrackFrame.oriented` rejects
    "unknown"/"assumed" so an undocumented value cannot enable the projection.
  - Airflow: `wind_vector_enu`, meteorological from-bearing, so the air travels
    toward bearing + 180: `(-s sin b, -s cos b)`.
  - Rotation: `tangent_to_enu`, `E = x sin t - y cos t`, `N = x cos t + y sin t`.
  - `wind_along_track_mps = dot(wind_enu, tangent_enu)`, POSITIVE = TAILWIND,
    and `v_air = v_car - wind_along_track`.
  - Unavailable is `available=False` with value `None`, never 0.0, so "unknown"
    cannot be read as "calm" and no headwind/tailwind is inferred.
- Wind double-counting is structurally prevented in `realfit._terms`:
  `v_air = v - w_along + v_wind`. `w_along` is measured (positive = tailwind);
  `v_wind` is fitted and has the OPPOSITE sign (positive = headwind), which is
  the historical convention. With no orientation `w_along` is None and the
  expression reduces exactly to the legacy `(v + v_wind)`, so the fitted term
  keeps its original meaning as the entire wind. With measured wind present it
  becomes a residual correction. `RealNuisanceFit.v_wind_is_residual` and
  `.wind_source` publish which meaning is in force; the field name is unchanged
  for compatibility.

## P1 aerodynamics (implemented, DISABLED BY DEFAULT)

- `F_down = 0.5 rho ClA v_air^2`: `xray.vehicle.downforce`
- `N = m g + F_down`: `xray.vehicle.normal_load`
- Load-dependent braking: `xray.vehicle.brake_mu`, `braking_distance`
- Wake split: `tow.cda_reduction` (drag) and `tow.dirty_air_downforce_loss`
  (load) from one decay in `xray.sim`; the apex-speed effect is DERIVED in
  `xray.vehicle.cornering_grip_from_downforce`, not configured in speed units.
- `xray.physics_context.PhysicsContext` (environment + optional tyre). Pit
  strategy is deliberately not in it.

## P1 tyres, stint and the tyre-aware DP (implemented)

- `xray.tyres`: `TyreModelParams`, `TyreState`, bounded thermal/wear/grip model.
  - `mu_eff = mu_base * f_temp(T) * f_wear(wear) * f_wet(wetness, compound)`
  - `N = m g + F_down`; `F_long,max = mu_long N`; `F_lat,max = mu_lat N`
  - `U = sqrt((F_long/F_long,max)^2 + (F_lat/F_lat,max)^2)`
  - `target = track_temp + heating_gain*U`; `T' = T + (target-T)(1-exp(-dt/tau))`
  - `dwear = base_wear_per_m * ds * thermal_mult(T) * (1 + gain*U^2)`
  - `f_temp` and `thermal_mult` have an OPTIMUM and rise/fall on both sides;
    neither is a single linear rule in temperature. Both are bounded.
  - **`tyre_life` is OBSERVED AGE IN LAPS and is never wear.** `wear_fraction`
    is modelled. `test_tyres` asserts the shortcut does not exist in the source.
- `xray.stint`: `StintState`, `PitContext`, and the oracle boundary.
  - causal sources: `team_plan`, `synthetic_plan`, `inferred`, `unknown`
  - `oracle_eval` is NOT causal; `require_causal` raises, and the decision
    service calls it before the solver sees anything.
  - There is **no validated rival pit-inference model**, so a rival's context is
    `unknown` with confidence 0 and the DP continues the worn state rather than
    inventing a reset.
- `xray.decision.TyreDecisionContext` + `_solve_exogenous_tyre`:
  `V[k, energy, wear]`. HOLD advances normal wear, ATTACK advances normal plus
  extra, a KNOWN stop resets wear to zero on the continuation.
  - The wear rates are DERIVED by integrating `tyres.advance` over a lap at two
    utilisations (`ASSUMED_UTIL_NORMAL` 0.60, `ASSUMED_UTIL_ATTACK` 0.95). They
    are not penalty constants.
  - **There is no near-pit reward multiplier.** `test_tyre_decision` scans
    `decision.py` for one and asserts the reward line mentions neither wear nor
    pits. "Close to pit is cheap" comes from the reset, and
    `attack_wear_continuation_penalty` is published so the claim is readable.
  - The wear axis is INTERPOLATED, not snapped. Nearest-bin quantisation made
    hold and attack land in the same bin and priced the entire tyre cost at
    exactly zero; a coarse grid is now exact.

## ASSUMED / SYNTHETIC parameters

| parameter | value | status |
|---|---|---|
| `overtake.p_pass` coefficients | b0..b3 | SYNTHETIC design anchors, never fitted |
| `vehicle.cla_straight` / `cla_corner` | 0.0 (reference 2.00 / 4.30) | SYNTHETIC, uncalibrated, **disabled by default** |
| `vehicle.brake_calibration_v_ms` | 80.0 | ASSUMED anchor speed |
| `tow.dirty_air_downforce_loss` | 0.20 | SYNTHETIC, 1-D in gap only |
| `tow.dirty_air_grip_loss` | 0.06 | legacy, active while ClA = 0 |
| environment rain/drying rates | 1/120, 1/600 per s | ASSUMED; `Rainfall` is measured, wetness is not |
| track-frame north offset | not supplied | wind projection REFUSES without it |
| `estimator.RESERVE_SIGMA` | 1.0e5 | synthetic observation-noise scale |
| `tyres.compounds.*` (all 11 coefficients x 6 compounds) | see config | SYNTHETIC, ordered plausibly, never calibrated |
| `tyres.ASSUMED_UTIL_NORMAL` / `_ATTACK` | 0.60 / 0.95 | ASSUMED effective lap utilisations |
| `MIN_THERMAL_GRIP` / `MAX_THERMAL_WEAR_MULTIPLIER` | 0.70 / 3.0 | ASSUMED bounds |
| pit-window distribution | uniform | ASSUMED; teams commit to windows, uniform is a guess |

Two things make the ClA pair plausible rather than calibrated: an L/D of about 3
against the existing CdA, and the fact that inverting `brake_decel_max` against
`cla_corner` gives a friction coefficient of 1.48, inside the real slick range
of roughly 1.4–1.8. Plausible is not measured.

**Why P1 aero ships disabled.** Enabling it flips the race outcome: the follower
overtakes, and the leader then spends 24.7% of the stint in a tow, breaking 18
fixtures that assume the leader ran in clear air. The physics is not wrong — the
apparently impossible 368 km/h is a towed car at CdA 0.495, not created energy —
but the parameter deciding the race order is one of the two uncalibrated ClA
numbers. An uncalibrated parameter does not get to pick the winner. Enabling it
is a separate step: calibrate, then regenerate every dependent fixture.

## Statistical / learned

- Overtake probability: `xray.overtake.p_pass`, `pass_model_calibration =
  "synthetic"`. Placeholder coefficients from design anchors, not a fit.
- Hidden usable-energy belief: `xray.estimator`, `xray.realfit`
- Field semantics: `usable_mean` (deployable store, J), `soc_mean` (raw store),
  `reserve_mean` (inferred buffer), `deployed_lap` (point estimate),
  `deployed_lap_posterior_mean` (posterior mean — a different quantity).

## Optimization

- Finite-horizon DP: `xray.decision.solve`
- Exogenous rival DP used by the web: `xray.decision.solve_exogenous`
- Canonical web/core interface: `xray.decision_service`

## Known limitations, measured

- **Per-lap deployed energy at 100 Hz is 7.7% mean / 9.5% worst** against a
  documented 4.6%. Almost pure bias (−9.47% mean, 0.65% spread) with CdA accurate
  to +0.68%: true deployment sits ON the regulatory ceiling and the point
  estimate is clipped to it, so noise can only land at or below the truth.
  Tracked by a strict `xfail` plus a separate bias test, not a widened bound.
  Needs boundary-aware deployment reconstruction.
- Set-membership falsification boundary sits at η ≈ 0.35, not 0.40.
- Decision sharpness: zone pass probabilities compress under the corrected
  regulation curve, so a narrow rival-energy belief self-agrees 0.825 of the
  time rather than 1.00.
- The policy-sensitivity fan was deleted, not ported;
  `/api/race/{id}/decision` returns `fan: {curves: [], n_policies: 0}`.
  `decision.policy_posterior` / `robustness` still exist and are what a
  replacement should use.
- Wake is 1-D in gap only; there is no lateral offset, so "alongside" and
  "directly behind" are the same place.
- Directional wind is unavailable on real tracks until a track-frame north
  offset is configured, and unavailable on Circuit Sigma permanently.
- **The decision service builds wear-INDEPENDENT zone maps.** `ZoneModel`
  supports wear surfaces and `calibrate_zone_with_wear` builds them, but
  `decision_service._cached_zone_models` still calls `build_model`, so in the
  service path the energy->speed map does not vary with wear and
  `attack_wear_continuation_penalty` comes out at 0. The DP machinery, the tyre
  model and the pit reset are all correct and tested in isolation
  (`test_tyre_decision`); what is missing is building and caching one speed
  surface per wear level, which is a per-wear-level `calibrate_zone` run.
- Rival pit timing is always `unknown` on real data: no validated inference
  model exists, and the actual pit laps are oracle-only.
- Tyre temperature has no public channel; the modelled value is unvalidated.
- Pit-window probabilities are uniform, which is an assumption, not a model.
