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
- ~~The decision service builds wear-INDEPENDENT zone maps.~~ **CLOSED.**
  `_cached_zone_models` now builds an energy-by-wear surface per physical
  context via `calibrate_zone_with_wear`, for both cars' compounds. The tyre
  reaches the map through the corner BEFORE the straight (`calibrate_zone`
  scales the run-up entry speed by `sqrt(grip_scale)`, the same law
  `vehicle._regime` uses), which is where a tyre actually changes a
  braking-point speed on a short straight. Superseded text follows for history:
- (historic) `ZoneModel`
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


## P1 closure (current tyre condition affects current physics)

- `decision_service.ZoneSurfaceContext` describes the reusable physical context
  of a speed surface: both compounds, the wear GRID, and quantised rho, track
  temperature and wetness. The current `wear_fraction` is deliberately NOT in
  it -- wear is the surface's axis, and keying on it would make 0.183742 and
  0.183891 separate cache entries and force a recalibration per sample.
- Quantisation steps (rho 0.005 kg/m^3, track temp 2 C, wetness 0.05) are
  deliberate, documented approximations, each far finer than the map's physical
  sensitivity.
- `TyreDecisionContext.rival_wear` carries the rival's CURRENT wear as exogenous
  state, the way their energy track already was. Our wear is the DP's state
  because our choices move it; theirs is a number we read.
- `tyre_state_at_opportunity` integrates wear over `max(laps seen, TyreLife)`
  when the fitting was NOT observed, and over the observed laps when it was.
  Age is an input to the integral, never its output.
- DP wear axis is 11 levels (`DP_WEAR_LEVELS`) and the surface axis 5
  (`SURFACE_WEAR_LEVELS`); both interpolate, so neither needs to be fine.


# P2 — strategic opportunity / deployment optimiser

## Modules

- `xray.opportunity` — `DecisionOpportunity`, `DecisionAction`, `ActionOutcome`,
  `TransitionModel`, `ScenarioSet`, `OpportunitySolution`, `solve_opportunities`.
  OPTIMIZATION only: every speed comes from the P1 `ZoneModel` surfaces, every
  wear increment from `xray.tyres`, every probability from `xray.overtake.p_pass`.
- `xray.decision_service.build_opportunity_horizon` /
  `evaluate_opportunity_decision` — ORCHESTRATION only. No Bellman recursion,
  no pass formula, no speed equation.
- `xray.passmodel` — dataset builder + audit for an empirical pass model.

## Action space

`HOLD`, plus one `ATTACK` per deployment level. Budgets are FRACTIONS of each
zone's own physical ceiling (`ZoneModel.energy_grid.max()`, i.e. what the P1
calibration actually deployed down that straight under the 2026 MGU-K curve),
configured at `decision.deployment_fractions` = [0.25, 0.5, 0.75, 1.0].
Fractions rather than joules so the set adapts per zone and survives a
regulation change; 1.0 keeps the legacy fixed-cost ATTACK representable.

Feasibility is reported in three separate causes: affordability
(budget > usable energy), deployability (budget > zone ceiling -> `saturated`,
with the undeployable remainder named), and store bounds (enforced upstream by
the P1 integrator). Measured ceilings on the test circuit: zone A 1.417 MJ,
B 1.178 MJ, C 1.306 MJ.

## Causality boundary in the horizon

Opportunity 0 is OBSERVED at its own decision point. Every later opportunity is
a FORECAST made at that instant and carries `decision_time_s = None`,
`source = "causal_forecast_..."`, and no posterior interval:

| field | source for opportunities 1..N |
|---|---|
| zone | track geometry, known in advance |
| gap | persistence of the current gap — HEURISTIC |
| rival energy | `_rival_energy_forecast`, the P1 bounded causal forecast |
| own energy / wear | not forecast; the DP carries them through its own transition |

The first implementation built later opportunities from actual later telemetry.
The causality test caught it: the action and every physical quantity matched,
but `value_action` moved, because the value of waiting had been computed from
data that had not happened yet.

## Equations (P2-new marked; the rest are reused)

- Bellman: `V_i(E,W) = max_a [ q(a)·R_i + (1-q(a))·(1-fail_cost)·V_{i+1} ]`
  for ATTACK, `V_{i+1}` for HOLD. `R_i = R_PASS·(N-i)/N` — the P0/P1 reward
  semantics, unchanged.
- Belief expectation: `Q(a) = Σ_j w_j Q(a | E_riv,j)` — the same form
  `xray.qmdp.decide` uses on the lap problem.
- Energy: `E' = clip(E − attack_deployed − normal_spend + harvest, 0, E_max)` —
  P0 semantics, three quantities kept separate.
- Wear: `W' = clip(W + inc, 0, 1)`, `inc` from `tyres.wear_per_lap` at
  `ASSUMED_UTIL_NORMAL` or `ASSUMED_UTIL_ATTACK`. A causal pit sets `W' = 0`.
- `decision_margin = EV(best) − EV(second best)` (P2-new)
- `action_consensus = Σ_j w_j · 1[argmax_a Q(a|j) = chosen]` (same definition as
  `qmdp.decide`)
- `expected_regret = Σ_j w_j [ max_a Q(a|j) − Q(chosen|j) ]` (P2-new)

State discretisation for the backward induction: energy on the P0/P1 bin width
(`E_STORE_MAX/19` ≈ 210 kJ), wear 0.02. Pinned against a 4× finer grid; it took
the 12-opportunity horizon from 45.1 s to 0.20 s.

## ASSUMED / SYNTHETIC parameters added by P2

| parameter | value | status |
|---|---|---|
| `decision.deployment_fractions` | 0.25/0.5/0.75/1.0 | HEURISTIC search resolution |
| `decision.scenario_weights` | 0.3/0.4/0.3 | ASSUMED quadrature over p10/mean/p90 |
| `decision.max_opportunities` | 12 | HEURISTIC horizon cap |
| `ENERGY_MEMO_QUANTUM_J` / `WEAR_MEMO_QUANTUM` | 210 kJ / 0.02 | numerical, pinned by test |
| forecast gap | persistence | HEURISTIC |
| `passmodel.CONFIRMATION_LAPS` | 2 | ASSUMED label window |

## Pass model: still SYNTHETIC, and why

`scripts/09.pass_dataset_audit.py` run against the 5 analysed races in
`out/races`: 5182 raw candidate rows, 1019 excluded (828 of them pit cycles
detected from compound changes and tyre-life resets — real published fields),
4163 usable, 148 positives (3.6%), 5 race groups across 5 circuits.

Sample count and grouping would pass. Two qualitative blockers stand:

1. **No track-status channel** in the payload, so safety-car and VSC position
   changes cannot be excluded. A probability fitted on that is not a probability
   of overtaking.
2. **No decision-time `delta_v` feature** reconstructible from these payloads —
   the physics feature the model exists to condition on.

`passmodel.fit` refuses while the audit blocks, and there is no force flag.
`pass_model_calibration` remains `"synthetic"` everywhere.


# P2 closure — frontend integration and closed-loop evaluation

## Frontend (display only)

- `simulation/api/main.py::p2_decision` — `GET /api/race/{rid}/p2`. Returns
  `decision_service.evaluate_opportunity_decision` unmodified.
- `simulation/app/src/lib/api.ts` — `P2Decision`/`P2CandidateAction`/
  `P2PosteriorRow` mirror `_serialise_p2` key for key, so a backend rename
  breaks the type rather than rendering `undefined`.
- `simulation/app/src/components/P2Panel.tsx` — recommendation, predicted
  state, strategic value, candidate-action table, rival-energy scenarios, input
  provenance. The only arithmetic is `*3.6` (km/h) and `*100` (percent);
  `tests/test_frontend_p2.py` enforces that, bans logistic/Bellman/threshold
  patterns across the app, and asserts every `p2.*` the panel reads exists in
  the real service response.
- Pass-model calibration is rendered as an unconditional inline badge, never a
  tooltip.
- P1-only / absent P2 data renders a fallback; the P1 trace is untouched.
- Validation: `tsc -b --force` exit 0 and `vite build` exit 0, run with the
  repo's own tsconfig against `app/node_modules` symlinked into
  `simulation/app/node_modules` (gitignored). oxlint clean.

## Closed-loop evaluation

`xray/closedloop.py` + `scripts/10.closedloop_eval.py`. The belief every policy
acts on is `observe()` -> `estimate()`; the simulator's stores are never read by
a policy. `oracle_p2` is the one labelled exception and is kept out of
`POLICIES`.

Adapter: `BudgetedAttack` extends the EXISTING `DeploymentPolicy.attack_lap` /
`attack_zone` hook with `attack_budget_j`. The allocation is a FLOOR on
deployment while it lasts, then reverts to the base policy. Two earlier
semantics were wrong and are recorded in the tests: reverting immediately made
the budget non-binding (0.4 MJ and 1.6 MJ both deployed 2.812 MJ), and cutting
to zero made a small allocation an anti-attack (peak speed fell 365.3 -> 364.9
km/h).

Result over 12 paired worlds (varying start gap, start energy, rival policy and
feed noise; same seed => same pass dice):

| policy | attacks | successes | mean ahead | interior budgets |
|---|---|---|---|---|
| hold | 143 att | 6 | 0.1569 | 0 |
| fixed_cost | 128 att, 12 plans | 7 | 0.1092 | 0 |
| max_deploy | 143 att, 7 plans | 6 | 0.1568 | 0 |
| p2 | 143 att, 7 plans | 6 | 0.1568 | 7 |
| oracle_p2 | 143 att, 12 plans | 6 | 0.1569 | 0 |

Paired: p2 vs hold 0/1/11 (delta -0.00006); p2 vs fixed_cost 1/3/8
(delta +0.04766); **p2 vs max_deploy 0/0/12 (delta 0.00000)**; oracle vs p2
1/0/11 (delta +0.00006). None of this is statistically significant and none is
claimed to be.

**The finding that matters: P2 and max-deployment are identical in every world.**
Circuit Sigma's zone A absorbs only ~0.449 MJ from the store in race conditions
-- the MGU-K taper and the zone length set that, not the optimiser -- so an
interior budget above that threshold is the same physical action as the maximum.
P2's budget axis is real in its own zone maps (`calibrate_zone` reports a
1.6 MJ ceiling from a full run-up) but the in-race drawdown is ~3.5x smaller, so
the deployment-magnitude feature has almost no room to express itself in Stage 1.

Secondary: `fixed_cost` attacks nearly twice as often and wins one more pass,
but is ahead **less** of the time (0.109 vs 0.157) -- spending everything costs
track position. And `oracle_p2` is within 0.00006 of `p2`, so in this setup the
rival-energy estimate is not the binding constraint on the decision.

---

# P3 — evidence, provenance and status

`xray/registry.py` is now the single source for every status word. This section is the prose
copy; the registry is the machine copy, and the API and UI read the registry, never this
file. Where they disagree, the registry is right and this file is stale.

## Statuses, and what they do NOT mean

| Status | Means | Does not mean |
|---|---|---|
| `REGULATION` | in the rulebook | measured |
| `PHYSICS` | derived from force/energy balance | validated end-to-end |
| `SYNTHETIC` | designed coefficients | fitted to anything |
| `HEURISTIC` | a rule of thumb | physical |
| `INFERRED` | produced by inference from observed data | **accurate** |
| `CALIBRATED` | fitted to data | validated out of sample |
| `EMPIRICAL` | fitted **and** validated out of sample | — |
| `DISABLED` | implemented, deliberately out of the path | broken |
| `UNAVAILABLE` | cannot be produced from the data we have | unimplemented |
| `RESEARCH_ONLY` | real code, real tests, never reaches a real race | wrong |

There is deliberately **no `VALIDATED` status.** No single word is both honest and short for
the energy estimator's situation — it is inferred, it has a mixed direct synthetic result,
and it has a negative real held-out result — so the status says what kind of number it is
and a separate validation record says what is known about it. A validation record is
allowed to say NO, and `available=False, result=NOT_ATTEMPTED` ("we never looked") is kept
distinct from `available=True, result=NO_ROBUST_IMPROVEMENT` ("we looked and it lost").

## Current statuses

| Component | Status | In a real race | Validation |
|---|---|---|---|
| Canonical energy inference | `INFERRED` | yes | real held-out: **NO ROBUST IMPROVEMENT** |
| Stage-1 estimator | `RESEARCH_ONLY` | no | not attempted |
| Set-membership / RBPF stack | `RESEARCH_ONLY` | no | not attempted |
| Pass probability `p_pass` | `SYNTHETIC` | yes | empirical fit **REFUSED** |
| Tyres | `SYNTHETIC` | yes | not attempted |
| Drag area CdA | `INFERRED` | yes | identified set; identifiability `LIMITED` |
| Downforce ClA | `DISABLED` | no | not demonstrated |
| Wake / dirty air | `SYNTHETIC` | yes | not attempted |
| Wetness | `HEURISTIC` | yes | not calibrated |
| Wind projection | `UNAVAILABLE` | no | declines rather than assuming 0° |
| Rival pit context | `UNAVAILABLE` | no | returns UNKNOWN |
| Future rival-energy forecast | `HEURISTIC` | yes | not attempted |
| Regulation variant | `REGULATION` | yes | correctness-driven |
| Vehicle mass model | `PHYSICS` (790 kg) | yes | **MEASURED — NOT ADOPTED** |
| Deployment-zone executable ceiling | `PHYSICS` | yes | measured through the shared integrator |

## The mass investigation: MEASURED — NOT ADOPTED

**790 kg remains active.** The alternative mass was measured and rejected.

A different mass won on the synthetic metric. It was not adopted, because the win depended
on simulator-specific fuel assumptions and did not transfer defensibly to the real path:
the synthetic fuel schedule is a config choice, so a mass tuned against it is partly tuned
against that choice. This is the cleanest example in the project of the P3 scientific rule:

> a synthetic metric improvement is **not** a real-path justification.

Recording the investigation as `MEASURED — NOT ADOPTED` rather than deleting it matters,
because the next person to see the synthetic number will otherwise re-derive it and adopt
it. The measurement stands; the decision not to act on it is the finding.

## The regulation-variant fix: correctness, not metrics

The regulation-variant fix is **correctness-driven and is not judged by whether identified
intervals narrowed.** Some did not. That is the expected consequence of applying the rule
actually in force instead of a more convenient one:

> the correct rule in force beats a prettier metric under the wrong rule.

A variant mismatch between simulator and fixtures once scaled every harvest measurement by
0.71× silently, which is why the artifact provenance for the regulation variant is
preserved rather than assumed.

## Pass model: still SYNTHETIC, empirical fit REFUSED

Dataset audit: 5,182 raw → 1,019 excluded → **4,163 usable, 148 positives (3.56%)** across
5 race groups. Blockers, all about the corpus rather than the fit:

- no track-status channel, so SC/VSC/yellow laps cannot be removed;
- no reliable pit timing;
- no decision-time physics `delta_v` in the persisted payload.

`fit()` refuses and has **no force flag**. The coefficients remain the only invented
constants in the codebase, and they propagate into every decision claim.

## What P3 measured about the energy estimator

Direct synthetic (mean of three seeds): deployment lap MAPE 5.71%, bias −5.32%, band
containment 93.4%; SOC MAE 0.180 MJ, SOC band containment 43.8% (35.7–55.0% across seeds);
CdA contained 3/3 but 658.6% of truth wide at identifiability 0.0.

Indirect real (48,920 held-out strict-future examples, 5 races, leave-one-race-out): X-RAY
minus fixed-energy MAE is **+0.437** m/s on future speed 5 s, **+0.279** on straight speed
3 s, and −0.325 on braking-point speed. Lower MAE is better, so X-RAY loses on two of three
targets.

**Claim that may not be made:** that the energy estimator is real-data validated, or that
real rival-energy accuracy is known. There is no public battery channel, so there is no
real ground truth to measure against at all.
