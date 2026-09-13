const BASE = (import.meta as any).env?.VITE_API ?? 'http://127.0.0.1:8011';

async function j<T>(url: string): Promise<T> {
  const r = await fetch(BASE + url);
  // 202 is "the decision trace for this pair is still solving" and its body is
  // a job record, not the payload. `r.ok` is true for 202 as well as 200, so
  // without this branch a progress report would be parsed as a decision
  // payload and render as a silently empty tab. Views gate on `useBundle`
  // before calling these, so reaching here means the bundle went away between
  // the poll and the fetch — surfaced, not swallowed.
  if (r.status === 202) throw new Error(`202 still solving ${url}`);
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json() as Promise<T>;
}

async function post<T>(url: string, body: unknown): Promise<T> {
  const r = await fetch(BASE + url, {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify(body),
  });
  if (!r.ok) throw new Error(`${r.status} ${url}`);
  return r.json() as Promise<T>;
}

export type RaceSummary = {
  id: string; event: string; circuit: string; date: string; round: number; year: number;
  n_cars: number; n_refused: number; has_elevation: boolean; track_length: number;
  identifiability: number; calibratable: boolean; cda_pooled: number | null;
  telemetry: { median_hz: number; laps_total: number; laps_usable: number; gap_limit_s: number };
  weather: Record<string, number>;
};
export type Geometry = {
  length: number; has_elevation: boolean;
  s: number[]; x: number[]; y: number[]; z: number[];
  grade: number[]; curvature: number[]; is_corner: boolean[];
  zones: { name: string; s_straight_start: number; s_straight_end: number; s_end: number;
           braking_severity: number; detection_point: number; length: number }[];
};
export type CarTrace = {
  t: number[];
  s: number[]; lap: number[]; v: number[];
  deploy_kw: number[]; harvest_kw: number[];
  usable_mean: number[]; usable_p10: number[]; usable_p90: number[];
  dry: boolean[]; coast: boolean[];
};
export type Car = {
  driver: string; identifiability: number; cda: number; cda_lo: number; cda_hi: number;
  inherited_pooled: boolean; n_coast_samples: number; reserve_mean: number;
  deployed_lap: Record<string, number>; harvested_lap: Record<string, number>;
  trace: CarTrace; cloud: number[][]; cloud_stride: number;
};
export type Refusal = { kind: string; message: string; identifiability?: number };
export type Regulation = {
  variant: string; p_harv_max_kw: number;
  p_dep_max_zone_kw: number; p_dep_max_elsewhere_kw: number;
  source: string;
  // An object in the artefact, not a string -- the old `string` type was a
  // latent lie that only survived because nothing rendered the field.
  zone_eligibility: { available: boolean; source: string; semantics: string;
                      separate_from_variant: boolean };
};
export type RaceDetail = RaceSummary & {
  circuit_geometry: Geometry; refusals: Record<string, Refusal>;
  // Optional: artefacts analysed before the variant was stamped do not carry it,
  // and the strip must say so rather than printing "undefined kW harvest".
  regulation?: Regulation | null;
  regulation_available?: boolean;
  calibration: any; drivers: string[]; laps: any[];
  // `solved`/`refusal`/`n_situations` come from the precomputed-bundle index.
  // `solved: false` means nobody has opened this pair yet, which is a third
  // state and is shown as one -- not folded into "refused".
  battles: { car: string; ahead: string; laps_close: number;
             median_gap: number; first_lap: number;
             solved: boolean; refusal: string | null;
             n_situations: number | null }[];
};

/** Job status for an on-demand Stage 2 analysis run, started via
 *  `api.analyzeRace`. Mirrors `simulation/api/main.py::_JOBS` verbatim. */
export type AnalyzeJob =
  | { status: 'running' }
  | { status: 'done'; race_id: string; path: string }
  | { status: 'error'; message: string; traceback?: string };

/** P2 strategic recommendation. Field names mirror
 *  `decision_service._serialise_p2` EXACTLY -- if one is renamed there, this
 *  breaks at the type level rather than rendering `undefined` in the UI.
 *  Every member is display data: nothing here is recomputed in TypeScript. */
export type P2CandidateAction = {
  action: string; kind: 'HOLD' | 'ATTACK';
  requested_budget_mj: number; actual_deployed_mj: number;
  feasible: boolean; infeasible_reason: string; saturated: boolean;
  own_speed_mps: number; rival_speed_mps: number; delta_v_mps: number;
  pass_probability: number; value: number | null;
};
export type P2PosteriorRow = {
  scenario_index: number; rival_usable_energy_mj: number; weight: number;
  optimal_action: string; optimal_value: number; chosen_action_value: number;
};
export type P2Decision = {
  decision: 'HOLD' | 'ATTACK';
  zone: string | null;
  opportunity_id: string; lap: number;
  decision_point_s: number; decision_time_s: number | null;
  deployment_budget_mj: number; actual_deployed_mj: number;
  deployment_saturated: boolean; saturation_reason: string;
  predicted_own_speed_mps: number; predicted_rival_speed_mps: number;
  predicted_delta_v_mps: number;
  pass_probability: number;
  pass_model_calibration: 'synthetic' | 'empirical';
  value_action: number; value_hold: number; decision_margin: number;
  next_best_action: { kind: string; zone: string | null;
                      deployment_budget_mj: number; value: number } | null;
  action_consensus: number; expected_regret: number;
  policy_posterior: P2PosteriorRow[];
  robustness: Record<string, number | string | null>;
  horizon: { opportunity_id: string; lap: number; zone: string;
             decision_s: number; decision_time_s: number | null; gap_s: number;
             own_usable_energy_mj: number; rival_usable_energy_mj: number;
             own_wear_fraction: number | null;
             rival_wear_fraction: number | null }[];
  candidate_actions: P2CandidateAction[];
  pit_reset_index: number | null;
  input_confidence: Record<string, string | number | null>;
  p2_version: string;
};

/** P3 evidence and model status. Mirrors `decision_service.p3_status` and
 *  `xray.registry`. Every status word here is a backend string -- the UI renders
 *  it and never derives one, because a status the frontend computed would be a
 *  second opinion about what a model is. */
export type P3Validation = {
  available: boolean; kind?: string; type?: string; result: string;
  metrics: Record<string, any>; artifact?: string | null;
  fingerprint?: string | null; provenance?: string | null; notes: string[];
};
export type P3Entry = {
  name: string; component: string; status: string; production: boolean;
  version: string | null; identifiability: string | null; reason: string | null;
  notes: string[]; validation: P3Validation | null;
};
export type P3EnergyStatus = {
  component: string; status: string; headline: string;
  synthetic_validation: P3Validation & { direct_truth: boolean };
  real_validation: P3Validation;
  identifiability: { status: string; reason: string };
  real_ground_truth: { available: boolean; reason: string };
  registry_version: string;
};
export type P3Status = {
  p3_version: string;
  registry: { registry_version: string; entries: P3Entry[];
              production_components: string[]; research_only_components: string[];
              data_quality_limitations: string[];
              energy_inference: P3EnergyStatus };
  energy_inference: P3EnergyStatus;
  data_quality_limitations: string[];
  pass_model: P3Entry; regulation_variant: P3Entry;
  mass_model: P3Entry; deployment_ceiling: P3Entry;
};
export type P3Replay = {
  p3_version: string; label: string; later_telemetry_use: string;
  cutoff_time_s: number; target_window_s: number[];
  information_at_cutoff: { n_samples: Record<string, number | null>;
                           laps_completed: any[]; pit_events_seen: any[];
                           weather: Record<string, any>;
                           input_fingerprint: string };
  inferred_energy_mj: Record<string, number> | null;
  p2_recommendation: P2Decision | null; p2_error: string | null;
  later_observable_outcome: Record<string, { n_samples: number;
    v_max_mps: number | null; v_mean_mps: number | null;
    position_at_cutoff: number | null; position_at_window_end: number | null }>;
  gap_to_rival_at_cutoff_s: number | null;
  gap_to_rival_at_window_end_s: number | null;
  // OBSERVED HISTORICAL OUTCOME: position delta over the evaluation window,
  // positive = places gained. An outcome, not a driver action.
  observed_position_before: number | null;
  observed_position_after: number | null;
  observed_position_delta: number | null;
  observed_outcome: string | null;
  observed_outcome_basis: string;
  // The driver's choice is not in any public channel, so it is reported as
  // unobserved and the counterfactual as unresolved rather than inferred from
  // the position delta.
  driver_action_observed: boolean;
  counterfactual_status: string;
  // LEGACY/INTERNAL, kept so existing readers do not break. `actual_action` is
  // the position delta relabelled as intent and `matches_recommendation` is P2
  // compared against that relabelling; neither is evidence about what the driver
  // did. Do not introduce new readers.
  actual_action: 'attacked' | 'held' | null;
  matches_recommendation: boolean | null;
  evaluation_fingerprint: string;
  quality_flags: Record<string, boolean>;
  energy_inference_status: string;
};

/** One real decision point, already replayed off-policy server-side.
 *
 *  Every field is copied through from `decision_service.historical_replay`; the
 *  frontend derives none of them. `observed_position_delta` is `null`, not 0,
 *  when the window has no published position at both ends -- a third state. */
/** One causal decision point, already replayed off-policy by the backend.
 *
 *  Every recommendation field below comes from ONE canonical P2 solver result --
 *  the same one that produced `matches_recommendation`. The old shape carried P1's
 *  `attack` next to P2's verdict, and the two disagreed on 137 of 852 rows; the
 *  P1 call is now `legacy_p1_*` and must not be rendered as the recommendation. */
export type Situation = {
  lap: number; decision_time_s: number; gap_s: number | null;
  // canonical P2 recommendation
  recommendation: 'HOLD' | 'ATTACK' | null;
  recommended_zone: string | null;
  deployment_budget_mj: number | null; actual_deployed_mj: number | null;
  pass_probability: number | null;
  value_action: number | null; value_hold: number | null;
  decision_margin: number | null;
  next_best_action: { kind: string; zone: string | null;
                      deployment_budget_mj: number; value: number } | null;
  pass_model_calibration: string | null;
  recommendation_source: string;
  // OBSERVED HISTORICAL OUTCOME: a position delta from the public
  // `laps[].position` field, positive = places gained. It is an outcome, not an
  // action -- pit stops, retirements ahead, penalties, incidents, traffic and
  // safety cars all move it, and an attack that failed moves it not at all.
  observed_position_before: number | null;
  observed_position_after: number | null;
  observed_position_delta: number | null;
  observed_outcome: string | null;
  observed_outcome_basis: string;
  // DRIVER ACTION: never observed. No public channel carries the driver's
  // choice, so there is no followed/disobeyed verdict to render, and the
  // counterfactual is unresolved because the race never branched onto P2's call.
  driver_action_observed: boolean;
  counterfactual_status: string;
  // LEGACY/INTERNAL. The old position-derived action and the MATCHED/DIVERGED
  // verdict built on it. Kept for compatibility only -- rendering either one is
  // the bug this pass removed; `Situations.tsx` reads neither.
  matches_recommendation: boolean | null;
  matches_recommendation_is_legacy?: boolean;
  actual_action: 'attacked' | 'held' | null;
  actual_action_is_legacy?: boolean;
  inferred_action_from_position: 'attacked' | 'held' | null;
  actual_action_basis: string;
  // legacy P1 per-lap call. Internal only: never display as the recommendation.
  legacy_p1_attack: boolean;
  legacy_p1_requested_zone: string | null;
  replay_error: string | null;
};
export type SituationList = {
  race: string; car: string; rival: string; horizon_s: number;
  situations: Situation[];
  // Set when the engine legitimately declined this pair. A refusal is a correct
  // output, not an error path, and it is rendered as the reason rather than
  // swallowed into an empty list. It no longer fires for a single
  // opportunity without a causal sample (Zandvoort's s = 0 zone on lap 1):
  // that opportunity is skipped and the rest of the race is still solved.
  refusal: string | null;
};

/** Stage 1's measured sample-rate ablation, served from `out/ablation.json`.
 *  These are SIMULATOR numbers. They were literals in the old Method view and
 *  had already drifted from the artefact (100 Hz read 5.0 against a measured
 *  7.68), which is exactly why they are fetched now. */
export type Ablation = {
  rates: number[]; mape: number[]; coverage: number[];
  cda_abs_err_pct: number[]; failures: Record<string, unknown>;
};

/** One axis of the LLM judge's rubric. `score` is served as 0, 1 or 2 and is
 *  rendered as the digit the server sent -- it is never averaged, re-banded or
 *  turned into a pass/fail here. */
export type JudgeAxis = {
  score: 0 | 1 | 2; note: string; evidence_refs: string[];
};

/** One verdict over one precomputed situation.
 *
 *  GENERATED COMMENTARY. Every string here was written by a language model
 *  reading the serialized evidence bundle, and the server labels it as such:
 *  `is_measurement` is always false. Nothing here is a measurement, nothing
 *  here is physics, and nothing here is recomputed in TypeScript.
 *
 *  The judge was never shown `actual_action` or `matches_recommendation`, so a
 *  verdict is NOT a second opinion about whether the driver agreed. It answers
 *  a different question: did the evidence support the call as stated.
 *
 *  `verdict_label` and `verdict_tone` come from the server on purpose. An
 *  enum-to-label or enum-to-colour map in this file would be the frontend
 *  holding its own opinion about what a verdict means, which is the drift the
 *  static scans in `tests/test_frontend_p2.py` exist to prevent. */
export type JudgeVerdict = {
  verdict: string;
  verdict_label: string;
  verdict_tone: 'ok' | 'warn' | 'bad' | 'neutral';
  summary: string;
  judge_confidence: string;
  abstained: boolean;
  abstain_reason: string | null;
  axes: Record<string, JudgeAxis>;
  overclaims: { claim: string; field: string; why_wrong: string }[];
  label: string;
  is_measurement: false;
  model_id: string;
  prompt_version: string;
  rubric_version: string;
  tool_calls: string[];
  n_turns: number;
  generated_at: string;
  judge_error: string | null;
};

/** Every cached verdict for one pair, keyed by `decision_time_s.toFixed(2)`.
 *
 *  `available: false` is a normal answer: `out/` is gitignored, so a fresh
 *  checkout has no verdicts until someone runs the batch. `stale` means the
 *  race was re-analysed after these verdicts were written -- they are still
 *  served, and flagged, rather than hidden. */
export type JudgeSet = {
  race: string; car: string; rival: string;
  available: boolean; reason: string | null;
  label: string; is_measurement: false; stale: boolean;
  model_id: string; prompt_version: string; rubric_version: string;
  judge_version: string; generated_at: string | null;
  verdicts: Record<string, JudgeVerdict>;
};

/** Progress of one pair's decision-trace solve. Mirrors
 *  `simulation/api/main.py::_bundle_job` verbatim.
 *
 *  A pair nobody has solved before costs ~112 s of particle-filter and DP work
 *  (measured, one 22-lap pair). That used to run inline on the request, which
 *  blocked every other endpoint in the API process — including the race fetch
 *  behind the track picker. It now runs off-process and reports here. */
export type BundleJob = {
  status: 'ready' | 'building' | 'error';
  job_id?: string;
  elapsed_s?: number;
  message?: string | null;
  race?: string; car?: string; rival?: string;
};

export const api = {
  races: () => j<RaceSummary[]>('/api/races'),
  race: (id: string) => j<RaceDetail>(`/api/race/${id}/summary`),
  car: (id: string, d: string) => j<Car>(`/api/race/${id}/car/${d}`),
  observability: (id: string) => j<any>(`/api/race/${id}/observability`),
  decision: (id: string, car: string, rival: string) =>
    j<any>(`/api/race/${id}/decision?car=${car}&rival=${rival}`),
  p2: (id: string, car: string, rival: string) =>
    j<P2Decision>(`/api/race/${id}/p2?car=${car}&rival=${rival}`),
  situations: (id: string, car: string, rival: string) =>
    j<SituationList>(`/api/race/${id}/situations?car=${car}&rival=${rival}`),
  p3Status: () => j<P3Status>('/api/p3/status'),
  // Horizon is deliberately not passed: the API derives it from the race's own
  // median lap time. The old fixed 30 s was shorter than a lap everywhere, so
  // the window closed before either car crossed the line and the observed
  // position delta came back null at every decision point on every circuit.
  replay: (id: string, car: string, rival: string, cutoff: number) =>
    j<P3Replay>(`/api/race/${id}/replay?car=${car}&rival=${rival}&cutoff=${cutoff}`),
  rdd: (cutoff: number, bandwidth = 0.6) =>
    j<any>(`/api/rdd?cutoff=${cutoff}&bandwidth=${bandwidth}`),
  ablation: () => j<Ablation>('/api/ablation'),
  // Read-only: the server serves verdicts produced offline by
  // `scripts/17.judge_situations.py`. No request here ever reaches a model.
  judge: (id: string, car: string, rival: string) =>
    j<JudgeSet>(`/api/race/${id}/judge?car=${car}&rival=${rival}`),
  judgeReport: () => j<any>('/api/judge/report'),
  // Cheap poll for "is this pair's decision trace ready". Asking starts the
  // solve if nothing is running; see `main.py::bundle_status`. Polled instead
  // of re-fetching `/situations`, which is ~940 kB per attempt.
  bundleStatus: (id: string, car: string, rival: string) =>
    j<BundleJob>(`/api/race/${id}/bundle?car=${car}&rival=${rival}`),
  analyzeRace: (round: number, year = 2026, session = 'R') =>
    post<{ job_id: string }>('/api/races/analyze', { round, year, session }),
  analyzeStatus: (jobId: string) => j<AnalyzeJob>(`/api/races/analyze/${jobId}`),
};
