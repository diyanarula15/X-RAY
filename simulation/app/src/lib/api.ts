const BASE = (import.meta as any).env?.VITE_API ?? 'http://127.0.0.1:8011';

async function j<T>(url: string): Promise<T> {
  const r = await fetch(BASE + url);
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
export type RaceDetail = RaceSummary & {
  circuit_geometry: Geometry; refusals: Record<string, Refusal>;
  calibration: any; drivers: string[]; laps: any[];
  battles: { car: string; ahead: string; laps_close: number;
             median_gap: number; first_lap: number }[];
};

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
    v_max_mps: number | null; v_mean_mps: number | null }>;
  evaluation_fingerprint: string;
  quality_flags: Record<string, boolean>;
  energy_inference_status: string;
};

export const api = {
  races: () => j<RaceSummary[]>('/api/races'),
  race: (id: string) => j<RaceDetail>(`/api/race/${id}/summary`),
  car: (id: string, d: string) => j<Car>(`/api/race/${id}/car/${d}`),
  battle: (id: string, a: string, b: string) =>
    j<{ a: Car; b: Car; gaps: any[]; zones: Geometry['zones']; track_length: number }>(
      `/api/race/${id}/battle/${a}/${b}`),
  observability: (id: string) => j<any>(`/api/race/${id}/observability`),
  decision: (id: string, car: string, rival: string) =>
    j<any>(`/api/race/${id}/decision?car=${car}&rival=${rival}`),
  p2: (id: string, car: string, rival: string) =>
    j<P2Decision>(`/api/race/${id}/p2?car=${car}&rival=${rival}`),
  p3Status: () => j<P3Status>('/api/p3/status'),
  p3Race: (id: string) => j<any>(`/api/race/${id}/p3`),
  replay: (id: string, car: string, rival: string, cutoff: number, horizon = 30) =>
    j<P3Replay>(`/api/race/${id}/replay?car=${car}&rival=${rival}` +
                `&cutoff=${cutoff}&horizon=${horizon}`),
  rdd: (cutoff: number, bandwidth = 0.6) =>
    j<any>(`/api/rdd?cutoff=${cutoff}&bandwidth=${bandwidth}`),
};
