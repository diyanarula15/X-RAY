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
  rdd: (cutoff: number, bandwidth = 0.6) =>
    j<any>(`/api/rdd?cutoff=${cutoff}&bandwidth=${bandwidth}`),
};
