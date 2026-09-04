import type { Car } from './api';

/** A car, sampled at a moment of race time. Everything on screen reads this. */
export type CarSample = {
  idx: number;          // nearest trace index (for cloud lookup)
  s: number;            // m along the lap, interpolated
  lap: number;
  v: number;            // m/s
  deploy: number;       // kW
  harvest: number;      // kW
  usable: number; p10: number; p90: number;   // MJ
  dry: boolean; coast: boolean;
  onTrack: boolean;     // false outside the car's observed window
};

/** Binary search for the last index with t[i] <= t. */
function locate(t: number[], x: number): number {
  let lo = 0, hi = t.length - 1;
  if (x <= t[0]) return 0;
  if (x >= t[hi]) return hi;
  while (hi - lo > 1) {
    const mid = (lo + hi) >> 1;
    if (t[mid] <= x) lo = mid; else hi = mid;
  }
  return lo;
}

export function sampleCar(car: Car | null, raceTime: number, trackLength: number): CarSample | null {
  if (!car) return null;
  const tr = car.trace as any;
  const t: number[] = tr.t;
  if (!t || t.length < 2) return null;
  const i = locate(t, raceTime);
  const j = Math.min(i + 1, t.length - 1);
  const dt = t[j] - t[i];
  // Don't interpolate across a hole: a 1 s gap at 300 km/h invents 80 m.
  const f = dt > 0 && dt < 3.0 ? Math.min(Math.max((raceTime - t[i]) / dt, 0), 1) : 0;
  const lerp = (a: number[] | undefined) => {
    if (!a) return 0;
    const x = a[i] ?? 0, y = a[j] ?? x;
    return x + (y - x) * f;
  };
  // unwrap distance across the start line so the car does not slide backwards
  let s0 = tr.s[i], s1 = tr.s[j];
  if (s1 < s0 - trackLength / 2) s1 += trackLength;
  const s = (s0 + (s1 - s0) * f) % trackLength;
  return {
    idx: f < 0.5 ? i : j, s, lap: tr.lap[i], v: lerp(tr.v),
    deploy: lerp(tr.deploy_kw), harvest: lerp(tr.harvest_kw),
    usable: lerp(tr.usable_mean), p10: lerp(tr.usable_p10), p90: lerp(tr.usable_p90),
    dry: !!tr.dry[i], coast: !!tr.coast[i],
    onTrack: raceTime >= t[0] - 2 && raceTime <= t[t.length - 1] + 2,
  };
}

export function timeRange(car: Car | null): [number, number] | null {
  const t = (car?.trace as any)?.t as number[] | undefined;
  if (!t || t.length < 2) return null;
  return [t[0], t[t.length - 1]];
}

/** Session time at which the car begins the given lap. */
export function lapStart(car: Car | null, lap: number): number | null {
  const tr = car?.trace as any;
  if (!tr?.t) return null;
  const k = tr.lap.findIndex((l: number) => l === lap);
  return k >= 0 ? tr.t[k] : null;
}


/**
 * The moment these two were actually closest on track.
 *
 * The battle metadata says which lap they raced on, but a lap is 90 seconds and
 * the gap swings across it, so starting at the lap boundary can open the view
 * with them eight seconds apart. Scan the shared window for the minimum
 * on-track gap and start there instead.
 */
export function bestBattleTime(a: Car | null, b: Car | null, trackLength: number,
                               range: [number, number]): number | null {
  if (!a || !b) return null;
  const step = Math.max((range[1] - range[0]) / 1200, 0.5);
  let best = Infinity, bestT: number | null = null;
  for (let t = range[0]; t <= range[1]; t += step) {
    const sa = sampleCar(a, t, trackLength);
    const sb = sampleCar(b, t, trackLength);
    if (!sa || !sb || !sa.onTrack || !sb.onTrack) continue;
    // Skip the standing start: on lap 1 the whole field is three metres apart
    // at 40 km/h, which is the smallest gap of the race and not a battle.
    if (sa.lap <= 1 || sb.lap <= 1) continue;
    if (sa.v < 40 || sb.v < 40) continue;          // both genuinely racing
    let d = sb.s - sa.s;
    if (d > trackLength / 2) d -= trackLength;
    else if (d < -trackLength / 2) d += trackLength;
    const gap = Math.abs(d) / Math.max(Math.min(sa.v, sb.v), 5);
    if (gap < best) { best = gap; bestT = t; }
  }
  // back off a little so the approach is visible, not just the moment itself
  return bestT == null ? null : Math.max(bestT - 12, range[0]);
}
