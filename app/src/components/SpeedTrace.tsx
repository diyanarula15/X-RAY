import type { Car } from '../lib/api';
import { C } from '../lib/theme';

const TAPER_START = 290 / 3.6, TAPER_END = 355 / 3.6;

/** Speed against race time, windowed on the playhead. Both cars on the same
 *  clock, so the traces line up the way the cars do. */
export function SpeedTrace({ subject, rival, raceTime, window = 45 }:
  { subject: Car | null; rival: Car | null; raceTime: number; window?: number }) {
  const W = 700, H = 118, pad = 26;
  const lo = raceTime - window * 0.8, hi = raceTime + window * 0.2;
  const yFor = (v: number) => H - 14 - (Math.min(v, 110) / 110) * (H - 30);
  const path = (c: Car | null) => {
    const tr = c?.trace as any;
    if (!tr?.t) return '';
    // A null sample must lift the pen and the next real one must start a fresh
    // subpath. Pushing an empty string instead made pts.length non-zero, so the
    // following point emitted "L" with no preceding "M" and the whole path
    // silently rendered nothing.
    const pts: string[] = [];
    let pen = false;
    for (let i = 0; i < tr.t.length; i++) {
      const t = tr.t[i];
      if (t < lo) continue;
      if (t > hi) break;
      const v = tr.v[i];
      if (v == null || !isFinite(v)) { pen = false; continue; }
      const x = pad + ((t - lo) / (hi - lo)) * (W - pad - 8);
      pts.push(`${pen ? 'L' : 'M'}${x.toFixed(1)},${yFor(v).toFixed(1)}`);
      pen = true;
    }
    return pts.join(' ');
  };
  const nowX = pad + ((raceTime - lo) / (hi - lo)) * (W - pad - 8);
  return (
    <div className="panel" style={{ padding: '8px 10px' }}>
      <svg width="100%" height={H} viewBox={`0 0 ${W} ${H}`} preserveAspectRatio="none">
        {[TAPER_START, TAPER_END].map((v, i) => (
          <g key={i}>
            <line x1={pad} x2={W - 8} y1={yFor(v)} y2={yFor(v)}
              stroke={C.dim} strokeDasharray="4 4" strokeWidth={1} />
            <text x={pad + 3} y={yFor(v) - 4} fill={C.dim} fontSize={9}>
              {Math.round(v * 3.6)} km/h {i ? 'deployment zero' : 'taper starts'}
            </text>
          </g>
        ))}
        <path d={path(subject)} fill="none" stroke={C.amber} strokeWidth={1.8} />
        <path d={path(rival)} fill="none" stroke={C.red} strokeWidth={1.8} />
        <line x1={nowX} x2={nowX} y1={6} y2={H - 10} stroke={C.white} strokeWidth={1.4}
          opacity={0.8} />
        <text x={4} y={12} fill={C.gray} fontSize={9.5}>km/h</text>
      </svg>
    </div>
  );
}
