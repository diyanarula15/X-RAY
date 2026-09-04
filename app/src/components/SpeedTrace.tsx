import type { Car } from '../lib/api';
import { C } from '../lib/theme';

const TAPER_START = 290 / 3.6, TAPER_END = 355 / 3.6;

export function SpeedTrace({ subject, rival, frame, window = 260 }:
  { subject: Car | null; rival: Car | null; frame: number; window?: number }) {
  const W = 700, H = 118, pad = 26;
  const lo = Math.max(0, frame - window), hi = frame + 10;
  const path = (c: Car | null) => {
    if (!c) return '';
    const pts: string[] = [];
    for (let i = lo; i < Math.min(hi, c.trace.v.length); i++) {
      const v = c.trace.v[i]; if (v == null) continue;
      const x = pad + ((i - lo) / (hi - lo)) * (W - pad - 8);
      const y = H - 14 - (Math.min(v, 110) / 110) * (H - 30);
      pts.push(`${pts.length ? 'L' : 'M'}${x.toFixed(1)},${y.toFixed(1)}`);
    }
    return pts.join(' ');
  };
  const yFor = (v: number) => H - 14 - (Math.min(v, 110) / 110) * (H - 30);
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
        <text x={4} y={12} fill={C.gray} fontSize={9.5}>km/h</text>
      </svg>
    </div>
  );
}
