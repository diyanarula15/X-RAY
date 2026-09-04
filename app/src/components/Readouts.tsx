import { motion } from 'framer-motion';
import { useEffect, useRef, useState } from 'react';
import { C } from '../lib/theme';

/** Interpolate values, never snap. A readout counting from 2.6 to 1.9 MJ over
 *  400 ms feels like a live instrument; a jump feels like a page reload. */
export function Num({ value, digits = 2, suffix = '' }:
  { value: number | null; digits?: number; suffix?: string }) {
  const [shown, setShown] = useState(value ?? 0);
  const raf = useRef(0);
  const from = useRef(shown);
  const t0 = useRef(performance.now());
  useEffect(() => {
    if (value == null || !isFinite(value)) return;
    from.current = shown; t0.current = performance.now();
    const tick = () => {
      const u = Math.min((performance.now() - t0.current) / 400, 1);
      const e = 1 - Math.pow(1 - u, 3);
      setShown(from.current + (value - from.current) * e);
      if (u < 1) raf.current = requestAnimationFrame(tick);
    };
    cancelAnimationFrame(raf.current);
    raf.current = requestAnimationFrame(tick);
    return () => cancelAnimationFrame(raf.current);
  }, [value]);
  if (value == null || !isFinite(value)) return <span className="num">—</span>;
  return <span className="num">{shown.toFixed(digits)}{suffix}</span>;
}

/** Uncertainty is always visible. There is no point estimate without its band. */
export function EnergyBar({
  label, mean, p10, p90, colour, max = 4.0, unknown = false, sub,
}: {
  label: string; mean: number | null; p10?: number | null; p90?: number | null;
  colour: string; max?: number; unknown?: boolean; sub?: string;
}) {
  const pct = (v: number | null | undefined) =>
    `${Math.max(0, Math.min((v ?? 0) / max, 1)) * 100}%`;
  return (
    <div style={{ marginBottom: 14 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline',
                    marginBottom: 6 }}>
        <span style={{ color: C.gray, fontSize: 11, letterSpacing: '0.09em',
                       fontWeight: 700 }}>{label}</span>
        <span style={{ color: unknown ? C.dim : colour, fontSize: 19, fontWeight: 700 }}>
          {unknown ? '—' : <><Num value={mean} /> <span style={{ fontSize: 12 }}>MJ</span></>}
        </span>
      </div>
      <div style={{ position: 'relative', height: 15, background: '#0E0E13',
                    border: `1px solid ${C.panelBorder}`, borderRadius: 5, overflow: 'hidden' }}>
        {unknown ? (
          <div style={{ position: 'absolute', inset: 0, display: 'grid',
                        placeItems: 'center', color: C.red, fontWeight: 800, fontSize: 13 }}>?</div>
        ) : (
          <>
            {p90 != null && (
              <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0,
                            width: pct(p90), background: colour, opacity: 0.26 }} />
            )}
            {p10 != null && (
              <div style={{ position: 'absolute', left: 0, top: 0, bottom: 0,
                            width: pct(p10), background: colour, opacity: 0.95 }} />
            )}
            <div style={{ position: 'absolute', left: pct(mean), top: 0, bottom: 0,
                          width: 2, background: C.white }} />
          </>
        )}
      </div>
      {sub && <div style={{ color: C.dim, fontSize: 10.5, marginTop: 4 }}>{sub}</div>}
    </div>
  );
}

export function Panel({ children, style }: { children: any; style?: any }) {
  return <div className="panel" style={{ padding: 14, ...style }}>{children}</div>;
}

/** Refusals are loud. Never a silent fallback, never a quiet default. */
export function Refusal({ title, message, tone = 'red' }:
  { title: string; message: string; tone?: 'red' | 'amber' }) {
  const col = tone === 'red' ? C.red : C.amber;
  return (
    <motion.div initial={{ opacity: 0, y: 8 }} animate={{ opacity: 1, y: 0 }}
      className={tone === 'red' ? 'hatch' : ''}
      style={{ border: `1px solid ${col}`, borderRadius: 10, padding: '11px 13px',
               background: tone === 'red' ? 'rgba(225,6,0,0.07)' : 'rgba(255,195,0,0.07)' }}>
      <div style={{ color: col, fontWeight: 800, fontSize: 11.5, letterSpacing: '0.08em',
                    marginBottom: 5 }}>{title}</div>
      <div style={{ color: C.gray, fontSize: 12.5, lineHeight: 1.5 }}>{message}</div>
    </motion.div>
  );
}
