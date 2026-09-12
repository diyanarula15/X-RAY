import { useEffect, useRef, useState } from 'react';
import { api } from '../lib/api';
import type { P3Replay } from '../lib/api';
import { ReplayPanel } from './P3Panel';
import { C } from '../lib/theme';

/**
 * Step through real situations from this race — DISPLAY + NAVIGATION ONLY.
 *
 * "Follow" and "disobey" never simulate anything: they walk forward through
 * the race's own decision points (the same causal cutoffs P1 already scores)
 * looking for one where the driver's real action did, or did not, match
 * X-RAY's off-policy call — both read verbatim from `replay.matches_recommendation`,
 * computed server-side. If no such situation exists in the remainder of the
 * race, that is what gets shown, not a fabricated one.
 */

type Candidate = { lap: number; t: number };

const MAX_SEARCH_STEPS = 40;

export function ScenarioWalkthrough({ raceId, car, rival }:
  { raceId: string; car: string; rival: string }) {
  const [candidates, setCandidates] = useState<Candidate[] | null>(null);
  const [idx, setIdx] = useState(0);
  const [replay, setReplay] = useState<P3Replay | null>(null);
  const [loading, setLoading] = useState(false);
  const [notice, setNotice] = useState<string | null>(null);
  const cancelled = useRef(false);

  useEffect(() => {
    cancelled.current = false;
    setCandidates(null); setReplay(null); setIdx(0); setNotice(null);
    api.decision(raceId, car, rival).then((d: any) => {
      if (cancelled.current) return;
      const rows: Candidate[] = (d?.laps ?? [])
        .filter((r: any) => r.decision_time_s != null)
        .map((r: any) => ({ lap: r.lap, t: r.decision_time_s }));
      setCandidates(rows);
    }).catch(() => setCandidates([]));
    return () => { cancelled.current = true; };
  }, [raceId, car, rival]);

  useEffect(() => {
    if (!candidates || !candidates.length) return;
    const c = candidates[idx];
    if (!c) return;
    setLoading(true);
    api.replay(raceId, car, rival, c.t).then((r) => {
      if (!cancelled.current) setReplay(r);
    }).catch(() => { if (!cancelled.current) setReplay(null); })
      .finally(() => { if (!cancelled.current) setLoading(false); });
  }, [candidates, idx, raceId, car, rival]);

  // Walk forward (or back) through the real cutoffs looking for one whose
  // real action matched (`want=true`) or diverged from (`want=false`) the
  // off-policy call. This is real-instance navigation, not a simulation: it
  // stops the moment it finds a real match, and admits it plainly if it never
  // does within the rest of the race.
  async function seek(direction: 1 | -1, want: boolean) {
    if (!candidates || !candidates.length) return;
    setLoading(true); setNotice(null);
    let steps = 0;
    let i = idx;
    while (steps < Math.min(MAX_SEARCH_STEPS, candidates.length)) {
      i += direction;
      if (i < 0 || i >= candidates.length) {
        setNotice(want
          ? `No further opportunity where the driver's action matched X-RAY's call, ${direction > 0 ? 'later' : 'earlier'} in this race.`
          : `No opportunity in this race where the driver's action diverged from X-RAY's call, ${direction > 0 ? 'later' : 'earlier'} in this race.`);
        setLoading(false);
        return;
      }
      steps += 1;
      const c = candidates[i];
      // eslint-disable-next-line no-await-in-loop
      const r = await api.replay(raceId, car, rival, c.t).catch(() => null);
      if (r && r.matches_recommendation === want) {
        setIdx(i); setReplay(r); setLoading(false);
        return;
      }
    }
    setNotice(`Checked ${steps} further opportunities without finding one that ${
      want ? 'matched' : 'diverged from'} the call — try the other direction.`);
    setLoading(false);
  }

  if (candidates === null) {
    return <div style={{ color: C.dim, fontSize: 12 }}>loading situations…</div>;
  }
  if (!candidates.length) {
    return (
      <div style={{ color: C.gray, fontSize: 12.5, lineHeight: 1.6, maxWidth: 340 }}>
        No decision points found for this pair to replay.
      </div>
    );
  }

  const cur = candidates[idx];
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 9, width: 360 }}>
      <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em', fontWeight: 700 }}>
        SCENARIO WALKTHROUGH — SITUATION {idx + 1} OF {candidates.length}
        {cur && ` · LAP ${cur.lap}`}
      </div>
      <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap' }}>
        <button onClick={() => setIdx(Math.max(idx - 1, 0))} disabled={loading}
          style={{ padding: '5px 9px', fontSize: 11, borderRadius: 6,
                   border: `1px solid ${C.panelBorder}`, background: 'transparent',
                   color: C.gray }}>◂ prev</button>
        <button onClick={() => setIdx(Math.min(idx + 1, candidates.length - 1))} disabled={loading}
          style={{ padding: '5px 9px', fontSize: 11, borderRadius: 6,
                   border: `1px solid ${C.panelBorder}`, background: 'transparent',
                   color: C.gray }}>next ▸</button>
        <button onClick={() => seek(1, true)} disabled={loading}
          style={{ padding: '5px 9px', fontSize: 11, borderRadius: 6,
                   border: `1px solid ${C.green}`, background: 'transparent',
                   color: C.green }}>follow — next matching</button>
        <button onClick={() => seek(1, false)} disabled={loading}
          style={{ padding: '5px 9px', fontSize: 11, borderRadius: 6,
                   border: `1px solid ${C.amber}`, background: 'transparent',
                   color: C.amber }}>disobey — next divergent</button>
      </div>
      {notice && (
        <div style={{ color: C.amber, fontSize: 11, lineHeight: 1.5 }}>{notice}</div>
      )}
      {/* The permanent off-policy disclaimer lives inside ReplayPanel itself
          (`replay.label`), rendered unconditionally there — not duplicated here. */}
      <ReplayPanel replay={replay} />
    </div>
  );
}
