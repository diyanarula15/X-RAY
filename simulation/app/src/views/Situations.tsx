import { useEffect, useMemo, useRef, useState } from 'react';
import { api, type Situation } from '../lib/api';
import type { JudgeSet, P3Replay } from '../lib/api';
import { ReplayPanel } from '../components/P3Panel';
import { JudgePanel } from '../components/JudgePanel';
import { SectionTitle } from '../components/Readouts';
import { C } from '../lib/theme';
import { usePlayback } from '../store/playback';

/**
 * Pick a real situation out of the race — DISPLAY + NAVIGATION ONLY.
 *
 * Every row is a causal decision point the engine already evaluated, and every
 * comparison shown (`matches_recommendation`, `actual_action`) is read verbatim
 * from `decision_service.historical_replay` via `/api/race/{id}/situations`.
 * Nothing here simulates a branch the race did not take: "diverged" means the
 * driver's real action differed from X-RAY's off-policy call, not that we know
 * what would have happened instead.
 *
 * This replaced a walkthrough that fired up to 40 SERIAL `/replay` requests to
 * find a divergent instance, with every control disabled for the duration. The
 * whole list now arrives in one request, precomputed to `out/decisions/`, so
 * filtering is instant.
 */

type Filter = 'all' | 'matched' | 'diverged' | 'attack' | 'judged';

const FILTERS: { id: Filter; label: string; hint: string }[] = [
  { id: 'all', label: 'all', hint: 'every causal decision point in the race' },
  { id: 'matched', label: 'driver matched the call',
    hint: 'the driver did what X-RAY would have called' },
  { id: 'diverged', label: 'driver diverged',
    hint: 'the driver did the opposite of the call' },
  { id: 'attack', label: 'ATTACK called',
    hint: 'points where the engine recommended attacking' },
  { id: 'judged', label: 'LLM-judged',
    hint: 'points a language model has commented on' },
];

/** The verdict key for a row. The server keys verdicts by the same
 *  `decision_time_s` this table already uses as its React key and selection
 *  identity, so the join is a lookup and never a search. */
function judgeKey(s: Situation): string {
  return s.decision_time_s.toFixed(2);
}

function matches(s: Situation, f: Filter, judge: JudgeSet | null): boolean {
  if (f === 'all') return true;
  if (f === 'attack') return s.attack;
  // Presence of a verdict, not a re-derivation of one.
  if (f === 'judged') return judge?.verdicts?.[judgeKey(s)] != null;
  // `matches_recommendation` is null when the comparison could not be made at
  // all -- those rows belong to neither bucket and are excluded from both,
  // rather than being folded into "diverged" as a false.
  if (f === 'matched') return s.matches_recommendation === true;
  return s.matches_recommendation === false;
}

export function Situations({ raceId, car, rival }:
  { raceId: string; car: string; rival: string }) {
  const [list, setList] = useState<Situation[] | null>(null);
  const [refusal, setRefusal] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const [sel, setSel] = useState<number | null>(null);
  const [replay, setReplay] = useState<P3Replay | null>(null);
  const [judge, setJudge] = useState<JudgeSet | null>(null);
  const [loadingReplay, setLoadingReplay] = useState(false);
  const setView = usePlayback((s) => s.setView);
  const setTime = usePlayback((s) => s.setTime);
  const token = useRef(0);

  useEffect(() => {
    const mine = ++token.current;
    setList(null); setErr(null); setRefusal(null); setSel(null); setReplay(null);
    setJudge(null);
    api.situations(raceId, car, rival)
      .then((d) => {
        if (token.current !== mine) return;
        setList(d.situations); setRefusal(d.refusal ?? null);
      })
      .catch((e) => { if (token.current === mine) setErr(String(e)); });
    // Verdicts are optional commentary generated offline, and most pairs have
    // none. A failure here must never block the table, so it resolves to null
    // rather than propagating to `err`.
    api.judge(raceId, car, rival)
      .then((d) => { if (token.current === mine) setJudge(d.available ? d : null); })
      .catch(() => { if (token.current === mine) setJudge(null); });
  }, [raceId, car, rival]);

  const shown = useMemo(
    () => (list ?? []).filter((s) => matches(s, filter, judge)),
    [list, filter, judge]);

  // Selection is by decision time, not list index: the index moves when the
  // filter changes, and a selection that silently jumps to a different lap
  // when you click a filter is worse than no selection at all.
  const current = useMemo(
    () => (list ?? []).find((s) => s.decision_time_s === sel) ?? null, [list, sel]);

  useEffect(() => {
    if (current == null) { setReplay(null); return; }
    const mine = ++token.current;
    setLoadingReplay(true);
    api.replay(raceId, car, rival, current.decision_time_s)
      .then((r) => { if (token.current === mine) setReplay(r); })
      .catch(() => { if (token.current === mine) setReplay(null); })
      .finally(() => { if (token.current === mine) setLoadingReplay(false); });
  }, [current, raceId, car, rival]);

  if (err) {
    return (
      <div style={{ padding: 34, color: C.red, fontSize: 13, lineHeight: 1.7 }}>
        Could not load situations for {car} vs {rival}. {err}
      </div>
    );
  }
  if (list === null) {
    return (
      <div style={{ padding: 34, color: C.dim, fontSize: 13, lineHeight: 1.7,
                    maxWidth: 640 }}>
        Building the decision trace for {car} vs {rival}…
        <div style={{ marginTop: 8, color: C.dim, fontSize: 12 }}>
          Precomputed pairs load instantly. A pair nobody has opened before is
          solved once, which takes about a minute, and is on disk from then on.
        </div>
      </div>
    );
  }
  if (!list.length) {
    return (
      <div style={{ padding: 34, maxWidth: 700 }}>
        <div style={{ color: C.amber, fontSize: 13, lineHeight: 1.7 }}>
          {refusal
            ? <>The engine declines this pair: <b style={{ color: C.white }}>{refusal}</b></>
            : <>No causal decision points for {car} vs {rival} in this race — these
               two were never close enough, for long enough, for the engine to have
               had a call to make.</>}
        </div>
        {refusal && (
          <div style={{ color: C.gray, fontSize: 12.5, lineHeight: 1.65, marginTop: 10 }}>
            A refusal is a result, not a failure — the estimator says so rather
            than producing a number it cannot support. Pick another pairing from
            the bar above; the battles listed there are ranked by how long the
            two actually raced each other.
          </div>
        )}
      </div>
    );
  }

  const resolved = list.filter((s) => s.matches_recommendation !== null).length;

  return (
    <div style={{ position: 'absolute', inset: 0, display: 'grid',
                  gridTemplateColumns: 'minmax(0,1fr) 400px', gap: 20,
                  padding: '22px 28px', overflow: 'hidden' }}>
      <div style={{ display: 'flex', flexDirection: 'column', minHeight: 0 }}>
        <h2 className="display" style={{ fontSize: 25, margin: 0 }}>
          Situations — {car} vs {rival}
        </h2>
        <p style={{ color: C.gray, fontSize: 13, lineHeight: 1.6, margin: '6px 0 0',
                    maxWidth: 780 }}>
          Every point in this race where the engine had a call to make. Pick one
          to see what was knowable at that moment and what the driver actually
          did next. {list.length} points, {resolved} with an observable outcome.
        </p>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '14px 0 10px' }}>
          {FILTERS.map((f) => {
            const n = (list ?? []).filter((s) => matches(s, f.id, judge)).length;
            const on = filter === f.id;
            return (
              <button key={f.id} onClick={() => setFilter(f.id)} title={f.hint}
                style={{ padding: '6px 11px', fontSize: 12, borderRadius: 7,
                         border: `1px solid ${on ? C.amber : C.panelBorder}`,
                         background: 'transparent', color: on ? C.amber : C.gray }}>
                {f.label} <span style={{ color: C.dim }}>{n}</span>
              </button>
            );
          })}
        </div>

        {!shown.length && (
          <div style={{ color: C.amber, fontSize: 12.5, lineHeight: 1.6, maxWidth: 700 }}>
            No situation in this race matches that filter. That is the real answer
            for this pair, not an empty page — nothing is being withheld.
          </div>
        )}

        <div className="panel" style={{ padding: 0, overflow: 'auto', minHeight: 0,
                                        flex: 1 }}>
          <table className="num" style={{ width: '100%', fontSize: 12.5,
                                          borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: C.dim, fontSize: 11, textAlign: 'left',
                           position: 'sticky', top: 0, background: C.panel }}>
                <th style={{ padding: '9px 12px' }}>lap</th>
                <th>zone</th>
                <th>gap</th>
                <th>X-RAY called</th>
                <th>driver</th>
                <th>outcome</th>
                <th style={{ paddingRight: 12 }}>LLM judge</th>
              </tr>
            </thead>
            <tbody>
              {shown.map((s) => {
                const on = s.decision_time_s === sel;
                return (
                  <tr key={s.decision_time_s}
                      onClick={() => setSel(s.decision_time_s)}
                      style={{ cursor: 'pointer', borderTop: `1px solid ${C.grid}`,
                               background: on ? 'rgba(255,195,0,0.10)' : 'transparent' }}>
                    <td style={{ padding: '8px 12px', color: C.white }}>{s.lap}</td>
                    <td style={{ color: C.gray }}>{s.requested_zone ?? '—'}</td>
                    <td style={{ color: C.gray }}>
                      {s.gap_s == null ? '—' : `${s.gap_s.toFixed(2)} s`}</td>
                    <td style={{ color: s.attack ? C.green : C.gray, fontWeight: 700 }}>
                      {s.attack ? 'ATTACK' : 'HOLD'}</td>
                    <td style={{ color: C.gray }}>{s.actual_action ?? '—'}</td>
                    <td style={{ color: s.matches_recommendation === true ? C.green
                                   : s.matches_recommendation === false ? C.amber : C.dim }}>
                      {s.matches_recommendation === true ? 'matched'
                        : s.matches_recommendation === false ? 'diverged'
                          : 'not observable'}</td>
                    {/* The label and its colour are both server fields. A
                        mapping table here would be a second opinion about what
                        a verdict means. */}
                    <td style={{ paddingRight: 12, color: C.dim }}>
                      {judge?.verdicts?.[judgeKey(s)]?.verdict_label ?? '—'}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>

      <div style={{ overflowY: 'auto', minHeight: 0, display: 'flex',
                    flexDirection: 'column', gap: 12 }}>
        {current == null ? (
          <div className="panel" style={{ padding: 15, color: C.dim, fontSize: 12.5,
                                          lineHeight: 1.65 }}>
            <SectionTitle>NO SITUATION SELECTED</SectionTitle>
            Pick a row to replay that moment off-policy: what the engine could see
            at the cutoff, what it would have called, and what the driver did in
            the window after.
          </div>
        ) : (
          <>
            <div className="panel" style={{ padding: 15 }}>
              <SectionTitle>LAP {current.lap}</SectionTitle>
              <div style={{ color: C.gray, fontSize: 12.5, lineHeight: 1.7 }}>
                Decision at <b className="num" style={{ color: C.white }}>
                  {current.decision_time_s.toFixed(1)} s</b> session time
                {current.requested_zone ? `, zone ${current.requested_zone}` : ''}.
              </div>
              <button
                onClick={() => { setTime(current.decision_time_s); setView('replay'); }}
                style={{ marginTop: 11, padding: '7px 13px', fontSize: 12,
                         borderRadius: 7, border: `1px solid ${C.amber}`,
                         background: 'transparent', color: C.amber, width: '100%' }}>
                watch this moment in Replay ▸
              </button>
            </div>
            {loadingReplay
              ? <div style={{ color: C.dim, fontSize: 12 }}>replaying…</div>
              : <ReplayPanel replay={replay} />}
            <JudgePanel verdict={judge?.verdicts?.[judgeKey(current)] ?? null}
                        stale={judge?.stale} />
          </>
        )}
      </div>
    </div>
  );
}
