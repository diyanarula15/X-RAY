import { useEffect, useMemo, useRef, useState } from 'react';
import { api, type Situation } from '../lib/api';
import type { P3Replay } from '../lib/api';
import { ReplayPanel } from '../components/P3Panel';
import { SectionTitle } from '../components/Readouts';
import { C } from '../lib/theme';
import { usePlayback } from '../store/playback';

/**
 * Pick a real situation out of the race — DISPLAY + NAVIGATION ONLY.
 *
 * THREE SEPARATED CONCEPTS, and the whole point of this view is that they are
 * not allowed to blur:
 *   X-RAY RECOMMENDATION   `recommendation`, canonical P2, the only call shown.
 *   OBSERVED OUTCOME       `observed_outcome`, a position delta over the window.
 *   DRIVER ACTION          not observed. No public channel carries it.
 *
 * This view used to print MATCHED / DIVERGED per row from
 * `matches_recommendation`, which was P2's call compared against an "action"
 * inferred from the position delta: a driver who attacked and failed was
 * recorded as having held, and a car promoted by the pit stop of the car ahead
 * was recorded as having attacked. 852 rows carried that verdict and none of
 * them could support it, so the verdict is gone. The position delta is still
 * shown, labelled as what it is, and the counterfactual — what would have
 * happened had the car taken X-RAY's call — is reported as unresolved, because
 * the race never branched. `matches_recommendation` / `actual_action` remain in
 * the payload as legacy fields and are not read here.
 *
 * This replaced a walkthrough that fired up to 40 SERIAL `/replay` requests to
 * find a divergent instance, with every control disabled for the duration. The
 * whole list now arrives in one request, precomputed to `out/decisions/`, so
 * filtering is instant.
 */

type Filter = 'all' | 'attack' | 'gained' | 'nochange';

// Chips describe the OBSERVED OUTCOME or the engine's own call. There is
// deliberately no "driver followed"/"driver disobeyed" chip: selecting rows by a
// claim the data cannot make is the same error as printing it per row.
const FILTERS: { id: Filter; label: string; hint: string }[] = [
  { id: 'all', label: 'all', hint: 'every causal decision point in the race' },
  { id: 'attack', label: 'ATTACK called',
    hint: 'points where canonical P2 recommended attacking' },
  { id: 'gained', label: 'gained position',
    hint: 'the car was classified further forward at the end of the window — '
        + 'an outcome, which pit stops and retirements ahead also produce' },
  { id: 'nochange', label: 'no position change',
    hint: 'same classified position at both ends of the window' },
];

function matches(s: Situation, f: Filter): boolean {
  if (f === 'all') return true;
  // Canonical P2 call, not P1's. The old `s.attack` was the P1 per-lap flag,
  // which made this chip select rows whose displayed recommendation came from a
  // different engine than the one that produced the number beside it.
  if (f === 'attack') return s.recommendation === 'ATTACK';
  // A null delta is an unresolved window, not a zero: it belongs to neither
  // bucket rather than being folded into "no change".
  if (f === 'gained') return (s.observed_position_delta ?? 0) > 0;
  return s.observed_position_delta === 0;
}

export function Situations({ raceId, car, rival }:
  { raceId: string; car: string; rival: string }) {
  const [list, setList] = useState<Situation[] | null>(null);
  const [refusal, setRefusal] = useState<string | null>(null);
  const [err, setErr] = useState<string | null>(null);
  const [filter, setFilter] = useState<Filter>('all');
  const [sel, setSel] = useState<number | null>(null);
  const [replay, setReplay] = useState<P3Replay | null>(null);
  const [loadingReplay, setLoadingReplay] = useState(false);
  const setView = usePlayback((s) => s.setView);
  const setTime = usePlayback((s) => s.setTime);
  const token = useRef(0);

  useEffect(() => {
    const mine = ++token.current;
    setList(null); setErr(null); setRefusal(null); setSel(null); setReplay(null);
    api.situations(raceId, car, rival)
      .then((d) => {
        if (token.current !== mine) return;
        setList(d.situations); setRefusal(d.refusal ?? null);
      })
      .catch((e) => { if (token.current === mine) setErr(String(e)); });
  }, [raceId, car, rival]);

  const shown = useMemo(
    () => (list ?? []).filter((s) => matches(s, filter)), [list, filter]);

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

  // Windows with a published position at BOTH ends. Not "rows with a verdict":
  // there is no verdict any more.
  const resolved = list.filter((s) => s.observed_position_delta !== null).length;

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
          to see what was knowable at that moment, and what physically happened
          in the window after it. {list.length} points, {resolved} with a
          published position at both ends of the window.
        </p>
        <p style={{ color: C.dim, fontSize: 12, lineHeight: 1.6, margin: '6px 0 0',
                    maxWidth: 780 }}>
          Counterfactual: unresolved from historical telemetry. The race never
          branched onto X-RAY&apos;s call, and no public channel says whether the
          driver chose to attack — so the outcome column is a change of classified
          position, not a driver action, and it also moves for pit stops,
          retirements ahead, penalties, incidents and safety cars.
        </p>

        <div style={{ display: 'flex', gap: 6, flexWrap: 'wrap', margin: '14px 0 10px' }}>
          {FILTERS.map((f) => {
            const n = (list ?? []).filter((s) => matches(s, f.id)).length;
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
                <th>X-RAY called (P2)</th>
                <th style={{ paddingRight: 12 }}>observed outcome</th>
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
                    <td style={{ color: C.gray }}>{s.recommended_zone ?? '—'}</td>
                    <td style={{ color: C.gray }}>
                      {s.gap_s == null ? '—' : `${s.gap_s.toFixed(2)} s`}</td>
                    {/* Canonical P2 decision — the only recommendation shown. */}
                    <td style={{ color: s.recommendation === 'ATTACK' ? C.green : C.gray,
                                 fontWeight: 700 }}>
                      {s.recommendation ?? '—'}</td>
                    {/* Observed outcome, not a verdict: no colour says right or
                        wrong, because a position delta cannot grade a call. */}
                    <td style={{ paddingRight: 12, color: C.gray }}>
                      {s.observed_outcome ?? 'position not published for this window'}</td>
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
            at the cutoff, what it would have called, and what physically
            happened in the window after. What the driver chose is not in the
            data, and what the recommendation would have produced is unresolved.
          </div>
        ) : (
          <>
            <div className="panel" style={{ padding: 15 }}>
              <SectionTitle>LAP {current.lap}</SectionTitle>
              <div style={{ color: C.gray, fontSize: 12.5, lineHeight: 1.7 }}>
                Decision at <b className="num" style={{ color: C.white }}>
                  {current.decision_time_s.toFixed(1)} s</b> session time
                {current.recommended_zone ? `, zone ${current.recommended_zone}` : ''}.
                <div style={{ marginTop: 7 }}>
                  X-RAY called <b style={{ color: current.recommendation === 'ATTACK'
                                             ? C.green : C.white }}>
                    {current.recommendation ?? '—'}</b>. Observed outcome:{' '}
                  <b style={{ color: C.white }}>
                    {current.observed_outcome ?? 'position not published for this window'}</b>.
                </div>
                <div style={{ marginTop: 7, color: C.dim, fontSize: 12 }}>
                  Counterfactual: unresolved from historical telemetry.
                </div>
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
          </>
        )}
      </div>
    </div>
  );
}
