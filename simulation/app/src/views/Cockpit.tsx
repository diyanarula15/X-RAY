import { useEffect, useRef, useState } from 'react';
import * as d3 from 'd3';
import { api } from '../lib/api';
import type { P2Decision, P3Status } from '../lib/api';
import { CandidateActionsTable } from '../components/CandidateActionsTable';
import { NextBestAction } from '../components/NextBestAction';
import { HorizonTimeline } from '../components/HorizonTimeline';
import { P2Panel } from '../components/P2Panel';
import { SectionTitle } from '../components/Readouts';
import { C } from '../lib/theme';

/**
 * Cockpit — the first thing a race engineer sees. One question: what should
 * we do, right now. Every field below is read verbatim from `/decision` and
 * `/p2`; no threshold, ranking or expected-value arithmetic lives here (see
 * `tests/test_frontend_p4.py` and `test_frontend_p2.py`, which both scan this
 * file for banned patterns). Validation tables live on the Evidence tab, not
 * here — the first viewport answers "attack or hold", not "how well-validated
 * is the estimator".
 *
 * The former "Strategy" tab is folded in below the fold: same P2Panel, same
 * opportunity horizon, same threshold trace. Two tabs both answering "attack
 * or hold" from the same two endpoints was a split with nothing behind it, and
 * the headline call and the reasoning behind it now sit on one screen.
 */

function Badge({ text, color }: { text: string; color: string }) {
  return (
    <span style={{ color, border: `1px solid ${color}`, borderRadius: 3,
                   padding: '2px 6px', fontSize: 10.5, fontWeight: 700,
                   letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>{text}</span>
  );
}

const mj = (x: number | null | undefined, d = 3) =>
  x == null ? '—' : `${x.toFixed(d)} MJ`;
const ms = (x: number | null | undefined) =>
  x == null ? '—' : `${x.toFixed(2)} m/s`;
const pct = (x: number | null | undefined, d = 0) =>
  x == null ? '—' : `${(x * 100).toFixed(d)}%`;

export function Cockpit({ raceId, car, rival }:
  { raceId: string; car: string; rival: string }) {
  const [d, setD] = useState<any>(null);
  const [p2, setP2] = useState<P2Decision | null>(null);
  const [p3, setP3] = useState<P3Status | null>(null);
  const [sel, setSel] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const ref = useRef<SVGSVGElement>(null);
  const token = useRef(0);

  useEffect(() => {
    // A stale response from the previously-selected pair must not overwrite the
    // current one. Swapping drivers twice quickly used to leave whichever
    // request happened to finish last on screen, regardless of what was picked.
    const mine = ++token.current;
    setD(null); setP2(null); setSel(null); setLoading(true);
    const guard = <T,>(f: (v: T) => void) => (v: T) => { if (token.current === mine) f(v); };
    Promise.allSettled([
      api.decision(raceId, car, rival).then(guard(setD)),
      api.p2(raceId, car, rival).then(guard(setP2)),
      api.p3Status().then(guard(setP3)),
    ]).finally(() => { if (token.current === mine) setLoading(false); });
  }, [raceId, car, rival]);

  // P1's per-lap row for "now" -- the current call if one exists, otherwise
  // the first row.
  const row = d?.call ?? d?.laps?.[0] ?? null;

  // The threshold trace. Redrawn whenever the decision payload changes, which
  // includes a driver swap -- the old view left the previous pair's scatter on
  // screen until the new payload happened to arrive.
  useEffect(() => {
    if (!ref.current) return;
    const svg = d3.select(ref.current); svg.selectAll('*').remove();
    if (!d?.laps?.length) return;
    const W = 820, H = 380, m = { l: 58, r: 20, t: 18, b: 44 };
    svg.attr('viewBox', `0 0 ${W} ${H}`);
    const laps = d.laps as any[];
    const x = d3.scaleLinear().domain(d3.extent(laps, (r) => r.lap) as [number, number])
      .range([m.l, W - m.r]);
    const y = d3.scaleLinear().domain([0, 1]).range([H - m.b, m.t]);

    svg.append('g').attr('transform', `translate(0,${H - m.b})`).call(d3.axisBottom(x).ticks(10))
      .call((g) => { g.selectAll('line,path').attr('stroke', C.grid);
                     g.selectAll('text').attr('fill', C.gray).attr('font-size', 11); });
    svg.append('g').attr('transform', `translate(${m.l},0)`).call(d3.axisLeft(y).ticks(5))
      .call((g) => { g.selectAll('line,path').attr('stroke', C.grid);
                     g.selectAll('text').attr('fill', C.gray).attr('font-size', 11); });
    svg.append('text').attr('x', W / 2).attr('y', H - 8).attr('fill', C.gray)
      .attr('font-size', 12).attr('text-anchor', 'middle').text('lap');
    svg.append('text').attr('transform', 'rotate(-90)').attr('x', -H / 2).attr('y', 14)
      .attr('fill', C.gray).attr('font-size', 12).attr('text-anchor', 'middle')
      .text('pass probability');

    svg.append('path').datum(laps)
      .attr('d', d3.line<any>().x((r) => x(r.lap)).y((r) => y(r.tau)) as any)
      .attr('stroke', C.white).attr('stroke-width', 2.2).attr('fill', 'none');
    svg.append('text').attr('x', x(laps[Math.floor(laps.length * 0.62)].lap))
      .attr('y', y(laps[Math.floor(laps.length * 0.62)].tau) - 9)
      .attr('fill', C.white).attr('font-size', 12).attr('font-weight', 700)
      .text('attack threshold τ');

    svg.append('g').selectAll('circle').data(laps).join('circle')
      .attr('cx', (r: any) => x(r.lap)).attr('cy', (r: any) => y(r.q))
      .attr('r', (r: any) => (r.attack ? 6 : 4))
      .attr('fill', (r: any) => (r.attack ? C.green : C.gray))
      .attr('stroke', C.bg).attr('stroke-width', 2).style('cursor', 'pointer')
      .on('click', (_e, r: any) => setSel(r));

    if (d.call) {
      svg.append('circle').attr('cx', x(d.call.lap)).attr('cy', y(d.call.q)).attr('r', 13)
        .attr('fill', 'none').attr('stroke', C.gray).attr('stroke-width', 2.4);
      svg.append('text').attr('x', x(d.call.lap) + 20).attr('y', y(d.call.q) - 18)
        .attr('fill', C.gray).attr('font-size', 13).attr('font-weight', 700)
        // "the call" was the P1 row's own label. On 29% of decision points it
        // names a different action than canonical P2, so the word "call" here
        // was claiming an authority this layer does not have.
        .text(`legacy P1 per-lap call — lap ${d.call.lap}, zone ${d.call.zone}`);
    }
  }, [d]);

  if (loading && !d && !p2) {
    return (
      <div style={{ padding: 34, color: C.dim, fontSize: 13, lineHeight: 1.7,
                    maxWidth: 640 }}>
        Solving the decision trace for {car} vs {rival}…
        <div style={{ marginTop: 8, fontSize: 12 }}>
          Precomputed pairs load instantly. A pair nobody has opened before is
          solved once, which takes about a minute, and is on disk from then on.
        </div>
      </div>
    );
  }
  if (!d && !p2) {
    return (
      <div style={{ padding: 34, color: C.amber, fontSize: 13, lineHeight: 1.7,
                    maxWidth: 640 }}>
        No decision trace for {car} vs {rival}. These two were never close enough,
        for long enough, for the engine to have had a call to make — or one of
        them was refused by the estimator for this race.
      </div>
    );
  }

  // P2 is the only thing allowed in the headline slot. This line used to read
  // `p2 ? p2.decision === 'ATTACK' : !!row?.attack`, so a failed /p2 fetch
  // silently promoted the P1 per-lap flag into the 46px ATTACK/HOLD readout
  // with nothing on screen saying which layer produced it. The audit measured
  // P1 disagreeing with canonical P2 on 961 of 3,298 decision points (29%), so
  // that fallback printed the wrong call about one time in three. `null` is the
  // third state -- decline rather than guess; the legacy trace below keeps the
  // diagnostic without being called a recommendation.
  const attack: boolean | null = p2 ? p2.decision === 'ATTACK' : null;
  const synthetic = p2 ? p2.pass_model_calibration !== 'empirical' : true;
  const chosenLabel = p2
    ? (attack ? `ATTACK(${p2.zone}, ${p2.deployment_budget_mj.toFixed(3)} MJ)` : 'HOLD')
    : '';

  return (
    <div style={{ padding: '26px 34px', height: '100%', overflow: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, flexWrap: 'wrap' }}>
        <div className="num" style={{ fontSize: attack == null ? 28 : 46, fontWeight: 800,
                                      color: attack == null ? C.gray
                                             : attack ? C.green : C.amber }}>
          {attack == null ? 'NO CANONICAL RECOMMENDATION'
                          : attack ? 'ATTACK' : 'HOLD'}
          {attack && p2?.zone ? ` — ZONE ${p2.zone}` : ''}
        </div>
        {/* P1's confidence is not the confidence in the headline, and unlabelled
            next to a P2 headline it read as if it were. Named for its layer. */}
        {row?.confidence != null && (
          <Badge text={`legacy P1 confidence ${pct(row.confidence, 0)}`}
                 color={C.gray} />
        )}
        {p3?.energy_inference && (
          <Badge text={`ENERGY: ${p3.energy_inference.status}`} color={C.amber} />
        )}
      </div>

      {!p2 && (
        <div style={{ color: C.amber, fontSize: 13, lineHeight: 1.7, maxWidth: 760,
                      marginTop: 8 }}>
          P2 unavailable — no canonical strategic recommendation for this pair.
          The legacy P1 per-lap trace below is still shown as a diagnostic, but it
          is not a recommendation: it disagrees with canonical P2 on 961 of 3,298
          audited decision points (29%).
        </div>
      )}

      {p2 && (
        <div style={{ color: C.gray, fontSize: 13, lineHeight: 1.7, maxWidth: 760,
                      marginTop: 8 }}>
          value {attack ? 'attack' : 'hold'} {p2.value_action.toFixed(3)} vs.
          the alternative {p2.value_hold.toFixed(3)} — margin{' '}
          {p2.decision_margin.toFixed(3)}
          {p2.decision_margin < 0.01 ? ' (a close call)' : ''}.
          {' '}Gap is {row?.gap_s != null ? `${row.gap_s.toFixed(2)} s` : 'unknown'}
          {row?.gap_source ? ` (${row.gap_source})` : ''}.
        </div>
      )}

      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr 1fr', gap: 14,
                    marginTop: 18, maxWidth: 1180 }}>
        <div className="panel" style={{ padding: 15 }}>
          <SectionTitle>ENERGY</SectionTitle>
          <table className="num" style={{ width: '100%', fontSize: 12.5 }}>
            <tbody>
              <tr><td style={{ color: C.gray }}>own usable</td>
                  <td style={{ textAlign: 'right', color: C.amber }}>
                    {mj(row?.own_usable_energy_mj)}</td></tr>
              <tr><td style={{ color: C.gray }}>rival usable (mean)</td>
                  <td style={{ textAlign: 'right', color: C.red }}>
                    {mj(row?.rival_usable_energy_mj)}</td></tr>
              <tr><td style={{ color: C.gray }}>rival range (p10–p90)</td>
                  <td style={{ textAlign: 'right', color: C.red }}>
                    {row?.rival_usable_p10_mj != null && row?.rival_usable_p90_mj != null
                      ? `${row.rival_usable_p10_mj.toFixed(3)}–${row.rival_usable_p90_mj.toFixed(3)}`
                      : '—'}</td></tr>
            </tbody>
          </table>
        </div>

        <div className="panel" style={{ padding: 15 }}>
          <SectionTitle>PREDICTED</SectionTitle>
          <table className="num" style={{ width: '100%', fontSize: 12.5 }}>
            <tbody>
              <tr><td style={{ color: C.gray }}>own speed</td>
                  <td style={{ textAlign: 'right' }}>{ms(row?.predicted_own_speed_mps)}</td></tr>
              <tr><td style={{ color: C.gray }}>rival speed</td>
                  <td style={{ textAlign: 'right' }}>{ms(row?.predicted_rival_speed_mps)}</td></tr>
              <tr><td style={{ color: C.gray }}>Δv</td>
                  <td style={{ textAlign: 'right', color: C.green }}>
                    {ms(row?.predicted_delta_v_mps)}</td></tr>
              <tr><td style={{ color: C.gray }}>P(pass)</td>
                  <td style={{ textAlign: 'right' }}>
                    {pct(p2?.pass_probability ?? row?.pass_probability, 1)}</td></tr>
            </tbody>
          </table>
          <div style={{ marginTop: 8, padding: '5px 8px', borderRadius: 4,
                        background: synthetic ? 'rgba(201,162,39,0.14)' : 'rgba(0,160,90,0.14)',
                        color: synthetic ? C.amber : C.green, fontSize: 10.5, fontWeight: 700 }}>
            PASS MODEL: {(p2?.pass_model_calibration ?? 'synthetic').toUpperCase()}
            {synthetic ? ' — NOT EMPIRICALLY CALIBRATED' : ''}
          </div>
        </div>

        <div className="panel" style={{ padding: 15 }}>
          <SectionTitle>TYRES &amp; PIT</SectionTitle>
          <table className="num" style={{ width: '100%', fontSize: 12.5 }}>
            <tbody>
              <tr><td style={{ color: C.gray }}>own tyre</td>
                  <td style={{ textAlign: 'right' }}>
                    {row?.own_tyre ? `${row.own_tyre.compound} · ${row.own_tyre.tyre_life_laps}L` : '—'}</td></tr>
              <tr><td style={{ color: C.gray }}>own wear</td>
                  <td style={{ textAlign: 'right', color: C.amber }}>
                    {pct(row?.own_tyre?.wear_fraction, 1)}</td></tr>
              <tr><td style={{ color: C.gray }}>rival wear</td>
                  <td style={{ textAlign: 'right', color: C.amber }}>
                    {pct(row?.rival_wear_fraction, 1)}</td></tr>
              <tr><td style={{ color: C.gray }}>pit</td>
                  <td style={{ textAlign: 'right' }}>
                    {row?.pit_resets_next_lap ? 'resets next lap' : (row?.pit_context?.source ?? 'unknown')}</td></tr>
            </tbody>
          </table>
        </div>
      </div>

      {p2 && (
        <div className="panel" style={{ padding: 15, marginTop: 14, maxWidth: 1180 }}>
          <SectionTitle>CANDIDATE ACTIONS</SectionTitle>
          <CandidateActionsTable rows={p2.candidate_actions} chosenLabel={chosenLabel} />
          <div style={{ marginTop: 9 }}>
            <NextBestAction nb={p2.next_best_action} decisionMargin={p2.decision_margin} />
          </div>
        </div>
      )}

      {/* ------ the former Strategy tab: the horizon and the reasoning ------ */}
      <div style={{ borderTop: `1px solid ${C.panelBorder}`, margin: '26px 0 0',
                    paddingTop: 20, maxWidth: 1180 }}>
        <h3 className="display" style={{ fontSize: 21, margin: '0 0 4px' }}>
          Strategy horizon
        </h3>
        <p style={{ color: C.gray, fontSize: 13, lineHeight: 1.65, margin: '0 0 14px',
                    maxWidth: 820 }}>
          Which opportunity, which zone, and why — from P2, the canonical
          optimiser that produced the headline call above.
        </p>

        <P2Panel p2={p2} />

        {p2 && (
          <div className="panel" style={{ padding: 15, marginTop: 12 }}>
            <SectionTitle>OPPORTUNITY HORIZON</SectionTitle>
            <HorizonTimeline horizon={p2.horizon} pitResetIndex={p2.pit_reset_index} />
          </div>
        )}

        {/* Everything below this line comes from /decision (P1), not /p2. It was
            previously unlabelled and sat under the same "Strategy horizon"
            heading as the P2 panel, so its per-lap ATTACK/HOLD flag read as the
            current recommendation. */}
        <div style={{ borderTop: `1px solid ${C.panelBorder}`, marginTop: 26,
                      paddingTop: 14 }}>
          <div style={{ color: C.gray, fontSize: 11.5, fontWeight: 700,
                        letterSpacing: '0.06em' }}>
            LEGACY P1 TRACE — historical per-lap diagnostic, not the canonical
            strategic recommendation
          </div>
          <div style={{ color: C.dim, fontSize: 12, lineHeight: 1.6, marginTop: 5,
                        maxWidth: 820 }}>
            The per-lap flag in this section disagrees with canonical P2 on 961 of
            3,298 audited decision points (29%). Read it as a record of what the
            older per-lap solver did, not as what to do now. The white line is
            that solver's threshold: the quality of chance it treated as worth
            taking on a lap. Click any dot for the numbers behind it.
          </div>
        </div>

        <div style={{ display: 'grid', gridTemplateColumns: 'minmax(0,1fr) 310px',
                      gap: 20, marginTop: 16 }}>
          <div className="panel" style={{ padding: 12 }}>
            <svg ref={ref} style={{ width: '100%', height: 'auto', display: 'block' }} />
          </div>
          <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
            <div className="panel" style={{ padding: 15 }}>
              <SectionTitle>LEGACY P1 MODEL</SectionTitle>
              {/* A 30px green number under the bare title "MODEL" read as
                  confidence in the headline call. It is the P1 row's own
                  confidence; named and desaturated so it cannot. */}
              <div style={{ color: C.gray, fontSize: 11.5 }}>P1 trace confidence</div>
              <div className="num" style={{ fontSize: 30, fontWeight: 800, color: C.gray }}>
                {(((d?.call?.confidence ?? d?.laps?.[0]?.confidence ?? 0) as number) * 100).toFixed(0)}%
              </div>
              <div style={{ color: C.gray, fontSize: 12.5, marginTop: 6, lineHeight: 1.6 }}>
                Decision model: {d?.metadata?.decision_model ?? 'Core DP'}<br />
                Physics: {d?.metadata?.physics ?? 'cached longitudinal simulation'}<br />
                Opponent state: {d?.metadata?.opponent_state ?? 'inferred'}<br />
                Pass model: {d?.metadata?.pass_model ?? 'synthetic placeholder logistic'}<br />
                Tyre model: {d?.metadata?.tyre_calibration ?? 'synthetic'} (wear and
                temperature are modelled; tyre life is observed age)<br />
                Wetness: {d?.metadata?.wetness ? 'inferred' : 'n/a'}<br />
                Pit context: {d?.call?.pit_source ?? d?.laps?.[0]?.pit_source ?? 'unknown'}
              </div>
            </div>
            <div className="panel" style={{ padding: 15 }}>
              <SectionTitle>ZONES ON THIS CIRCUIT</SectionTitle>
              <table className="num" style={{ width: '100%', fontSize: 12, color: C.gray }}>
                <tbody>
                  {(d?.zone_models ?? []).map((z: any) => (
                    <tr key={z.name}>
                      <td style={{ padding: '3px 0' }}>zone {z.name}</td>
                      <td style={{ textAlign: 'right' }}>severity {z.severity}</td>
                      <td style={{ textAlign: 'right', color: C.white }}>
                        {z.dv_per_mj.toFixed(1)} m/s per MJ</td>
                    </tr>
                  ))}
                </tbody>
              </table>
            </div>
            {sel && (
              <div className="panel" style={{ padding: 15 }}>
                <SectionTitle>LEGACY P1 — LAP {sel.lap}</SectionTitle>
                <table className="num" style={{ width: '100%', fontSize: 12.5, color: C.gray }}>
                  <tbody>
                    <tr><td>opportunity q</td><td style={{ textAlign: 'right', color: C.white }}>{sel.q.toFixed(3)}</td></tr>
                    <tr><td>threshold τ</td><td style={{ textAlign: 'right', color: C.white }}>{sel.tau.toFixed(3)}</td></tr>
                    <tr><td>value attack</td><td style={{ textAlign: 'right', color: C.white }}>
                      {sel.value_attack == null
                        ? `not affordable (${(sel.value_attack_hypothetical ?? 0).toFixed(3)} if it were)`
                        : sel.value_attack.toFixed(3)}</td></tr>
                    <tr><td>value hold</td><td style={{ textAlign: 'right', color: C.white }}>{sel.value_wait.toFixed(3)}</td></tr>
                    <tr><td>gap</td><td style={{ textAlign: 'right', color: C.white }}>{sel.gap_s.toFixed(3)} s</td></tr>
                    <tr><td>delta v</td><td style={{ textAlign: 'right', color: C.white }}>{sel.predicted_delta_v_mps.toFixed(2)} m/s</td></tr>
                    <tr><td>your usable</td><td style={{ textAlign: 'right', color: C.amber }}>{sel.own_mj.toFixed(2)} MJ</td></tr>
                    <tr><td>rival deployable</td><td style={{ textAlign: 'right', color: C.red }}>{sel.rival_mj.toFixed(2)} MJ</td></tr>
                    <tr><td>best zone</td><td style={{ textAlign: 'right', color: C.white }}>{sel.zone}</td></tr>
                    <tr><td>gap source</td><td style={{ textAlign: 'right', color: C.gray }}>{sel.gap_source}</td></tr>
                    {/* Every value below is rendered exactly as the backend
                        returned it -- no equation in this file. tyre life is AGE
                        IN LAPS and wear is modelled separately, so they are shown
                        as two rows and never combined into one. */}
                    {sel.own_tyre && (
                      <>
                        <tr><td>tyre</td><td style={{ textAlign: 'right', color: C.white }}>
                          {sel.own_tyre.compound} · {sel.own_tyre.tyre_life_laps} laps old</td></tr>
                        <tr><td>modelled wear</td><td style={{ textAlign: 'right', color: C.amber }}>
                          {(sel.own_tyre.wear_fraction * 100).toFixed(1)}%</td></tr>
                        <tr><td>modelled tyre temp</td><td style={{ textAlign: 'right', color: C.gray }}>
                          {sel.own_tyre.estimated_temp_c.toFixed(0)} °C</td></tr>
                      </>
                    )}
                    {sel.attack_wear_continuation_penalty != null && (
                      <tr><td>attack wear cost</td><td style={{ textAlign: 'right', color: C.white }}>
                        {sel.attack_wear_continuation_penalty.toFixed(4)}
                        {sel.pit_resets_next_lap ? ' (stop next lap)' : ''}</td></tr>
                    )}
                    {sel.pit_context && (
                      <tr><td>laps to pit</td><td style={{ textAlign: 'right', color: C.white }}>
                        {sel.pit_context.laps_to_pit_mean == null
                          ? `unknown (${sel.pit_context.source})`
                          : `${sel.pit_context.laps_to_pit_mean.toFixed(1)} (${sel.pit_context.source})`}</td></tr>
                    )}
                    {sel.environment && (
                      <>
                        <tr><td>air density ρ</td><td style={{ textAlign: 'right', color: C.gray }}>
                          {sel.environment.rho.toFixed(3)} kg/m³</td></tr>
                        <tr><td>track temp</td><td style={{ textAlign: 'right', color: C.gray }}>
                          {sel.environment.track_temp_c == null ? '—'
                            : `${sel.environment.track_temp_c.toFixed(0)} °C`}</td></tr>
                        <tr><td>weather source</td><td style={{ textAlign: 'right', color: C.gray }}>
                          {sel.weather_source}</td></tr>
                      </>
                    )}
                  </tbody>
                </table>
                {/* The rule is the DP's value comparison, not q against tau.
                    Saying "q clears the threshold" described the heuristic the
                    API used before it delegated to the core solver, and the two
                    disagree: the DP can attack with q below tau, and holds
                    whenever the attack is unaffordable whatever q says.

                    It was also rendered as bare prose in green -- 'ATTACK —
                    V(attack) exceeds V(hold)' -- which is how the canonical call
                    is styled at the top of this view, so a P1 row that disagreed
                    with P2 (961 of 3,298 decision points, 29%) read as the
                    recommendation. Same information, named for its layer and
                    desaturated. */}
                <div style={{ marginTop: 10, fontSize: 12, color: C.gray,
                              lineHeight: 1.6 }}>
                  legacy P1 flag for lap {sel.lap}:{' '}
                  {sel.attack
                    ? 'ATTACK — V(attack) exceeded V(hold) in the per-lap trace'
                    : sel.attack_affordable === false
                      ? 'HOLD — not enough usable energy to fund an attack'
                      : 'HOLD — V(hold) is at least V(attack)'}
                  <br />Diagnostic only — the canonical recommendation is the P2
                  headline at the top of this view.
                </div>
              </div>
            )}
          </div>
        </div>
      </div>
    </div>
  );
}
