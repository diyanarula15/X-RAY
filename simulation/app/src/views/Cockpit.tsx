import { useEffect, useState } from 'react';
import { api } from '../lib/api';
import type { P2Decision, P3Status } from '../lib/api';
import { CandidateActionsTable } from '../components/CandidateActionsTable';
import { NextBestAction } from '../components/NextBestAction';
import { C } from '../lib/theme';

/**
 * Cockpit — the first thing a race engineer sees. One question: what should
 * we do, right now. Every field below is read verbatim from `/decision` and
 * `/p2`; no threshold, ranking or expected-value arithmetic lives here (see
 * `tests/test_frontend_p4.py`, which mirrors `test_frontend_p2.py`'s bans).
 * Validation tables live on the Evidence tab, not here — the first viewport
 * answers "attack or hold", not "how well-validated is the estimator".
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

  useEffect(() => {
    api.decision(raceId, car, rival).then(setD).catch(() => setD(null));
    api.p2(raceId, car, rival).then(setP2).catch(() => setP2(null));
    api.p3Status().then(setP3).catch(() => setP3(null));
  }, [raceId, car, rival]);

  // P1's per-lap row for "now" -- the current call if one exists, otherwise
  // the first row, exactly the fallback `Strategy.tsx` already uses.
  const row = d?.call ?? d?.laps?.[0] ?? null;

  if (!d && !p2) {
    return <div style={{ padding: 34, color: C.dim }}>No decision trace for this pair.</div>;
  }

  const attack = p2 ? p2.decision === 'ATTACK' : !!row?.attack;
  const synthetic = p2 ? p2.pass_model_calibration !== 'empirical' : true;
  const chosenLabel = p2
    ? (attack ? `ATTACK(${p2.zone}, ${p2.deployment_budget_mj.toFixed(3)} MJ)` : 'HOLD')
    : '';

  return (
    <div style={{ padding: '26px 34px', height: '100%', overflow: 'auto' }}>
      <div style={{ display: 'flex', alignItems: 'baseline', gap: 16, flexWrap: 'wrap' }}>
        <div className="num" style={{ fontSize: 46, fontWeight: 800,
                                      color: attack ? C.green : C.amber }}>
          {attack ? 'ATTACK' : 'HOLD'}{attack && p2?.zone ? ` — ZONE ${p2.zone}` : ''}
        </div>
        {row?.confidence != null && (
          <Badge text={`confidence ${pct(row.confidence, 0)}`}
                 color={row.confidence > 0.3 ? C.amber : C.gray} />
        )}
        {p3?.energy_inference && (
          <Badge text={`ENERGY: ${p3.energy_inference.status}`} color={C.amber} />
        )}
      </div>

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
          <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                        fontWeight: 700, marginBottom: 9 }}>ENERGY</div>
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
          <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                        fontWeight: 700, marginBottom: 9 }}>PREDICTED</div>
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
          <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                        fontWeight: 700, marginBottom: 9 }}>TYRES &amp; PIT</div>
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
          <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                        fontWeight: 700, marginBottom: 9 }}>CANDIDATE ACTIONS</div>
          <CandidateActionsTable rows={p2.candidate_actions} chosenLabel={chosenLabel} />
          <div style={{ marginTop: 9 }}>
            <NextBestAction nb={p2.next_best_action} decisionMargin={p2.decision_margin} />
          </div>
        </div>
      )}
    </div>
  );
}
