import type { P3Status, P3Entry, P3Replay } from '../lib/api';
import { C } from '../lib/theme';

/**
 * P3 evidence and model status — DISPLAY ONLY.
 *
 * Every status word, verdict and metric below is a backend string or number,
 * read verbatim from `/api/p3/status`. This file contains no threshold, no
 * pass/fail rule and no derived status: `tests/test_frontend_p3.py` greps it to
 * keep that true. Colour is the one thing chosen here, and it is chosen from the
 * backend's own status string rather than from any comparison of numbers.
 *
 * The negative result is the point of the panel. X-RAY's inferred energy loses
 * to a fixed-energy baseline on two of three held-out targets, and that is
 * rendered in the same weight and position as everything else — not in a
 * tooltip, not behind a disclosure arrow, not below the fold of a metric table.
 */

const STATUS_COLOUR: Record<string, string> = {
  REGULATION: C.white, PHYSICS: C.green, EMPIRICAL: C.green,
  CALIBRATED: C.green, INFERRED: C.amber, SYNTHETIC: C.amber,
  HEURISTIC: C.amber, DISABLED: C.gray, UNAVAILABLE: C.gray,
  RESEARCH_ONLY: C.red,
};
const VERDICT_COLOUR: Record<string, string> = {
  PASSED: C.green, MIXED: C.amber, NOT_ATTEMPTED: C.gray,
  REFUSED: C.amber, NOT_DEMONSTRATED: C.red, NO_ROBUST_IMPROVEMENT: C.red,
};
const words = (s: string) => s.replace(/_/g, ' ');
const mps = (x: number | null | undefined, d = 3) =>
  x == null ? '—' : `${x.toFixed(d)} m/s`;
const pctOf = (x: number | null | undefined, d = 1) =>
  x == null ? '—' : `${(x * 100).toFixed(d)}%`;
const fp = (s: string | null | undefined) => (s ? s.slice(0, 12) : '—');

function Section({ title, children }: { title: string; children: any }) {
  return (
    <div className="panel" style={{ padding: 15 }}>
      <div style={{ color: C.gray, fontSize: 10.5, letterSpacing: '0.09em',
                    fontWeight: 700, marginBottom: 9 }}>{title}</div>
      {children}
    </div>
  );
}

function Badge({ text, color }: { text: string; color: string }) {
  return (
    <span style={{ color, border: `1px solid ${color}`, borderRadius: 3,
                   padding: '2px 6px', fontSize: 10.5, fontWeight: 700,
                   letterSpacing: '0.04em', whiteSpace: 'nowrap' }}>{text}</span>
  );
}

/** The energy headline. Both halves, always, in one block. */
export function EnergyStatusPanel({ p3 }: { p3: P3Status | null }) {
  if (!p3) {
    return (
      <Section title="ENERGY INFERENCE STATUS">
        <div style={{ color: C.gray, fontSize: 12.5 }}>
          Status unavailable. Treat every energy number on this page as unverified.
        </div>
      </Section>
    );
  }
  const e = p3.energy_inference;
  const real = e.real_validation;
  return (
    <Section title="RIVAL ENERGY — WHAT THIS NUMBER IS">
      <div style={{ display: 'flex', gap: 7, flexWrap: 'wrap', marginBottom: 10 }}>
        <Badge text={e.status} color={STATUS_COLOUR[e.status] ?? C.white} />
        <Badge text={`REAL VALIDATION: ${words(real.result)}`}
               color={VERDICT_COLOUR[real.result] ?? C.gray} />
        <Badge text={`IDENTIFIABILITY: ${e.identifiability.status}`} color={C.amber} />
      </div>
      <table className="num" style={{ width: '100%', fontSize: 12.5 }}>
        <tbody>
          <tr><td style={{ color: C.gray, paddingRight: 10 }}>source</td>
              <td style={{ textAlign: 'right' }}>X-RAY canonical inference</td></tr>
          <tr><td style={{ color: C.gray }}>true battery energy</td>
              <td style={{ textAlign: 'right', color: C.red }}>NOT AVAILABLE</td></tr>
          <tr><td style={{ color: C.gray }}>real held-out validation</td>
              <td style={{ textAlign: 'right',
                           color: VERDICT_COLOUR[real.result] ?? C.gray }}>
                {words(real.result)}</td></tr>
        </tbody>
      </table>
      <div style={{ color: C.gray, fontSize: 11.5, marginTop: 9, lineHeight: 1.55 }}>
        {e.real_ground_truth.reason}. {e.identifiability.reason}.
      </div>
      <div style={{ marginTop: 10, padding: '8px 10px', borderRadius: 4,
                    background: 'rgba(225,6,0,0.14)', color: C.red,
                    fontSize: 11.5, fontWeight: 700, lineHeight: 1.5 }}>
        {e.headline}
      </div>
    </Section>
  );
}

/** Direct synthetic evidence, and the real held-out comparison, kept apart. */
export function ValidationPanel({ p3 }: { p3: P3Status | null }) {
  if (!p3) return null;
  const e = p3.energy_inference;
  const syn = e.synthetic_validation, real = e.real_validation;
  const sm = syn.metrics ?? {}, rm = real.metrics ?? {};
  const targets = Object.keys(rm.xray_minus_fixed_mae ?? {});
  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12 }}>
      <Section title="DIRECT SYNTHETIC VALIDATION — AGAINST HIDDEN SIMULATOR TRUTH">
        <div style={{ marginBottom: 9 }}>
          <Badge text={words(syn.result)} color={VERDICT_COLOUR[syn.result] ?? C.gray} />
        </div>
        <table className="num" style={{ width: '100%', fontSize: 12.5 }}>
          <tbody>
            <tr><td style={{ color: C.gray, paddingRight: 10 }}>deployment lap MAPE</td>
                <td style={{ textAlign: 'right', color: C.green }}>
                  {sm.deployment_lap_mape_pct == null ? '—'
                    : `${sm.deployment_lap_mape_pct.toFixed(2)}%`}</td></tr>
            <tr><td style={{ color: C.gray }}>deployment bias</td>
                <td style={{ textAlign: 'right' }}>
                  {sm.deployment_lap_bias_pct == null ? '—'
                    : `${sm.deployment_lap_bias_pct.toFixed(2)}%`}</td></tr>
            <tr><td style={{ color: C.gray }}>deployment band containment</td>
                <td style={{ textAlign: 'right', color: C.green }}>
                  {pctOf(sm.deployment_band_containment)}</td></tr>
            <tr><td style={{ color: C.gray }}>SOC mean MAE</td>
                <td style={{ textAlign: 'right' }}>
                  {sm.soc_mean_mae_mj == null ? '—'
                    : `${sm.soc_mean_mae_mj.toFixed(3)} MJ`}</td></tr>
            <tr><td style={{ color: C.gray }}>SOC band containment</td>
                <td style={{ textAlign: 'right', color: C.red }}>
                  {pctOf(sm.soc_band_containment)}
                  {sm.soc_band_containment_range
                    ? ` (${sm.soc_band_containment_range
                        .map((x: number) => `${(x * 100).toFixed(1)}%`).join('–')})`
                    : ''}</td></tr>
            <tr><td style={{ color: C.gray }}>CdA identifiability</td>
                <td style={{ textAlign: 'right', color: C.red }}>
                  {sm.cda_identifiability == null ? '—'
                    : sm.cda_identifiability.toFixed(3)}</td></tr>
            <tr><td style={{ color: C.gray }}>CdA interval width</td>
                <td style={{ textAlign: 'right', color: C.red }}>
                  {sm.cda_width_pct_of_truth == null ? '—'
                    : `${sm.cda_width_pct_of_truth.toFixed(1)}% of truth`}</td></tr>
            <tr><td style={{ color: C.gray }}>artifact</td>
                <td style={{ textAlign: 'right', color: C.dim }}>
                  {fp(syn.fingerprint)}</td></tr>
          </tbody>
        </table>
        <ul style={{ color: C.gray, fontSize: 11.5, lineHeight: 1.6,
                     margin: '9px 0 0', paddingLeft: 17 }}>
          {(syn.notes ?? []).map((n) => <li key={n}>{n}</li>)}
        </ul>
      </Section>

      <Section title="INDIRECT REAL VALIDATION — HELD-OUT FUTURE TELEMETRY">
        <div style={{ marginBottom: 9 }}>
          <Badge text={words(real.result)} color={VERDICT_COLOUR[real.result] ?? C.gray} />
        </div>
        <table className="num" style={{ width: '100%', fontSize: 12.5 }}>
          <tbody>
            <tr><td style={{ color: C.gray, paddingRight: 10 }}>held-out races</td>
                <td style={{ textAlign: 'right' }}>{rm.n_races ?? '—'}</td></tr>
            <tr><td style={{ color: C.gray }}>scored examples</td>
                <td style={{ textAlign: 'right' }}>
                  {rm.n_examples == null ? '—' : rm.n_examples.toLocaleString()}</td></tr>
            <tr><td style={{ color: C.gray }}>artifact</td>
                <td style={{ textAlign: 'right', color: C.dim }}>
                  {fp(real.fingerprint)}</td></tr>
          </tbody>
        </table>
        <div style={{ overflowX: 'auto', marginTop: 10 }}>
          <table className="num" style={{ width: '100%', fontSize: 11.5,
                                          borderCollapse: 'collapse' }}>
            <thead>
              <tr style={{ color: C.gray }}>
                <th style={{ textAlign: 'left' }}>target</th>
                <th style={{ textAlign: 'right' }}>X-RAY MAE</th>
                <th style={{ textAlign: 'right' }}>fixed-E MAE</th>
                <th style={{ textAlign: 'right' }}>neutral-E MAE</th>
                <th style={{ textAlign: 'right' }}>X-RAY − fixed</th>
                <th style={{ textAlign: 'right' }}>X-RAY − neutral</th>
              </tr>
            </thead>
            <tbody>
              {targets.map((t) => {
                const agg = (rm.aggregate_mae ?? {})[t] ?? {};
                const vf = (rm.xray_minus_fixed_mae ?? {})[t];
                const vn = (rm.xray_minus_neutral_mae ?? {})[t];
                return (
                  <tr key={t} style={{ color: C.gray }}>
                    <td style={{ textAlign: 'left', color: C.white }}>{words(t)}</td>
                    <td style={{ textAlign: 'right' }}>{mps(agg.xray, 2)}</td>
                    <td style={{ textAlign: 'right' }}>{mps(agg.fixed_energy, 2)}</td>
                    <td style={{ textAlign: 'right' }}>{mps(agg.energy_neutral, 2)}</td>
                    <td style={{ textAlign: 'right',
                                 color: vf == null ? C.gray : vf < 0 ? C.green : C.red }}>
                      {vf == null ? '—' : `${vf > 0 ? '+' : ''}${vf.toFixed(3)}`}</td>
                    <td style={{ textAlign: 'right',
                                 color: vn == null ? C.gray : vn < 0 ? C.green : C.red }}>
                      {vn == null ? '—' : `${vn > 0 ? '+' : ''}${vn.toFixed(3)}`}</td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
        <div style={{ color: C.gray, fontSize: 11, marginTop: 7, lineHeight: 1.55 }}>
          MAE in m/s, lower is better — so a POSITIVE difference is X-RAY losing.
          Baselines: fixed-E holds energy at the training-race mean; neutral-E
          assumes no stored deployable energy at all.
        </div>
        <div style={{ marginTop: 10, padding: '8px 10px', borderRadius: 4,
                      background: 'rgba(225,6,0,0.14)', color: C.red,
                      fontSize: 11.5, fontWeight: 700, lineHeight: 1.5 }}>
          CURRENT INFERRED ENERGY DOES NOT ADD ROBUST HELD-OUT PREDICTIVE VALUE
        </div>
        <ul style={{ color: C.gray, fontSize: 11.5, lineHeight: 1.6,
                     margin: '9px 0 0', paddingLeft: 17 }}>
          {(real.notes ?? []).map((n) => <li key={n}>{n}</li>)}
        </ul>
      </Section>
    </div>
  );
}

/** Every model a decision depends on, with the backend's own status word. */
export function ModelStatusPanel({ p3 }: { p3: P3Status | null }) {
  if (!p3) return null;
  const rows: P3Entry[] = p3.registry.entries ?? [];
  return (
    <Section title="MODEL STATUS — WHAT EACH NUMBER IS">
      <div style={{ overflowX: 'auto' }}>
        <table className="num" style={{ width: '100%', fontSize: 11.5,
                                        borderCollapse: 'collapse' }}>
          <thead>
            <tr style={{ color: C.gray }}>
              <th style={{ textAlign: 'left' }}>model</th>
              <th style={{ textAlign: 'left' }}>status</th>
              <th style={{ textAlign: 'left' }}>in a real race</th>
              <th style={{ textAlign: 'left' }}>validation</th>
            </tr>
          </thead>
          <tbody>
            {rows.map((e) => (
              <tr key={e.component} style={{ borderTop: `1px solid ${C.grid}` }}>
                <td style={{ textAlign: 'left', color: C.white, padding: '5px 8px 5px 0' }}>
                  {e.name}
                  {e.reason && (
                    <div style={{ color: C.dim, fontSize: 10.5, lineHeight: 1.45,
                                  maxWidth: 420 }}>{e.reason}</div>
                  )}
                </td>
                <td style={{ textAlign: 'left', paddingRight: 8 }}>
                  <Badge text={words(e.status)}
                         color={STATUS_COLOUR[e.status] ?? C.white} /></td>
                <td style={{ textAlign: 'left', paddingRight: 8,
                             color: e.production ? C.white : C.dim }}>
                  {e.production ? 'yes' : 'no'}</td>
                <td style={{ textAlign: 'left',
                             color: VERDICT_COLOUR[e.validation?.result ?? ''] ?? C.gray }}>
                  {e.validation ? words(e.validation.result) : '—'}</td>
              </tr>
            ))}
          </tbody>
        </table>
      </div>
      <div style={{ color: C.gray, fontSize: 11.5, marginTop: 9, lineHeight: 1.55 }}>
        A status says what kind of number it is, never how good it is — INFERRED
        means "produced by inference", not "verified". Research-only models are
        real code with real tests that never execute on a real race; their metrics
        are not evidence about real-race inference.
      </div>
    </Section>
  );
}

/** The corpus's own limits. Facts about the data, not hedges about the model. */
export function DataQualityPanel({ p3 }: { p3: P3Status | null }) {
  if (!p3) return null;
  const items = p3.data_quality_limitations ?? [];
  if (!items.length) return null;
  return (
    <Section title="DATA-QUALITY LIMITATIONS">
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {items.map((t) => {
          const [head, ...rest] = t.split(' — ');
          return (
            <div key={t} style={{ padding: '7px 9px', borderRadius: 4,
                                  background: 'rgba(255,195,0,0.10)',
                                  borderLeft: `3px solid ${C.amber}` }}>
              <div style={{ color: C.amber, fontSize: 11.5, fontWeight: 700,
                            letterSpacing: '0.03em' }}>{head}</div>
              {rest.length > 0 && (
                <div style={{ color: C.gray, fontSize: 11.5, marginTop: 3,
                              lineHeight: 1.5 }}>{rest.join(' — ')}</div>
              )}
            </div>
          );
        })}
      </div>
    </Section>
  );
}

/** Off-policy replay. The label is permanent, not conditional. */
export function ReplayPanel({ replay }: { replay: P3Replay | null }) {
  if (!replay) return null;
  const rec = replay.p2_recommendation;
  const cars = Object.keys(replay.later_observable_outcome ?? {});
  return (
    <Section title="HISTORICAL REPLAY">
      <div style={{ padding: '8px 10px', borderRadius: 4, marginBottom: 10,
                    background: 'rgba(255,195,0,0.14)', color: C.amber,
                    fontSize: 11.5, fontWeight: 700, lineHeight: 1.5 }}>
        {replay.label}
      </div>
      <table className="num" style={{ width: '100%', fontSize: 12.5 }}>
        <tbody>
          <tr><td style={{ color: C.gray, paddingRight: 10 }}>cutoff</td>
              <td style={{ textAlign: 'right' }}>
                {replay.cutoff_time_s.toFixed(2)} s</td></tr>
          <tr><td style={{ color: C.gray }}>samples known at cutoff</td>
              <td style={{ textAlign: 'right' }}>
                {Object.entries(replay.information_at_cutoff?.n_samples ?? {})
                  .map(([k, v]) => `${k} ${v ?? '—'}`).join(' · ')}</td></tr>
          <tr><td style={{ color: C.gray }}>inferred energy at cutoff</td>
              <td style={{ textAlign: 'right', color: C.amber }}>
                {replay.inferred_energy_mj
                  ? Object.entries(replay.inferred_energy_mj)
                      .map(([k, v]) => `${k} ${v.toFixed(3)} MJ`).join(' · ')
                  : '—'}</td></tr>
          <tr><td style={{ color: C.gray }}>P2 recommendation</td>
              <td style={{ textAlign: 'right', color: C.white }}>
                {rec ? `${rec.decision}${rec.zone ? ` zone ${rec.zone}` : ''}` +
                       `${rec.decision === 'ATTACK'
                          ? ` at ${rec.deployment_budget_mj.toFixed(3)} MJ` : ''}`
                     : (replay.p2_error ?? '—')}</td></tr>
          {cars.map((d) => {
            const o = replay.later_observable_outcome[d];
            return (
              <tr key={d}><td style={{ color: C.gray }}>{d} observed after cutoff</td>
                <td style={{ textAlign: 'right' }}>
                  {o.n_samples} samples · max {mps(o.v_max_mps, 1)}
                  {o.position_at_cutoff != null && o.position_at_window_end != null
                    ? ` · P${o.position_at_cutoff} → P${o.position_at_window_end}`
                    : ''}</td></tr>
            );
          })}
          <tr><td style={{ color: C.gray }}>gap to rival, cutoff → window end</td>
              <td style={{ textAlign: 'right' }}>
                {replay.gap_to_rival_at_cutoff_s == null
                  ? '—' : `${replay.gap_to_rival_at_cutoff_s.toFixed(3)} s`}
                {' → '}
                {replay.gap_to_rival_at_window_end_s == null
                  ? '—' : `${replay.gap_to_rival_at_window_end_s.toFixed(3)} s`}</td></tr>
          <tr><td style={{ color: C.gray }}>what the driver actually did</td>
              <td style={{ textAlign: 'right', color: C.white }}>
                {replay.actual_action ?? 'unknown — no later position to compare'}</td></tr>
          <tr><td style={{ color: C.gray }}>matched the recommendation</td>
              <td style={{ textAlign: 'right',
                           color: replay.matches_recommendation == null ? C.gray
                             : replay.matches_recommendation ? C.green : C.amber }}>
                {replay.matches_recommendation == null ? 'unknown'
                  : replay.matches_recommendation ? 'yes' : 'no'}</td></tr>
          <tr><td style={{ color: C.gray }}>input fingerprint</td>
              <td style={{ textAlign: 'right', color: C.dim }}>
                {fp(replay.information_at_cutoff?.input_fingerprint)}</td></tr>
        </tbody>
      </table>
      <div style={{ color: C.gray, fontSize: 11.5, marginTop: 9, lineHeight: 1.55 }}>
        Later telemetry is used only for evaluation and never reaches the
        recommendation. The car did not execute this recommendation, so the
        observed outcome is the outcome of what the driver actually did — this
        cannot show what the recommendation would have achieved. "Matched" only
        says the driver's real action agrees with the call; a mismatch is not
        shown as a worse outcome, because no counterfactual outcome exists for
        the action not taken.
      </div>
      <div style={{ color: C.amber, fontSize: 11.5, marginTop: 7, fontWeight: 700 }}>
        {replay.energy_inference_status}
      </div>
    </Section>
  );
}
