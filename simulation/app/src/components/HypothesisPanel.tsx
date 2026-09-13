import type { Situation } from '../lib/api';
import { SectionTitle } from './Readouts';
import { C } from '../lib/theme';

/**
 * "What if the driver had ATTACKED here?" — a USER-SUPPLIED hypothesis, held
 * against X-RAY's call for the same moment.
 *
 * This replaces, and is deliberately not, the old `ScenarioWalkthrough`
 * "follow — next matching" / "disobey — next diverging" buttons. Those walked
 * the race looking for a point where the driver's action did or did not match
 * the recommendation, reading `matches_recommendation` — a verdict built on an
 * action INFERRED FROM A POSITION DELTA. A driver who attacked and failed was
 * recorded as having held, and a car promoted by the pit stop of the car ahead
 * was recorded as having attacked; 852 rows carried that verdict before it was
 * removed. Nothing here reads `actual_action`, `inferred_action_from_position`
 * or `matches_recommendation`, and no control in this panel claims to know what
 * the driver did.
 *
 * What it CAN say is what the solver valued, because those numbers are already
 * on the row. What it must not say is what would have happened: the race never
 * branched, so `counterfactual_status` is printed verbatim rather than resolved.
 */

const num = (x: number | null | undefined, d = 4) =>
  x == null ? '—' : x.toFixed(d);

type Pick = 'ATTACK' | 'HOLD';

/** The solver's value for a hypothesised action, or null if the payload does
 *  not carry it.
 *
 *  `value_action` is the value of the action P2 CHOSE, not of attacking —
 *  reading it as "V(attack)" would print the value of holding under an ATTACK
 *  heading every time P2 held. So: the chosen branch comes from `value_action`,
 *  holding always from `value_hold`, and the remaining case (you suppose an
 *  attack where P2 held) only from `next_best_action`, and only when that
 *  runner-up is in fact an attack. Otherwise null, which renders as "not in
 *  this payload" rather than as a number that means something else. */
function valueFor(s: Situation, pick: Pick): { value: number | null; note: string } {
  if (s.recommendation === pick) {
    return { value: s.value_action, note: "the solver's chosen action" };
  }
  if (pick === 'HOLD') {
    return { value: s.value_hold, note: 'the alternative the solver scored' };
  }
  const nb = s.next_best_action;
  if (nb && nb.kind === 'ATTACK') {
    return { value: nb.value, note: `runner-up action${nb.zone ? `, zone ${nb.zone}` : ''}` };
  }
  return { value: null, note: 'not scored in this payload' };
}

/** `pick` is held by the view, not here: selecting a different row has to clear
 *  it, and state owned by this component would survive that. */
export function HypothesisPanel({ s, pick, onPick }: {
  s: Situation; pick: Pick | null; onPick: (p: Pick | null) => void;
}) {
  const btn = (p: Pick) => {
    const on = pick === p;
    return (
      <button key={p} onClick={() => onPick(on ? null : p)}
        style={{ flex: 1, padding: '7px 10px', fontSize: 12, borderRadius: 7,
                 fontWeight: 700,
                 border: `1px solid ${on ? C.amber : C.panelBorder}`,
                 background: on ? 'rgba(255,195,0,0.10)' : 'transparent',
                 color: on ? C.amber : C.gray }}>
        {p}
      </button>
    );
  };

  const chosen = pick ? valueFor(s, pick) : null;
  const agrees = pick != null && s.recommendation != null && pick === s.recommendation;
  const synthetic = s.pass_model_calibration !== 'empirical';

  return (
    <div className="panel" style={{ padding: 15 }}>
      <SectionTitle>YOUR HYPOTHESIS</SectionTitle>
      <div style={{ color: C.gray, fontSize: 12, lineHeight: 1.6, marginBottom: 9 }}>
        Suppose the driver had done this at lap {s.lap}. This is your
        supposition, not a record: no public channel carries the driver&apos;s
        choice, so nothing here is what they actually did.
      </div>
      <div style={{ display: 'flex', gap: 6 }}>{(['ATTACK', 'HOLD'] as Pick[]).map(btn)}</div>

      {pick == null ? (
        <div style={{ color: C.dim, fontSize: 12, lineHeight: 1.6, marginTop: 10 }}>
          Pick one to see it against X-RAY&apos;s call for this moment.
        </div>
      ) : (
        <>
          <table className="num" style={{ width: '100%', fontSize: 12.5, marginTop: 11 }}>
            <tbody>
              <tr><td style={{ color: C.gray, paddingRight: 10 }}>you supposed</td>
                  <td style={{ textAlign: 'right', color: C.amber, fontWeight: 700 }}>
                    {pick}</td></tr>
              <tr><td style={{ color: C.gray }}>X-RAY called (P2)</td>
                  <td style={{ textAlign: 'right', color: C.white, fontWeight: 700 }}>
                    {s.recommendation ?? '—'}
                    {s.recommendation === 'ATTACK' && s.recommended_zone
                      ? ` zone ${s.recommended_zone}` : ''}</td></tr>
              {/* A statement about your pick against the engine's call. NOT a
                  statement about the driver, and deliberately uncoloured: an
                  agreement is not a correct call and a disagreement is not a
                  mistake — neither was ever tested against an outcome. */}
              <tr><td style={{ color: C.gray }}>your pick vs the call</td>
                  <td style={{ textAlign: 'right', color: C.gray }}>
                    {s.recommendation == null ? 'no call for this point'
                      : agrees ? 'same action' : 'different action'}</td></tr>
              {s.recommendation === 'ATTACK' && s.deployment_budget_mj != null && (
                <tr><td style={{ color: C.gray }}>call&apos;s budget</td>
                    <td style={{ textAlign: 'right' }}>
                      {s.deployment_budget_mj.toFixed(3)} MJ</td></tr>
              )}
              <tr><td style={{ color: C.gray }}>value of your pick</td>
                  <td style={{ textAlign: 'right', color: C.white }}>
                    {num(chosen?.value)}</td></tr>
              <tr><td style={{ color: C.dim, fontSize: 11 }} colSpan={2}>
                    <span style={{ color: C.dim, fontSize: 11 }}>
                      — {chosen?.note}</span></td></tr>
              <tr><td style={{ color: C.gray }}>value of holding</td>
                  <td style={{ textAlign: 'right' }}>{num(s.value_hold)}</td></tr>
              <tr><td style={{ color: C.gray }}>decision margin</td>
                  <td style={{ textAlign: 'right' }}>
                    {num(s.decision_margin, 3)}
                    {s.decision_margin != null && s.decision_margin < 0.01
                      ? ' (a close call)' : ''}</td></tr>
              <tr><td style={{ color: C.gray }}>P(pass) at this point</td>
                  <td style={{ textAlign: 'right' }}>
                    {s.pass_probability == null
                      ? '—' : `${(s.pass_probability * 100).toFixed(1)}%`}</td></tr>
            </tbody>
          </table>

          {/* The pass model's coefficients are design anchors, not a fit to real
              race data, and every value above that involves an attack inherits
              them. Same chip as Cockpit so the two views state it identically. */}
          <div style={{ marginTop: 8, padding: '5px 8px', borderRadius: 4,
                        background: synthetic ? 'rgba(201,162,39,0.14)' : 'rgba(0,160,90,0.14)',
                        color: synthetic ? C.amber : C.green, fontSize: 10.5,
                        fontWeight: 700 }}>
            PASS MODEL: {(s.pass_model_calibration ?? 'synthetic').toUpperCase()}
            {synthetic ? ' — NOT EMPIRICALLY CALIBRATED' : ''}
          </div>

          <div style={{ color: C.dim, fontSize: 11.5, lineHeight: 1.6, marginTop: 9 }}>
            What this does not say: whether your supposition would have worked.
            Counterfactual: {s.counterfactual_status.replace(/_/g, ' ')}. The race
            never branched onto either action, so these are the values the solver
            assigned at the cutoff — not outcomes.
          </div>
        </>
      )}
    </div>
  );
}
