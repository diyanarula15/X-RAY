import type { P2Decision } from '../lib/api';
import { C } from '../lib/theme';

const num = (x: number | null | undefined, d = 4) =>
  x == null ? '—' : x.toFixed(d);

/** The runner-up action, extracted from `P2Panel.tsx` so `Cockpit.tsx` can
 *  show it without a second copy. Read verbatim, no ranking done here. */
export function NextBestAction({ nb, decisionMargin }: {
  nb: P2Decision['next_best_action']; decisionMargin: number;
}) {
  if (!nb) return null;
  return (
    <div style={{ color: C.gray, fontSize: 12.5, lineHeight: 1.6 }}>
      Next best: <span style={{ color: C.white }}>
        {nb.kind}{nb.zone ? ` zone ${nb.zone}` : ''}
        {nb.kind === 'ATTACK' ? ` at ${nb.deployment_budget_mj.toFixed(3)} MJ` : ''}
      </span> — value {num(nb.value)}
      {decisionMargin < 0.01 ? ' (a close call: the margin is small)' : ''}
    </div>
  );
}
