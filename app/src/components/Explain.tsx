import { AnimatePresence, motion } from 'framer-motion';
import { useEffect, useState } from 'react';
import { C } from '../lib/theme';
import type { View } from '../store/playback';

/** What each view is for, in one sentence, plus what to actually do in it. */
export const GUIDE: Record<View, { title: string; what: string; do1: string[] }> = {
  theatre: {
    title: 'Race theatre',
    what: 'A real 2026 race, replayed. The amber car is you; the red car is the rival '
      + 'whose battery you are trying to read. Neither car has ever published its energy '
      + 'state — the red bar is reconstructed from its speed alone.',
    do1: [
      'Press play (or Space) to run the battle. Arrow keys jump 5 s.',
      'Watch the red cloud beside the rival: each dot is one guess at how much energy '
        + 'they have left. Tight cloud = we can see clearly. Spread out = we cannot.',
      'The big number top-left is the real gap. Under 1.000 s it turns green — that is '
        + 'the Manual Override boundary.',
      'Try the track paint buttons: "deployment" paints where the car spent energy.',
    ],
  },
  observability: {
    title: 'Observability map',
    what: 'The circuit coloured by how much the estimator can learn at each point. '
      + 'This is the same fact as the breathing cloud, drawn on the ground.',
    do1: [
      'Green = the trace tells us about energy here. Red = it tells us nothing.',
      'Hover any stretch of track to see the score and, where we refuse, why.',
      'Switch to the drag layer: high-speed running informs the car’s aerodynamics, '
        + 'not its energy. The two maps are nearly opposites.',
    ],
  },
  rdd: {
    title: 'RDD explorer',
    what: 'The test you run to try to break us. There is no public energy channel for a '
      + 'real car, so we cannot check the estimates directly. Instead we test a rule the '
      + 'regulation makes: a car within 1.000 s of the one ahead gets extra deployment.',
    do1: [
      'Drag the cutoff slider. If our estimates are real, the jump appears at 1.000 s.',
      'Now hunt: try 0.7, 1.3, 1.8. Finding "effects" where the rules put none means we '
        + 'are fitting noise.',
      'Read the UNDERPOWERED panel. Our answer here is a null, and that panel says '
        + 'whether that means anything.',
    ],
  },
  decision: {
    title: 'Decision explorer',
    what: 'When to attack. The white line is the threshold: the quality of chance worth '
      + 'taking on that lap. Dots above it are worth taking, dots below are not.',
    do1: [
      'Click any dot to see the numbers behind it — your energy, theirs, the zone.',
      'The faint fan is the same question re-answered for 200 different guesses about '
        + 'the rival’s style. The robustness number is how many agree.',
    ],
  },
  fingerprint: {
    title: 'Opponent fingerprint',
    what: 'Each driver’s deployment style, estimated from the speed trace alone.',
    do1: [
      'The shaded ring is uncertainty. It tightens as the race supplies more laps.',
      'Scrub the race in Race theatre, come back, and watch the ring close.',
    ],
  },
  method: {
    title: 'Method & limits',
    what: 'What this system cannot do, stated before anything it can.',
    do1: ['Read it first if you are deciding whether to believe the rest.'],
  },
};

export function HelpButton({ view }: { view: View }) {
  const [open, setOpen] = useState(false);
  const g = GUIDE[view];
  return (
    <>
      <button onClick={() => setOpen(true)} title="What is this view?"
        style={{ width: 26, height: 26, borderRadius: 13, fontSize: 13,
          border: `1px solid ${C.panelBorder}`, background: 'transparent',
          color: C.gray, lineHeight: 1 }}>?</button>
      <AnimatePresence>
        {open && (
          <motion.div initial={{ opacity: 0 }} animate={{ opacity: 1 }} exit={{ opacity: 0 }}
            onClick={() => setOpen(false)}
            style={{ position: 'fixed', inset: 0, background: 'rgba(5,5,8,0.72)',
              zIndex: 100, display: 'grid', placeItems: 'center', padding: 30 }}>
            <motion.div initial={{ y: 14, scale: 0.98 }} animate={{ y: 0, scale: 1 }}
              onClick={(e) => e.stopPropagation()} className="panel"
              style={{ maxWidth: 620, padding: 26 }}>
              <div className="display" style={{ fontSize: 22, marginBottom: 10 }}>{g.title}</div>
              <p style={{ color: C.gray, fontSize: 13.5, lineHeight: 1.7, marginTop: 0 }}>
                {g.what}
              </p>
              <div style={{ color: C.amber, fontSize: 10.5, letterSpacing: '0.09em',
                fontWeight: 700, margin: '16px 0 8px' }}>TRY THIS</div>
              <ul style={{ color: C.gray, fontSize: 13, lineHeight: 1.75,
                margin: 0, paddingLeft: 18 }}>
                {g.do1.map((t, i) => <li key={i} style={{ marginBottom: 5 }}>{t}</li>)}
              </ul>
              <button onClick={() => setOpen(false)}
                style={{ marginTop: 20, padding: '8px 16px', borderRadius: 8,
                  border: `1px solid ${C.amber}`, background: 'transparent',
                  color: C.amber, fontSize: 12.5 }}>got it</button>
            </motion.div>
          </motion.div>
        )}
      </AnimatePresence>
    </>
  );
}

/** Shown once on first visit. The whole idea in four steps. */
export function Onboarding() {
  const [step, setStep] = useState<number | null>(null);
  useEffect(() => {
    if (!localStorage.getItem('xray-seen')) setStep(0);
  }, []);
  const steps = [
    {
      t: 'A car’s battery is invisible',
      b: 'Formula 1 publishes speed, and nothing about energy. No team, broadcaster or '
        + 'feed says how much electrical energy a rival has left — but that is what '
        + 'decides whether you can pass them.',
    },
    {
      t: 'We reconstruct it from speed alone',
      b: 'How hard a car accelerates, against how hard the air pushes back, says how '
        + 'much power it is using. Subtract what the engine can produce and the '
        + 'remainder is electrical. Do that all lap and you have their battery.',
    },
    {
      t: 'The red cloud is the uncertainty',
      b: 'Four hundred dots, each one guess at the rival’s remaining energy. They '
        + 'pull together where the speed trace is informative and spread apart where it '
        + 'is not. That is not decoration — it is the algorithm, drawn.',
    },
    {
      t: 'Then it tells you when to attack',
      b: 'Knowing their energy and yours gives a probability of the pass on every lap. '
        + 'Tested against a simulator where the true answer is known, it picks the right '
        + 'lap within one, 97% of the time. Against attacking blind: 0.49 expected value '
        + 'versus 0.02.',
    },
  ];
  if (step === null) return null;
  const s = steps[step];
  const done = () => { localStorage.setItem('xray-seen', '1'); setStep(null); };
  return (
    <div style={{ position: 'fixed', inset: 0, background: 'rgba(5,5,8,0.85)', zIndex: 200,
      display: 'grid', placeItems: 'center', padding: 30 }}>
      <motion.div key={step} initial={{ opacity: 0, y: 12 }} animate={{ opacity: 1, y: 0 }}
        className="panel" style={{ maxWidth: 560, padding: 30 }}>
        <div style={{ color: C.amber, fontSize: 10.5, letterSpacing: '0.1em',
          fontWeight: 700 }}>X-RAY · {step + 1} of {steps.length}</div>
        <div className="display" style={{ fontSize: 25, margin: '12px 0 12px' }}>{s.t}</div>
        <p style={{ color: C.gray, fontSize: 14, lineHeight: 1.75, margin: 0 }}>{s.b}</p>
        <div style={{ display: 'flex', gap: 10, marginTop: 24, alignItems: 'center' }}>
          <button onClick={() => (step + 1 < steps.length ? setStep(step + 1) : done())}
            style={{ padding: '9px 18px', borderRadius: 8, border: 'none',
              background: C.amber, color: C.bg, fontSize: 13, fontWeight: 700 }}>
            {step + 1 < steps.length ? 'next' : 'start'}
          </button>
          <button onClick={done}
            style={{ padding: '9px 14px', borderRadius: 8, background: 'transparent',
              border: `1px solid ${C.panelBorder}`, color: C.gray, fontSize: 12.5 }}>
            skip
          </button>
          <div style={{ marginLeft: 'auto', display: 'flex', gap: 6 }}>
            {steps.map((_, i) => (
              <div key={i} style={{ width: 7, height: 7, borderRadius: 4,
                background: i === step ? C.amber : C.panelBorder }} />
            ))}
          </div>
        </div>
      </motion.div>
    </div>
  );
}

/** Always-on key for the 3D scene: what every colour on screen means. */
export function SceneLegend() {
  const [open, setOpen] = useState(true);
  const rows = [
    { c: C.amber, t: 'your car, and the energy you have' },
    { c: C.red, t: 'the rival, and our belief about theirs' },
    { c: '#39C6E0', t: 'recovering energy under braking' },
    { c: C.green, t: 'within 1.000 s — Override eligible' },
  ];
  return (
    <div className="panel" style={{ padding: open ? '11px 13px' : '7px 11px' }}>
      <button onClick={() => setOpen(!open)}
        style={{ background: 'none', border: 'none', padding: 0, width: '100%',
          textAlign: 'left', color: C.gray, fontSize: 10, letterSpacing: '0.09em',
          fontWeight: 700 }}>
        {open ? '▾' : '▸'} KEY
      </button>
      {open && (
        <div style={{ marginTop: 9 }}>
          {rows.map((r) => (
            <div key={r.t} style={{ display: 'flex', gap: 8, alignItems: 'center',
              marginBottom: 5 }}>
              <span style={{ width: 9, height: 9, borderRadius: 5, background: r.c,
                flexShrink: 0 }} />
              <span style={{ color: C.gray, fontSize: 11.5 }}>{r.t}</span>
            </div>
          ))}
          <div style={{ borderTop: `1px solid ${C.panelBorder}`, marginTop: 8,
            paddingTop: 8, color: C.dim, fontSize: 11, lineHeight: 1.55 }}>
            The red cloud is 400 separate guesses at the rival's remaining energy.
            Tight = we can see. Spread = we cannot.
          </div>
        </div>
      )}
    </div>
  );
}
