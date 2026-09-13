import { create } from 'zustand';

/**
 * ONE clock, and it is RACE TIME in seconds of session time — not a frame index.
 * Both cars' traces carry the session clock, so sampling each at the same
 * raceTime is what makes them actually race each other with the true gap. The
 * 3D scene, every overlay and every chart subscribe to this. Nothing keeps its
 * own timer.
 */
/**
 * Five tabs, down from seven. `energy` held a radar whose axes were computed in
 * this frontend from invented constants (`mean(dep)/4`, `reserve_mean/1.4e6`),
 * which is the one thing the frontend guards exist to prevent; `strategy` folded
 * into `cockpit` because both answered "attack or hold"; `method` folded into
 * `evidence` because both answered "what do we actually know"; and `context`
 * became `observability` once the RDD explorer's cutoff slider was removed (it
 * refit against 130 MB of re-parsed JSON on every drag). `situations` is new and
 * is the surface for picking a real decision point out of the race.
 */
export type View = 'cockpit' | 'situations' | 'replay' | 'observability'
  | 'evidence';

type State = {
  raceId: string | null; view: View;
  raceTime: number; tRange: [number, number];
  playing: boolean; speed: number;
  lite: boolean;
  paintMode: 'neutral' | 'deployment' | 'observability' | 'refusal';
  camera: 'chase' | 'duel' | 'tactical';
  subject: string | null; rival: string | null;
  demo: boolean; demoBeat: number; showCloud: boolean; introTick: number;
  autoLite: boolean;
  setRace: (id: string) => void; setView: (v: View) => void;
  setTime: (t: number) => void; setRange: (r: [number, number]) => void;
  play: () => void; pause: () => void; toggle: () => void;
  setSpeed: (s: number) => void; setLite: (b: boolean) => void;
  setPaint: (p: State['paintMode']) => void; setCamera: (c: State['camera']) => void;
  setCars: (subject: string, rival: string) => void;
  startDemo: (t0?: number) => void; stopDemo: () => void; setBeat: (b: number) => void;
  setShowCloud: (b: boolean) => void;
};

export const usePlayback = create<State>((set) => ({
  raceId: null, view: 'cockpit', raceTime: 0, tRange: [0, 1],
  playing: false, speed: 2,
  lite: new URLSearchParams(location.search).get('lite') === '1',
  paintMode: 'neutral', camera: 'chase', subject: null, rival: null,
  demo: false, demoBeat: -1, showCloud: true, introTick: 0, autoLite: false,
  setRace: (raceId) => set({ raceId }),
  setView: (view) => set({ view }),
  setTime: (raceTime) => set({ raceTime }),
  setRange: (tRange) => set({ tRange, raceTime: tRange[0] }),
  play: () => set({ playing: true }),
  pause: () => set({ playing: false }),
  toggle: () => set((s) => ({ playing: !s.playing })),
  setSpeed: (speed) => set({ speed }),
  setLite: (lite) => set({ lite }),
  setPaint: (paintMode) => set({ paintMode }),
  setCamera: (camera) => set({ camera }),
  setCars: (subject, rival) => set({ subject, rival }),
  startDemo: (t0) => set((s) => ({ demo: true, demoBeat: 0, playing: true, speed: 1,
    raceTime: t0 ?? s.tRange[0], view: 'replay', introTick: s.introTick + 1 })),
  stopDemo: () => set({ demo: false, demoBeat: -1 }),
  setBeat: (demoBeat) => set({ demoBeat }),
  setShowCloud: (showCloud) => set({ showCloud }),
}));
