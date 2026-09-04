import { create } from 'zustand';

/**
 * ONE clock. The 3D scene, every 2D overlay and every chart subscribe to this.
 * Nothing keeps its own timer: desync between the 3D view and the charts is the
 * single most demo-destroying bug available in this project.
 */
export type View = 'theatre' | 'observability' | 'rdd' | 'decision' | 'fingerprint' | 'method';

type State = {
  raceId: string | null; view: View;
  frame: number; nFrames: number; playing: boolean; speed: number;
  lite: boolean;
  paintMode: 'neutral' | 'deployment' | 'observability' | 'refusal';
  camera: 'chase' | 'duel' | 'tactical';
  subject: string | null; rival: string | null;
  demo: boolean; demoBeat: number; showCloud: boolean;
  setRace: (id: string) => void; setView: (v: View) => void;
  setFrame: (f: number) => void; setNFrames: (n: number) => void;
  play: () => void; pause: () => void; toggle: () => void;
  setSpeed: (s: number) => void; setLite: (b: boolean) => void;
  setPaint: (p: State['paintMode']) => void; setCamera: (c: State['camera']) => void;
  setCars: (subject: string, rival: string) => void;
  startDemo: () => void; stopDemo: () => void; setBeat: (b: number) => void;
  setShowCloud: (b: boolean) => void;
};

export const usePlayback = create<State>((set) => ({
  raceId: null, view: 'theatre', frame: 0, nFrames: 0, playing: false, speed: 1,
  lite: new URLSearchParams(location.search).get('lite') === '1',
  paintMode: 'neutral', camera: 'chase', subject: null, rival: null,
  demo: false, demoBeat: -1, showCloud: true,
  setRace: (raceId) => set({ raceId, frame: 0 }),
  setView: (view) => set({ view }),
  setFrame: (frame) => set({ frame }),
  setNFrames: (nFrames) => set({ nFrames }),
  play: () => set({ playing: true }),
  pause: () => set({ playing: false }),
  toggle: () => set((s) => ({ playing: !s.playing })),
  setSpeed: (speed) => set({ speed }),
  setLite: (lite) => set({ lite }),
  setPaint: (paintMode) => set({ paintMode }),
  setCamera: (camera) => set({ camera }),
  setCars: (subject, rival) => set({ subject, rival }),
  startDemo: () => set({ demo: true, demoBeat: 0, playing: true, frame: 0 }),
  stopDemo: () => set({ demo: false, demoBeat: -1 }),
  setBeat: (demoBeat) => set({ demoBeat }),
  setShowCloud: (showCloud) => set({ showCloud }),
}));
