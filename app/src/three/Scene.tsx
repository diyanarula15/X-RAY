import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Bloom, EffectComposer, Noise, Vignette } from '@react-three/postprocessing';
import * as THREE from 'three';
import type { Car, Geometry } from '../lib/api';
import { C } from '../lib/theme';
import { usePlayback } from '../store/playback';
import { BeliefCloud, CloudAxis } from './BeliefCloud';
import { CarMesh, EnergyTrail } from './Car';
import { TrackEdges, TrackMesh, paintRibbon, useRibbon } from './TrackRibbon';

const SCALE = 1 / 12;

function posAt(geo: Geometry, s: number): { p: THREE.Vector3; heading: number; bank: number } {
  const n = geo.s.length;
  const f = ((s % geo.length) + geo.length) % geo.length / geo.length;
  const i = Math.min(Math.floor(f * n), n - 1);
  const j = (i + 1) % n, k = (i - 1 + n) % n;
  const p = new THREE.Vector3(geo.x[i] * SCALE,
    (geo.has_elevation ? geo.z[i] : 0) * SCALE + 0.12, geo.y[i] * SCALE);
  const heading = Math.atan2(geo.y[j] - geo.y[k], geo.x[j] - geo.x[k]);
  const bank = Math.min((geo.curvature?.[i] ?? 0) * 900, 1) * 0.14;
  return { p, heading: -heading, bank };
}

function Rig({ subject, rival, radius }: {
  subject: THREE.Vector3; rival: THREE.Vector3; radius: number;
}) {
  const { camera } = useThree();
  const mode = usePlayback((s) => s.camera);
  const demo = usePlayback((s) => s.demo);
  const target = useRef(new THREE.Vector3());
  const desired = useRef(new THREE.Vector3(0, radius, radius));
  const intro = useRef(demo ? 0 : 1);
  const fwd = useRef(new THREE.Vector3(1, 0, 0));

  useFrame((_, dt) => {
    intro.current = Math.min(intro.current + dt / 4, 1);
    const mid = subject.clone().lerp(rival, 0.5);

    if (mode === 'tactical') {
      // top-down, framed so the whole circuit fits: at a 42 deg vertical fov
      // the visible half-height is h*tan(21 deg), so h must be radius/tan(21).
      desired.current.set(0, radius * 2.75, 0.001);
      target.current.lerp(new THREE.Vector3(0, 0, 0), 0.06);
    } else if (mode === 'duel') {
      // perpendicular to the two cars' axis, so the gap is legible
      const axis = rival.clone().sub(subject).setY(0);
      if (axis.lengthSq() < 1e-6) axis.set(1, 0, 0);
      const perp = new THREE.Vector3(-axis.z, 0, axis.x).normalize().multiplyScalar(2.2);
      desired.current.copy(mid).add(perp).add(new THREE.Vector3(0, 0.85, 0));
      target.current.lerp(mid, 0.12);
    } else {
      // chase: 12 m back, 4 m up. One world unit is 12 m.
      const v = subject.clone().sub(target.current).setY(0);
      if (v.lengthSq() > 1e-7) fwd.current.lerp(v.normalize(), 0.08).normalize();
      desired.current.copy(subject)
        .sub(fwd.current.clone().multiplyScalar(1.15))
        .add(new THREE.Vector3(0, 0.42, 0));
      target.current.lerp(subject, 0.16);
    }

    // 4-second orbit on load, settling into whatever mode is selected
    if (intro.current < 1) {
      const a = intro.current * Math.PI * 2;
      const r = radius * (2.1 - 0.9 * intro.current);
      desired.current.set(Math.cos(a) * r, radius * (1.1 - 0.5 * intro.current),
        Math.sin(a) * r);
      target.current.set(0, 0, 0);
    }
    camera.position.lerp(desired.current,
      1 - Math.exp(-dt * (intro.current < 1 ? 5 : 3.4)));
    camera.lookAt(target.current);
  });
  return null;
}

export function Scene({ geo, subject, rival, obs }: {
  geo: Geometry; subject: Car | null; rival: Car | null; obs: any;
}) {
  const frame = usePlayback((s) => s.frame);
  const lite = usePlayback((s) => s.lite);
  const paint = usePlayback((s) => s.paintMode);
  const showCloud = usePlayback((s) => s.showCloud);
  const { geometry, n } = useRibbon(geo);
  // how big is this circuit in world units? every camera distance scales off it
  const radius = useMemo(() => {
    let r = 0;
    for (let i = 0; i < geo.x.length; i++) {
      r = Math.max(r, Math.hypot(geo.x[i], geo.y[i]));
    }
    return Math.max(r / 12, 5);
  }, [geo]);

  const fields = useMemo(() => {
    const dep = new Float32Array(n), ob = new Float32Array(n);
    const ref = new Uint8Array(n);
    if (subject) {
      const cnt = new Float32Array(n);
      subject.trace.s.forEach((s, i) => {
        const k = Math.min(Math.floor(((s % geo.length) / geo.length) * n), n - 1);
        dep[k] += Math.max(subject.trace.deploy_kw[i] ?? 0, 0) / 350;
        cnt[k] += 1;
      });
      for (let i = 0; i < n; i++) dep[i] = cnt[i] ? Math.min(dep[i] / cnt[i], 1) : 0;
    }
    if (obs?.s?.length) {
      for (let i = 0; i < n; i++) {
        const s = (i / n) * geo.length;
        const k = Math.min(Math.floor((s / geo.length) * obs.s.length), obs.s.length - 1);
        ob[i] = obs.deployment_info?.[k] ?? 0;
        ref[i] = (obs.deployment_info?.[k] ?? 0) < 0.05 ? 1 : 0;
      }
    }
    return { dep, ob, ref };
  }, [n, geo, subject, obs]);

  useEffect(() => {
    paintRibbon(geometry, n, paint, fields.dep, fields.ob, fields.ref);
  }, [geometry, n, paint, fields]);

  const sIdx = subject ? Math.min(frame, subject.trace.s.length - 1) : 0;
  const rIdx = rival ? Math.min(frame, rival.trace.s.length - 1) : 0;
  const sT = subject ? posAt(geo, subject.trace.s[sIdx]) : posAt(geo, 0);
  const rT = rival ? posAt(geo, rival.trace.s[rIdx]) : posAt(geo, 120);

  const trailS = useRef<{ p: THREE.Vector3; flow: number }[]>([]);
  const trailR = useRef<{ p: THREE.Vector3; flow: number }[]>([]);
  useEffect(() => {
    if (subject) {
      const flow = ((subject.trace.deploy_kw[sIdx] ?? 0) - (subject.trace.harvest_kw[sIdx] ?? 0)) / 350;
      trailS.current.push({ p: sT.p.clone(), flow });
      if (trailS.current.length > 60) trailS.current.shift();
    }
    if (rival) {
      const flow = ((rival.trace.deploy_kw[rIdx] ?? 0) - (rival.trace.harvest_kw[rIdx] ?? 0)) / 350;
      trailR.current.push({ p: rT.p.clone(), flow });
      if (trailR.current.length > 60) trailR.current.shift();
    }
  }, [frame]);

  const cloud = useMemo(() => {
    if (!rival?.cloud?.length) return [];
    const k = Math.min(Math.floor(rIdx / (rival.cloud_stride ?? 6)), rival.cloud.length - 1);
    return rival.cloud[Math.max(k, 0)] ?? [];
  }, [rival, rIdx]);

  return (
    <Canvas
      camera={{ position: [0, 120, 200], fov: 42, near: 0.05, far: 20000 }}
      gl={{ antialias: !lite, powerPreference: 'high-performance' }}
      dpr={lite ? 1 : [1, 1.75]}
      style={{ position: 'absolute', inset: 0 }}
    >
      <color attach="background" args={[C.bg]} />
      <fog attach="fog" args={[C.bg, radius * 0.9, radius * 7]} />
      <directionalLight position={[radius, radius * 1.2, radius * 0.6]}
        intensity={1.6} color="#FFE6C0" castShadow={!lite} />
      <hemisphereLight args={['#8FA8C8', '#141420', 0.55]} />
      <ambientLight intensity={0.35} />

      <TrackMesh geometry={geometry} />
      <TrackEdges geo={geo} />

      <CarMesh position={sT.p} heading={sT.heading} bank={sT.bank} accent={C.amber} />
      <CarMesh position={rT.p} heading={rT.heading} bank={rT.bank} accent={C.red} />

      {!lite && <EnergyTrail history={trailS.current} />}
      {!lite && <EnergyTrail history={trailR.current} />}

      {showCloud && (
        <>
          <BeliefCloud particles={cloud} target={rT.p} count={lite ? 120 : 400}
            height={1.15} spread={0.20} offset={0.42} />
          <CloudAxis target={rT.p} height={1.15} offset={0.42} />
        </>
      )}

      <Rig subject={sT.p} rival={rT.p} radius={radius} />

      {!lite && (
        <EffectComposer>
          <Bloom intensity={0.8} luminanceThreshold={0.65} resolutionScale={0.5} mipmapBlur />
          <Vignette offset={0.3} darkness={0.55} />
          <Noise opacity={0.025} />
        </EffectComposer>
      )}
    </Canvas>
  );
}
