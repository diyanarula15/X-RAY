import { useEffect, useMemo, useRef } from 'react';
import { Canvas, useFrame, useThree } from '@react-three/fiber';
import { Bloom, EffectComposer, Noise, Vignette, SMAA } from '@react-three/postprocessing';
import * as THREE from 'three';
import type { Car, Geometry } from '../lib/api';
import type { CarSample } from '../lib/carState';
import { C } from '../lib/theme';
import { usePlayback } from '../store/playback';
import { BeliefCloud, CloudAxis } from './BeliefCloud';
import { CarMesh, EnergyTrail } from './Car';
import { Ground, TrackDressing } from './Environment';
import { TrackEdges, TrackMesh, paintRibbon, useRibbon } from './TrackRibbon';

const S = 1 / 12;

/** Position and orientation on the real centreline, interpolated between
 *  grid points so the car glides rather than stepping. */
function poseAt(geo: Geometry, s: number) {
  const n = geo.s.length;
  const f = (((s % geo.length) + geo.length) % geo.length) / geo.length * n;
  const i = Math.floor(f) % n;
  const j = (i + 1) % n;
  const u = f - Math.floor(f);
  const x = (geo.x[i] + (geo.x[j] - geo.x[i]) * u) * S;
  const y = (geo.y[i] + (geo.y[j] - geo.y[i]) * u) * S;
  const zr = geo.has_elevation ? geo.z[i] + (geo.z[j] - geo.z[i]) * u : 0;
  const k = (i - 1 + n) % n;
  const heading = -Math.atan2(geo.y[j] - geo.y[k], geo.x[j] - geo.x[k]);
  const bank = Math.min((geo.curvature?.[i] ?? 0) * 700, 1) * 0.16;
  return { p: new THREE.Vector3(x, zr * S + 0.005, y), heading, bank };
}

function Rig({ subject, rival, radius, centre }: {
  subject: THREE.Vector3; rival: THREE.Vector3; radius: number; centre: THREE.Vector3;
}) {
  const { camera } = useThree();
  const mode = usePlayback((s) => s.camera);
  const introTick = usePlayback((s) => s.introTick);
  const target = useRef(new THREE.Vector3());
  const desired = useRef(new THREE.Vector3(0, radius, radius));
  const fwd = useRef(new THREE.Vector3(1, 0, 0));
  const intro = useRef(1);
  const lastTick = useRef(introTick);

  useFrame((_, dt) => {
    if (lastTick.current !== introTick) { lastTick.current = introTick; intro.current = 0; }
    intro.current = Math.min(intro.current + dt / 4, 1);
    const mid = subject.clone().lerp(rival, 0.5);

    if (mode === 'tactical') {
      // Top-down over the circuit's BOUNDING-BOX centre, not the origin: the
      // centreline is mean-centred, and for an elongated circuit the mean sits
      // well away from the middle of the bounds, which framed a third of Spa.
      desired.current.set(centre.x, radius * 2.9, centre.z + 0.001);
      target.current.lerp(centre, 0.08);
    } else if (mode === 'duel') {
      // perpendicular to the two cars' axis, pulled back as the gap opens so
      // both stay in frame
      const axis = rival.clone().sub(subject).setY(0);
      const d = Math.max(axis.length(), 0.3);
      if (axis.lengthSq() < 1e-6) axis.set(1, 0, 0);
      // pull back with the gap, but only so far: past a few car lengths there
      // is nothing to frame and the shot becomes empty track
      const back = Math.min(1.5 + d * 0.9, 4.2);
      const perp = new THREE.Vector3(-axis.z, 0, axis.x).normalize().multiplyScalar(back);
      desired.current.copy(mid).add(perp)
        .add(new THREE.Vector3(0, 0.45 + Math.min(d, 2.5) * 0.22, 0));
      target.current.lerp(mid, 0.12);
    } else {
      // chase: ~12 m back, ~4 m up, with lag on the heading so corners swing
      const v = subject.clone().sub(target.current).setY(0);
      if (v.lengthSq() > 1e-7) fwd.current.lerp(v.normalize(), 0.07).normalize();
      desired.current.copy(subject)
        .sub(fwd.current.clone().multiplyScalar(1.05))
        .add(new THREE.Vector3(0, 0.38, 0));
      target.current.lerp(subject.clone().add(fwd.current.clone().multiplyScalar(0.4)), 0.16);
    }

    if (intro.current < 1) {
      const a = intro.current * Math.PI * 2;
      const r = radius * (2.2 - 1.0 * intro.current);
      desired.current.set(Math.cos(a) * r, radius * (1.15 - 0.6 * intro.current),
        Math.sin(a) * r);
      target.current.lerp(new THREE.Vector3(0, 0, 0), 0.1);
    }
    camera.position.lerp(desired.current,
      1 - Math.exp(-dt * (intro.current < 1 ? 5 : 3.2)));
    camera.lookAt(target.current);
  });
  return null;
}

/** Trails are accumulated inside the canvas, on the render clock, so they are
 *  smooth regardless of how often React re-renders. */
function Trails({ sPose, rPose, sSample, rSample, lite }: any) {
  const a = useRef<{ p: THREE.Vector3; flow: number }[]>([]);
  const b = useRef<{ p: THREE.Vector3; flow: number }[]>([]);
  const acc = useRef(0);
  useFrame((_, dt) => {
    acc.current += dt;
    if (acc.current < 1 / 30) return;
    acc.current = 0;
    const push = (arr: any, pose: any, smp: any) => {
      if (!pose || !smp) return;
      arr.current.push({ p: pose.p.clone(),
        flow: ((smp.deploy ?? 0) - (smp.harvest ?? 0)) / 350 });
      if (arr.current.length > 70) arr.current.shift();
    };
    push(a, sPose, sSample); push(b, rPose, rSample);
  });
  return (
    <>
      <EnergyTrail history={a.current} lite={lite} />
      <EnergyTrail history={b.current} lite={lite} />
    </>
  );
}

export function Scene({ geo, subject, rival, sSample, rSample, obs }: {
  geo: Geometry; subject: Car | null; rival: Car | null;
  sSample: CarSample | null; rSample: CarSample | null; obs: any;
}) {
  const lite = usePlayback((s) => s.lite);
  const paint = usePlayback((s) => s.paintMode);
  const showCloud = usePlayback((s) => s.showCloud);
  const { geometry, n } = useRibbon(geo);

  const { radius, centre } = useMemo(() => {
    let x0 = Infinity, x1 = -Infinity, y0 = Infinity, y1 = -Infinity;
    for (let i = 0; i < geo.x.length; i++) {
      x0 = Math.min(x0, geo.x[i]); x1 = Math.max(x1, geo.x[i]);
      y0 = Math.min(y0, geo.y[i]); y1 = Math.max(y1, geo.y[i]);
    }
    const cx = (x0 + x1) / 2 / 12, cz = (y0 + y1) / 2 / 12;
    // half-diagonal of the bounds, in world units
    const r = Math.max(Math.hypot(x1 - x0, y1 - y0) / 2 / 12, 5);
    return { radius: r, centre: new THREE.Vector3(cx, 0, cz) };
  }, [geo]);

  const fields = useMemo(() => {
    const dep = new Float32Array(n), ob = new Float32Array(n), ref = new Uint8Array(n);
    if (subject) {
      const cnt = new Float32Array(n);
      subject.trace.s.forEach((s, i) => {
        const k = Math.min(Math.floor(((s % geo.length) / geo.length) * n), n - 1);
        dep[k] += Math.max(subject.trace.deploy_kw[i] ?? 0, 0) / 350; cnt[k] += 1;
      });
      // normalise against what this car actually did, not against the 350 kW
      // regulation cap: a real car averages a fraction of that, so scaling by
      // the cap paints the whole circuit black and shows nothing
      for (let i = 0; i < n; i++) dep[i] = cnt[i] ? dep[i] / cnt[i] : 0;
      let mx = 0;
      for (let i = 0; i < n; i++) mx = Math.max(mx, dep[i]);
      if (mx > 1e-6) for (let i = 0; i < n; i++) dep[i] = Math.min(dep[i] / mx, 1);
    }
    if (obs?.s?.length) {
      for (let i = 0; i < n; i++) {
        const k = Math.min(Math.floor((i / n) * obs.s.length), obs.s.length - 1);
        ob[i] = obs.deployment_info?.[k] ?? 0;
        ref[i] = (obs.deployment_info?.[k] ?? 0) < 0.05 ? 1 : 0;
      }
    }
    return { dep, ob, ref };
  }, [n, geo, subject, obs]);

  useEffect(() => { paintRibbon(geometry, n, paint, fields.dep, fields.ob, fields.ref); },
    [geometry, n, paint, fields]);

  const sPose = sSample ? poseAt(geo, sSample.s) : poseAt(geo, 0);
  const rPose = rSample ? poseAt(geo, rSample.s) : poseAt(geo, 60);

  const cloud = useMemo(() => {
    if (!rival?.cloud?.length || !rSample) return [];
    const k = Math.min(Math.floor(rSample.idx / (rival.cloud_stride ?? 4)),
      rival.cloud.length - 1);
    return rival.cloud[Math.max(k, 0)] ?? [];
  }, [rival, rSample?.idx]);

  return (
    <Canvas
      shadows={!lite}
      camera={{ position: [0, radius * 1.2, radius * 2], fov: 42, near: 0.05, far: 20000 }}
      gl={{ antialias: false, powerPreference: 'high-performance' }}
      dpr={lite ? 1 : [1, 1.75]}
      style={{ position: 'absolute', inset: 0 }}
    >
      <color attach="background" args={['#07070A']} />
      {/* Fog has to clear the tactical camera, which sits ~3 radii up: at
          radius*1.1 it erased everything but the ground directly beneath it and
          made the circuit look half-drawn. */}
      <fog attach="fog" args={['#07070A', radius * 3.2, radius * 14]} />

      <directionalLight
        position={[radius * 0.8, radius * 1.4, radius * 0.5]} intensity={3.0}
        color="#FFE2B8" castShadow={!lite}
        shadow-mapSize={[1024, 1024]} shadow-camera-far={radius * 6}
        shadow-camera-left={-radius} shadow-camera-right={radius}
        shadow-camera-top={radius} shadow-camera-bottom={-radius} />
      <hemisphereLight args={['#8FB2DC', '#1A1A24', 1.15]} />
      <ambientLight intensity={0.55} />
      {/* a low fill from the opposite side so the cars are not silhouettes */}
      <directionalLight position={[-radius, radius * 0.5, -radius * 0.6]}
        intensity={0.75} color="#9FB8D8" />

      <Ground radius={radius} geo={geo} />
      <TrackMesh geometry={geometry} />
      <TrackDressing geo={geo} lite={lite} />
      <TrackEdges geo={geo} />

      <CarMesh position={sPose.p} heading={sPose.heading} bank={sPose.bank}
        accent={C.amber} speed={sSample?.v ?? 0} ghost={sSample ? !sSample.onTrack : true} />
      <CarMesh position={rPose.p} heading={rPose.heading} bank={rPose.bank}
        accent={C.red} speed={rSample?.v ?? 0} ghost={rSample ? !rSample.onTrack : true} />

      {!lite && <Trails sPose={sPose} rPose={rPose} sSample={sSample}
        rSample={rSample} lite={lite} />}

      {showCloud && rSample?.onTrack && (
        <>
          <BeliefCloud particles={cloud} target={rPose.p} count={lite ? 120 : 400}
            height={1.15} spread={0.20} offset={0.42} />
          <CloudAxis target={rPose.p} height={1.15} offset={0.42} />
        </>
      )}

      <Rig subject={sPose.p} rival={rPose.p} radius={radius} centre={centre} />

      {!lite && (
        <EffectComposer>
          <SMAA />
          <Bloom intensity={0.75} luminanceThreshold={0.62} resolutionScale={0.5} mipmapBlur />
          <Vignette offset={0.3} darkness={0.55} />
          <Noise opacity={0.022} />
        </EffectComposer>
      )}
    </Canvas>
  );
}
