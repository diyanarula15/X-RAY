import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { C } from '../lib/theme';

/**
 * A stylised open-wheel car built from primitives, at metre scale, then scaled
 * into the 1/12 world. No liveried or team-identifiable model: no licensing
 * question, and a clean silhouette reads better at projector distance anyway.
 * Forward is +X. Wheels spin with road speed.
 */
const WHEEL_R = 0.36;

function Wheel({ x, z, w, spin }: { x: number; z: number; w: number; spin: React.MutableRefObject<number> }) {
  const g = useRef<THREE.Group>(null);
  useFrame(() => { if (g.current) g.current.rotation.z = -spin.current; });
  return (
    <group position={[x, WHEEL_R, z]}>
      <group ref={g}>
        <mesh rotation={[Math.PI / 2, 0, 0]} castShadow>
          <cylinderGeometry args={[WHEEL_R, WHEEL_R, w, 24]} />
          <meshStandardMaterial color="#141416" roughness={0.92} metalness={0.05} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[WHEEL_R * 0.55, WHEEL_R * 0.55, w + 0.02, 16]} />
          <meshStandardMaterial color="#3A3A42" roughness={0.35} metalness={0.8} />
        </mesh>
        {/* a spoke so the spin is visible */}
        <mesh position={[0, WHEEL_R * 0.3, 0]}>
          <boxGeometry args={[0.06, WHEEL_R * 0.5, w + 0.03]} />
          <meshStandardMaterial color="#8A8A94" metalness={0.7} roughness={0.4} />
        </mesh>
      </group>
    </group>
  );
}

export function CarMesh({
  position, heading, bank = 0, accent, ghost = false, speed = 0,
}: {
  position: THREE.Vector3; heading: number; bank?: number;
  accent: string; ghost?: boolean; speed?: number;   // speed in m/s
}) {
  const g = useRef<THREE.Group>(null);
  const spin = useRef(0);
  useFrame((_, dt) => {
    if (!g.current) return;
    g.current.position.copy(position);
    g.current.rotation.set(0, heading, bank);
    spin.current += (speed / WHEEL_R) * dt;
  });

  const body = useMemo(() => new THREE.MeshStandardMaterial({
    color: ghost ? '#9AA0A6' : '#E4E4EA', metalness: 0.55, roughness: 0.32,
    transparent: ghost, opacity: ghost ? 0.35 : 1,
    blending: ghost ? THREE.AdditiveBlending : THREE.NormalBlending, depthWrite: !ghost,
  }), [ghost]);
  const dark = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#1B1B21', metalness: 0.3, roughness: 0.6,
    transparent: ghost, opacity: ghost ? 0.3 : 1,
  }), [ghost]);
  const acc = useMemo(() => new THREE.MeshStandardMaterial({
    color: accent, emissive: new THREE.Color(accent), emissiveIntensity: ghost ? 0.3 : 1.1,
    metalness: 0.4, roughness: 0.4, transparent: ghost, opacity: ghost ? 0.35 : 1,
  }), [accent, ghost]);

  const sh = !ghost;
  return (
    <group ref={g}>
      <group scale={1 / 12}>
        {/* floor */}
        <mesh position={[-0.2, 0.12, 0]} material={dark} castShadow={sh}>
          <boxGeometry args={[4.4, 0.06, 1.5]} />
        </mesh>
        {/* monocoque */}
        <mesh position={[0.5, 0.5, 0]} material={body} castShadow={sh}>
          <boxGeometry args={[2.6, 0.5, 0.62]} />
        </mesh>
        {/* nose */}
        <mesh position={[2.35, 0.42, 0]} rotation={[0, 0, -Math.PI / 2]} material={body} castShadow={sh}>
          <coneGeometry args={[0.24, 1.4, 12]} />
        </mesh>
        {/* sidepods, accent */}
        {[-0.62, 0.62].map((z) => (
          <mesh key={z} position={[-0.55, 0.38, z]} material={acc} castShadow={sh}>
            <boxGeometry args={[1.9, 0.42, 0.46]} />
          </mesh>
        ))}
        {/* engine cover + airbox */}
        <mesh position={[-1.25, 0.7, 0]} material={body} castShadow={sh}>
          <boxGeometry args={[1.8, 0.42, 0.42]} />
        </mesh>
        <mesh position={[-0.05, 0.98, 0]} material={dark}>
          <boxGeometry args={[0.45, 0.3, 0.34]} />
        </mesh>
        {/* halo + helmet */}
        <mesh position={[0.55, 0.9, 0]} rotation={[Math.PI / 2, 0, 0]} material={dark}>
          <torusGeometry args={[0.42, 0.035, 8, 20, Math.PI]} />
        </mesh>
        <mesh position={[0.45, 0.8, 0]} material={acc}>
          <sphereGeometry args={[0.15, 12, 10]} />
        </mesh>
        {/* front wing */}
        <mesh position={[2.75, 0.14, 0]} material={acc} castShadow={sh}>
          <boxGeometry args={[0.55, 0.05, 1.9]} />
        </mesh>
        {[-0.96, 0.96].map((z) => (
          <mesh key={z} position={[2.75, 0.27, z]} material={dark}>
            <boxGeometry args={[0.6, 0.3, 0.04]} />
          </mesh>
        ))}
        {/* rear wing */}
        <mesh position={[-2.6, 0.98, 0]} material={acc} castShadow={sh}>
          <boxGeometry args={[0.5, 0.06, 1.0]} />
        </mesh>
        {[-0.5, 0.5].map((z) => (
          <mesh key={z} position={[-2.6, 0.72, z]} material={dark}>
            <boxGeometry args={[0.55, 0.55, 0.04]} />
          </mesh>
        ))}
        <mesh position={[-2.5, 0.6, 0]} material={dark}>
          <boxGeometry args={[0.08, 0.6, 0.12]} />
        </mesh>
        {/* wheels */}
        <Wheel x={1.75} z={0.82} w={0.36} spin={spin} />
        <Wheel x={1.75} z={-0.82} w={0.36} spin={spin} />
        <Wheel x={-1.7} z={0.85} w={0.46} spin={spin} />
        <Wheel x={-1.7} z={-0.85} w={0.46} spin={spin} />
        {/* suspension arms */}
        {[[1.75, 0.82], [1.75, -0.82], [-1.7, 0.85], [-1.7, -0.85]].map(([x, z], i) => (
          <mesh key={i} position={[x, 0.4, z * 0.55]} rotation={[Math.PI / 2, 0, 0]} material={dark}>
            <cylinderGeometry args={[0.025, 0.025, Math.abs(z) * 0.9, 6]} />
          </mesh>
        ))}
      </group>
    </group>
  );
}

/** Energy trail: emissive colour driven by instantaneous energy flow. */
export function EnergyTrail({ history, lite = false }:
  { history: { p: THREE.Vector3; flow: number }[]; lite?: boolean }) {
  const geom = useMemo(() => new THREE.BufferGeometry(), []);
  const obj = useMemo(() => new THREE.Line(geom, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: lite ? 0.6 : 0.95,
    blending: THREE.AdditiveBlending, depthWrite: false,
  })), [geom, lite]);
  const frame = useRef(0);
  useFrame(() => {
    frame.current++;
    if (frame.current % 3 !== 0) return;
    const n = history.length;
    if (n < 2) return;
    const pos = new Float32Array(n * 3), col = new Float32Array(n * 3);
    const hot = new THREE.Color(C.amber), cold = new THREE.Color('#39C6E0');
    const dim = new THREE.Color('#3A3A44'), t = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const h = history[i];
      pos[i * 3] = h.p.x; pos[i * 3 + 1] = h.p.y + 0.03; pos[i * 3 + 2] = h.p.z;
      const fade = i / n;
      if (h.flow > 0.02) t.copy(dim).lerp(hot, Math.min(h.flow, 1));
      else if (h.flow < -0.02) t.copy(dim).lerp(cold, Math.min(-h.flow, 1));
      else t.copy(dim);
      t.multiplyScalar(0.25 + 0.75 * fade);
      col[i * 3] = t.r; col[i * 3 + 1] = t.g; col[i * 3 + 2] = t.b;
    }
    geom.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    geom.setAttribute('color', new THREE.BufferAttribute(col, 3));
    geom.computeBoundingSphere();
  });
  return <primitive object={obj} />;
}
