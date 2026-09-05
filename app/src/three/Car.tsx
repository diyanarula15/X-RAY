import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { C } from '../lib/theme';

/**
 * Deliberately low-poly and flat-shaded. This is a simulation of a car, not an
 * attempt at one: chunky boxes, eight-sided wheels, visible facets, no bevels
 * and no smoothing. Reading as "model" rather than "photograph" is the honest
 * signal here -- the same restraint the statistical views get. It also has no
 * livery and no team identity, so there is no licensing question.
 *
 * Forward is +X. Wheels spin with road speed.
 */
const WHEEL_R = 0.38;
const WHEEL_SEG = 8;      // octagonal: you can see it turn

function Wheel({ x, z, w, spin, ghost }: {
  x: number; z: number; w: number;
  spin: React.MutableRefObject<number>; ghost: boolean;
}) {
  const g = useRef<THREE.Group>(null);
  useFrame(() => { if (g.current) g.current.rotation.z = -spin.current; });
  return (
    <group position={[x, WHEEL_R, z]}>
      <group ref={g}>
        <mesh rotation={[Math.PI / 2, 0, 0]} castShadow={!ghost}>
          <cylinderGeometry args={[WHEEL_R, WHEEL_R, w, WHEEL_SEG]} />
          <meshStandardMaterial color="#1C1C22" roughness={1} metalness={0}
            flatShading transparent={ghost} opacity={ghost ? 0.35 : 1} />
        </mesh>
        {/* one flat face so rotation is unmistakable */}
        <mesh position={[0, WHEEL_R * 0.55, 0]}>
          <boxGeometry args={[0.16, WHEEL_R * 0.7, w + 0.02]} />
          <meshStandardMaterial color="#6E6E7A" flatShading roughness={1}
            transparent={ghost} opacity={ghost ? 0.35 : 1} />
        </mesh>
      </group>
    </group>
  );
}

export function CarMesh({
  position, heading, bank = 0, accent, ghost = false, speed = 0,
}: {
  position: THREE.Vector3; heading: number; bank?: number;
  accent: string; ghost?: boolean; speed?: number;
}) {
  const g = useRef<THREE.Group>(null);
  const spin = useRef(0);
  useFrame((_, dt) => {
    if (!g.current) return;
    g.current.position.copy(position);
    g.current.rotation.set(0, heading, bank);
    spin.current += (speed / WHEEL_R) * dt;
  });

  const shell = useMemo(() => new THREE.MeshStandardMaterial({
    color: ghost ? '#8A8A94' : '#D8D8E0', roughness: 1, metalness: 0,
    flatShading: true, transparent: ghost, opacity: ghost ? 0.32 : 1,
  }), [ghost]);
  const dark = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#25252E', roughness: 1, metalness: 0, flatShading: true,
    transparent: ghost, opacity: ghost ? 0.3 : 1,
  }), [ghost]);
  const acc = useMemo(() => new THREE.MeshStandardMaterial({
    color: accent, emissive: new THREE.Color(accent),
    emissiveIntensity: ghost ? 0.25 : 0.65, roughness: 1, metalness: 0,
    flatShading: true, transparent: ghost, opacity: ghost ? 0.35 : 1,
  }), [accent, ghost]);

  const sh = !ghost;
  return (
    <group ref={g}>
      <group scale={1 / 12}>
        {/* floor slab */}
        <mesh position={[-0.1, 0.16, 0]} material={dark} castShadow={sh}>
          <boxGeometry args={[4.6, 0.12, 1.5]} />
        </mesh>
        {/* tub */}
        <mesh position={[0.35, 0.48, 0]} material={acc} castShadow={sh}>
          <boxGeometry args={[2.9, 0.5, 0.72]} />
        </mesh>
        {/* nose wedge: one box, no cone */}
        <mesh position={[2.2, 0.38, 0]} material={shell} castShadow={sh}>
          <boxGeometry args={[1.5, 0.3, 0.42]} />
        </mesh>
        {/* sidepods */}
        {[-0.66, 0.66].map((z) => (
          <mesh key={z} position={[-0.5, 0.42, z]} material={shell} castShadow={sh}>
            <boxGeometry args={[2.0, 0.44, 0.44]} />
          </mesh>
        ))}
        {/* engine cover */}
        <mesh position={[-1.5, 0.66, 0]} material={shell} castShadow={sh}>
          <boxGeometry args={[1.5, 0.4, 0.5]} />
        </mesh>
        {/* airbox */}
        <mesh position={[-0.35, 0.92, 0]} material={dark}>
          <boxGeometry args={[0.5, 0.36, 0.4]} />
        </mesh>
        {/* helmet: a low-poly ball, obviously faceted */}
        <mesh position={[0.5, 0.85, 0]} material={dark}>
          <sphereGeometry args={[0.19, 6, 4]} />
        </mesh>
        {/* front wing */}
        <mesh position={[2.9, 0.13, 0]} material={acc} castShadow={sh}>
          <boxGeometry args={[0.7, 0.09, 2.0]} />
        </mesh>
        {[-1.0, 1.0].map((z) => (
          <mesh key={z} position={[2.9, 0.3, z]} material={dark}>
            <boxGeometry args={[0.7, 0.34, 0.07]} />
          </mesh>
        ))}
        {/* rear wing */}
        <mesh position={[-2.65, 1.0, 0]} material={acc} castShadow={sh}>
          <boxGeometry args={[0.6, 0.1, 1.15]} />
        </mesh>
        {[-0.57, 0.57].map((z) => (
          <mesh key={z} position={[-2.65, 0.7, z]} material={dark}>
            <boxGeometry args={[0.6, 0.6, 0.07]} />
          </mesh>
        ))}
        <Wheel x={1.85} z={0.86} w={0.42} spin={spin} ghost={ghost} />
        <Wheel x={1.85} z={-0.86} w={0.42} spin={spin} ghost={ghost} />
        <Wheel x={-1.75} z={0.9} w={0.54} spin={spin} ghost={ghost} />
        <Wheel x={-1.75} z={-0.9} w={0.54} spin={spin} ghost={ghost} />
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
