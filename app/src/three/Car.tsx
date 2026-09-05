import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { C } from '../lib/theme';

/**
 * A simple, smooth open-wheel silhouette.
 *
 * The body is one extruded side-profile rather than a pile of boxes: a handful
 * of polygons, a clean wedge, and no attempt at realism. It reads as a model of
 * a car, which is the honest signal, without looking like it was assembled out
 * of bricks. No livery and no team identity, so no licensing question.
 *
 * Forward is +X. Wheels turn at road speed.
 */
const WHEEL_R = 0.36;

/** Side profile, in metres, nose to the right. */
function bodyProfile() {
  const p = new THREE.Shape();
  p.moveTo(-2.75, 0.10);          // rear floor
  p.lineTo(-2.75, 0.62);          // rear wing post
  p.lineTo(-2.30, 0.66);
  p.quadraticCurveTo(-1.60, 0.92, -0.85, 0.86);   // engine cover
  p.quadraticCurveTo(-0.45, 0.84, -0.30, 0.72);   // airbox shoulder
  p.lineTo(0.28, 0.70);                            // cockpit rim
  p.quadraticCurveTo(0.95, 0.66, 1.35, 0.48);      // nose shoulder
  p.quadraticCurveTo(2.30, 0.34, 2.95, 0.26);      // nose tip
  p.lineTo(2.95, 0.12);
  p.lineTo(2.30, 0.10);
  p.lineTo(-2.75, 0.10);
  return p;
}

function Wheel({ x, z, w, r, spin, ghost }: {
  x: number; z: number; w: number; r: number;
  spin: React.MutableRefObject<number>; ghost: boolean;
}) {
  const g = useRef<THREE.Group>(null);
  useFrame(() => { if (g.current) g.current.rotation.z = -spin.current; });
  return (
    <group position={[x, r, z]}>
      <group ref={g}>
        <mesh rotation={[Math.PI / 2, 0, 0]}>
          <cylinderGeometry args={[r, r, w, 16]} />
          <meshStandardMaterial color="#17171C" roughness={0.95} metalness={0}
            transparent={ghost} opacity={ghost ? 0.35 : 1} />
        </mesh>
        <mesh rotation={[Math.PI / 2, 0, 0]} position={[0, 0, 0]}>
          <cylinderGeometry args={[r * 0.5, r * 0.5, w + 0.015, 12]} />
          <meshStandardMaterial color="#5C5C68" roughness={0.5} metalness={0.6}
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

  const body = useMemo(() => {
    const geo = new THREE.ExtrudeGeometry(bodyProfile(), {
      depth: 0.78, bevelEnabled: true, bevelSize: 0.05, bevelThickness: 0.05,
      bevelSegments: 2, steps: 1, curveSegments: 6,
    });
    geo.rotateY(0);
    geo.translate(0, 0, -0.39);
    return geo;
  }, []);

  const shell = useMemo(() => new THREE.MeshStandardMaterial({
    color: ghost ? '#8A8A94' : '#DCDCE4', roughness: 0.55, metalness: 0.15,
    transparent: ghost, opacity: ghost ? 0.32 : 1,
  }), [ghost]);
  const dark = useMemo(() => new THREE.MeshStandardMaterial({
    color: '#22222A', roughness: 0.8, metalness: 0.1,
    transparent: ghost, opacity: ghost ? 0.3 : 1,
  }), [ghost]);
  const acc = useMemo(() => new THREE.MeshStandardMaterial({
    color: accent, emissive: new THREE.Color(accent),
    emissiveIntensity: ghost ? 0.25 : 0.55, roughness: 0.5, metalness: 0.1,
    transparent: ghost, opacity: ghost ? 0.35 : 1,
  }), [accent, ghost]);

  return (
    <group ref={g}>
      <group scale={1 / 12}>
        <mesh geometry={body} material={shell} />
        {/* engine cover stripe carries the identity colour along the spine */}
        <mesh position={[-1.15, 0.9, 0]} material={acc}>
          <boxGeometry args={[2.4, 0.06, 0.30]} />
        </mesh>
        {/* cockpit opening and halo */}
        <mesh position={[0.15, 0.72, 0]} material={dark}>
          <boxGeometry args={[0.9, 0.07, 0.44]} />
        </mesh>
        <mesh position={[0.42, 0.85, 0]} rotation={[Math.PI / 2, 0, 0]} material={dark}>
          <torusGeometry args={[0.34, 0.03, 6, 14, Math.PI]} />
        </mesh>
        {/* sidepod inlets */}
        {[-0.5, 0.5].map((z) => (
          <mesh key={z} position={[-0.35, 0.42, z * 0.86]} material={dark}>
            <boxGeometry args={[1.7, 0.34, 0.10]} />
          </mesh>
        ))}
        {/* front wing */}
        <mesh position={[2.92, 0.11, 0]} material={acc}>
          <boxGeometry args={[0.62, 0.05, 1.95]} />
        </mesh>
        {[-0.97, 0.97].map((z) => (
          <mesh key={z} position={[2.92, 0.26, z]} material={dark}>
            <boxGeometry args={[0.62, 0.3, 0.05]} />
          </mesh>
        ))}
        {/* rear wing */}
        <mesh position={[-2.78, 1.0, 0]} material={acc}>
          <boxGeometry args={[0.52, 0.055, 1.1]} />
        </mesh>
        {[-0.54, 0.54].map((z) => (
          <mesh key={z} position={[-2.78, 0.8, z]} material={dark}>
            <boxGeometry args={[0.52, 0.42, 0.05]} />
          </mesh>
        ))}
        <Wheel x={1.85} z={0.85} w={0.4} r={WHEEL_R} spin={spin} ghost={ghost} />
        <Wheel x={1.85} z={-0.85} w={0.4} r={WHEEL_R} spin={spin} ghost={ghost} />
        <Wheel x={-1.85} z={0.9} w={0.52} r={WHEEL_R * 1.08} spin={spin} ghost={ghost} />
        <Wheel x={-1.85} z={-0.9} w={0.52} r={WHEEL_R * 1.08} spin={spin} ghost={ghost} />
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
