import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { C } from '../lib/theme';

/** Built in code as extruded geometry: no liveried model, no licensing risk,
 *  and a clean silhouette reads better at projector distance than detail. */
function carShape() {
  const s = new THREE.Shape();
  s.moveTo(-1.9, 0); s.lineTo(-1.55, 0.30); s.lineTo(-0.5, 0.34);
  s.lineTo(-0.15, 0.62); s.lineTo(0.45, 0.60); s.lineTo(0.95, 0.30);
  s.lineTo(2.05, 0.26); s.lineTo(2.15, 0);
  s.lineTo(-1.9, 0);
  return s;
}

export function CarMesh({
  position, heading, bank = 0, accent, ghost = false,
}: {
  position: THREE.Vector3; heading: number; bank?: number;
  accent: string; ghost?: boolean;
}) {
  const g = useRef<THREE.Group>(null);
  const geom = useMemo(() => {
    const e = new THREE.ExtrudeGeometry(carShape(), {
      depth: 0.72, bevelEnabled: true, bevelSize: 0.05, bevelThickness: 0.04,
      bevelSegments: 1, steps: 1,
    });
    e.center();
    e.scale(0.117, 0.117, 0.117);  // 4 shape-units -> 5.6 m at 1/12 world scale
    return e;
  }, []);
  useFrame(() => {
    if (!g.current) return;
    g.current.position.copy(position);
    g.current.rotation.set(0, heading, bank);
  });
  return (
    <group ref={g}>
      <mesh geometry={geom} castShadow={!ghost}>
        <meshStandardMaterial
          color={ghost ? C.gray : '#E8E8EE'} metalness={0.35}
          roughness={0.42} transparent={ghost} opacity={ghost ? 0.35 : 1}
          emissive={new THREE.Color(accent)} emissiveIntensity={ghost ? 0.15 : 0.35}
          blending={ghost ? THREE.AdditiveBlending : THREE.NormalBlending}
        />
      </mesh>
      {/* front and rear wing accents carry the identity colour */}
      {[-0.21, 0.23].map((x, i) => (
        <mesh key={i} position={[x, 0.02, 0]}>
          <boxGeometry args={[0.035, 0.018, 0.22]} />
          <meshStandardMaterial color={accent} emissive={new THREE.Color(accent)}
            emissiveIntensity={ghost ? 0.3 : 1.4} transparent={ghost}
            opacity={ghost ? 0.35 : 1} />
        </mesh>
      ))}
    </group>
  );
}

/**
 * Energy trail. Emissive colour is driven by instantaneous energy flow, so a
 * judge watching a car burn hot amber down the straight and go cold cyan under
 * braking understands the premise without narration.
 */
export function EnergyTrail({
  history, lite = false,
}: { history: { p: THREE.Vector3; flow: number }[]; lite?: boolean }) {
  const geom = useMemo(() => new THREE.BufferGeometry(), []);
  const obj = useMemo(() => new THREE.Line(geom, new THREE.LineBasicMaterial({
    vertexColors: true, transparent: true, opacity: lite ? 0.6 : 0.95,
    blending: THREE.AdditiveBlending, depthWrite: false,
  })), [geom, lite]);
  const frame = useRef(0);
  useFrame(() => {
    frame.current++;
    if (frame.current % 3 !== 0) return;       // rebuild at most every 3rd frame
    const n = history.length;
    if (n < 2) return;
    const pos = new Float32Array(n * 3);
    const col = new Float32Array(n * 3);
    const hot = new THREE.Color(C.amber), cold = new THREE.Color('#39C6E0');
    const dim = new THREE.Color('#3A3A44');
    const t = new THREE.Color();
    for (let i = 0; i < n; i++) {
      const h = history[i];
      pos[i * 3] = h.p.x; pos[i * 3 + 1] = h.p.y + 0.05; pos[i * 3 + 2] = h.p.z;
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
