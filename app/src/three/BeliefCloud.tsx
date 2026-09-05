import { useMemo, useRef } from 'react';
import { useFrame } from '@react-three/fiber';
import * as THREE from 'three';
import { C } from '../lib/theme';

/**
 * THE SIGNATURE VISUAL.
 *
 * This is not a bar with a shaded region. It is the particle filter itself:
 * every point is one particle's estimate of how much energy the rival can still
 * deploy, positioned on a vertical energy axis beside their car. When the trace
 * is informative the cloud contracts; through a slow corner it disperses; at a
 * resampling step the particles visibly collapse and redistribute.
 *
 * It is scientifically literal and it is the argument. If anything has to be
 * cut for performance, it is not this.
 */
export function BeliefCloud({
  particlesRef, target, height = 3.2, maxMJ = 4.0, count = 400,
  spread = 0.55, offset = 1.5,
}: {
  // A ref, not a prop value: the cloud changes every frame and passing it
  // through React would re-render the Canvas subtree at 60 Hz, which is exactly
  // what made playback stutter. The render loop reads it directly.
  particlesRef: React.MutableRefObject<number[]>;
  target: THREE.Vector3;        // where the rival car is (mutated in place)
  height?: number; maxMJ?: number; count?: number;
  spread?: number; offset?: number;
}) {
  const n = count;
  const points = useRef<THREE.Points>(null);
  const cur = useRef<Float32Array>(new Float32Array(n * 3));
  // Without this the spring below never starts: `cur[i] || target` treats a
  // legitimate 0.0 as "unset", the increment evaluates to zero, and every
  // particle stays pinned at the world origin forever.
  const seeded = useRef(false);
  const jitter = useMemo(
    () => Array.from({ length: n }, () => [
      (Math.random() - 0.5) * spread, (Math.random() - 0.5) * spread,
    ] as [number, number]), [n, spread]);

  const geom = useMemo(() => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 3), 3));
    return g;
  }, [n]);

  useFrame((_, dt) => {
    if (!points.current) return;
    const pos = geom.getAttribute('position') as THREE.BufferAttribute;
    const colr = geom.getAttribute('color') as THREE.BufferAttribute;
    const arr = pos.array as Float32Array;
    const carr = colr.array as Float32Array;
    // spring toward the new estimate: physical, not stepped
    const k = 1 - Math.exp(-dt * 9);
    const hot = new THREE.Color(C.amber), cold = new THREE.Color(C.red);
    const tmp = new THREE.Color();
    const particles = particlesRef.current;
    for (let i = 0; i < n; i++) {
      const mj = particles.length ? particles[i % particles.length] : 0;
      const ty = target.y + offset * 0.35 + (Math.min(mj, maxMJ) / maxMJ) * height;
      const tx = target.x + offset + jitter[i][0];
      const tz = target.z + jitter[i][1];
      if (!seeded.current) {
        cur.current[i * 3] = tx; cur.current[i * 3 + 1] = ty; cur.current[i * 3 + 2] = tz;
      } else {
        cur.current[i * 3] += (tx - cur.current[i * 3]) * k;
        cur.current[i * 3 + 1] += (ty - cur.current[i * 3 + 1]) * k;
        cur.current[i * 3 + 2] += (tz - cur.current[i * 3 + 2]) * k;
      }
      arr[i * 3] = cur.current[i * 3];
      arr[i * 3 + 1] = cur.current[i * 3 + 1];
      arr[i * 3 + 2] = cur.current[i * 3 + 2];
      // brighter where the cloud is dense — that is where the weight is
      const f = Math.min(mj / maxMJ, 1);
      tmp.copy(cold).lerp(hot, f * 0.55);
      carr[i * 3] = tmp.r; carr[i * 3 + 1] = tmp.g; carr[i * 3 + 2] = tmp.b;
    }
    seeded.current = true;
    pos.needsUpdate = true; colr.needsUpdate = true;
  });

  return (
    <points ref={points} geometry={geom}>
      <pointsMaterial size={0.055} vertexColors transparent opacity={0.95}
        sizeAttenuation depthWrite={false} blending={THREE.AdditiveBlending} />
    </points>
  );
}

/** The axis the cloud is read against: without it the cloud is just pretty. */
export function CloudAxis({ target, height = 3.2, offset = 1.5 }:
  { target: THREE.Vector3; height?: number; offset?: number }) {
  const pts = useMemo(() => {
    const p: number[] = [];
    const y0 = target.y + offset * 0.35;
    for (let i = 0; i <= 4; i++) {
      const y = y0 + (i / 4) * height;
      p.push(target.x + offset * 0.72, y, target.z, target.x + offset * 0.86, y, target.z);
    }
    p.push(target.x + offset * 0.79, y0, target.z,
           target.x + offset * 0.79, y0 + height, target.z);
    return new Float32Array(p);
  }, [target, height, offset]);
  return (
    <lineSegments>
      <bufferGeometry>
        <bufferAttribute attach="attributes-position" args={[pts, 3]} />
      </bufferGeometry>
      <lineBasicMaterial color={C.dim} transparent opacity={0.65} />
    </lineSegments>
  );
}
