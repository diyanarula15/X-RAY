import { useMemo } from 'react';
import * as THREE from 'three';
import type { Geometry } from '../lib/api';
import { C } from '../lib/theme';

const S = 1 / 12;

function frame(geo: Geometry, i: number) {
  const n = geo.s.length;
  const j = (i + 1) % n, k = (i - 1 + n) % n;
  const tx = geo.x[j] - geo.x[k], ty = geo.y[j] - geo.y[k];
  const L = Math.hypot(tx, ty) || 1;
  return {
    x: geo.x[i] * S, y: geo.y[i] * S,
    z: (geo.has_elevation ? geo.z[i] : 0) * S,
    nx: -ty / L, ny: tx / L,
  };
}

/**
 * Kerbs on the corners, run-off around the whole lap, a start line and marshal
 * posts. All positioned from the real centreline and the real curvature — the
 * kerbs land on the actual corners because `is_corner` came from the geometry,
 * not from a guess.
 */
export function TrackDressing({ geo, width = 20, lite = false }:
  { geo: Geometry; width?: number; lite?: boolean }) {
  const { kerbGeom, apron, startLine, posts } = useMemo(() => {
    const n = geo.s.length;
    const halfW = (width * S) / 2;

    // --- kerbs: alternating red/white blocks along the outside of each corner
    const kp: number[] = [], kc: number[] = [], ki: number[] = [];
    const red = new THREE.Color(C.red), white = new THREE.Color('#E8E8EE');
    let v = 0;
    for (let i = 0; i < n; i++) {
      if (!geo.is_corner[i]) continue;
      const a = frame(geo, i), b = frame(geo, (i + 1) % n);
      const col = (Math.floor(i / 2) % 2 === 0) ? red : white;
      for (const sgn of [1, -1]) {
        const w0 = halfW * sgn, w1 = (halfW + 1.6 * S) * sgn;
        const pts = [
          [a.x + a.nx * w0, a.z + 0.004, a.y + a.ny * w0],
          [a.x + a.nx * w1, a.z + 0.004, a.y + a.ny * w1],
          [b.x + b.nx * w0, b.z + 0.004, b.y + b.ny * w0],
          [b.x + b.nx * w1, b.z + 0.004, b.y + b.ny * w1],
        ];
        pts.forEach((p) => { kp.push(p[0], p[1], p[2]); kc.push(col.r, col.g, col.b); });
        ki.push(v, v + 1, v + 2, v + 1, v + 3, v + 2);
        v += 4;
      }
    }
    const kerbGeom = new THREE.BufferGeometry();
    kerbGeom.setAttribute('position', new THREE.BufferAttribute(new Float32Array(kp), 3));
    kerbGeom.setAttribute('color', new THREE.BufferAttribute(new Float32Array(kc), 3));
    kerbGeom.setIndex(ki);
    kerbGeom.computeVertexNormals();

    // --- run-off apron: a wider, darker band under the whole circuit
    const ap: number[] = [], ai: number[] = [];
    const aw = halfW + 4.5 * S;
    for (let i = 0; i < n; i++) {
      const a = frame(geo, i);
      ap.push(a.x + a.nx * aw, a.z - 0.02, a.y + a.ny * aw);
      ap.push(a.x - a.nx * aw, a.z - 0.02, a.y - a.ny * aw);
      const p = i * 2, q = ((i + 1) % n) * 2;
      ai.push(p, p + 1, q, p + 1, q + 1, q);
    }
    const apron = new THREE.BufferGeometry();
    apron.setAttribute('position', new THREE.BufferAttribute(new Float32Array(ap), 3));
    apron.setIndex(ai);
    apron.computeVertexNormals();

    // --- start / finish line
    const f0 = frame(geo, 0);
    const startLine = {
      pos: [f0.x, f0.z + 0.006, f0.y] as [number, number, number],
      rot: Math.atan2(f0.nx, f0.ny),
      w: halfW * 2,
    };

    // --- marshal posts every ~600 m, on the outside
    const posts: { p: [number, number, number] }[] = [];
    const step = Math.max(Math.round(600 / (geo.length / n)), 4);
    for (let i = 0; i < n; i += step) {
      const a = frame(geo, i);
      const w = halfW + 3.2 * S;
      posts.push({ p: [a.x + a.nx * w, a.z, a.y + a.ny * w] });
    }
    return { kerbGeom, apron, startLine, posts };
  }, [geo, width]);

  return (
    <>
      <mesh geometry={apron}>
        <meshStandardMaterial color="#17171E" roughness={0.98} metalness={0}
          side={THREE.DoubleSide} />
      </mesh>
      <mesh geometry={kerbGeom}>
        <meshStandardMaterial vertexColors side={THREE.DoubleSide}
          roughness={0.7} emissive={new THREE.Color('#201014')} emissiveIntensity={0.5} />
      </mesh>
      <mesh position={startLine.pos} rotation={[-Math.PI / 2, 0, startLine.rot]}>
        <planeGeometry args={[startLine.w, 0.55 * S * 12]} />
        <meshStandardMaterial color="#FFFFFF" emissive={new THREE.Color('#FFFFFF')}
          emissiveIntensity={0.35} roughness={0.6} />
      </mesh>
      {!lite && posts.map((m, i) => (
        <mesh key={i} position={[m.p[0], m.p[1] + 0.14, m.p[2]]}>
          <boxGeometry args={[0.04, 0.28, 0.04]} />
          <meshStandardMaterial color="#2A2A33" emissive={new THREE.Color(C.amber)}
            emissiveIntensity={0.35} />
        </mesh>
      ))}
    </>
  );
}

/** Ground plane, placed below the LOWEST point of the circuit.
 *
 *  A fixed y = -1.2 looks fine on a flat track and is badly wrong on a real one:
 *  Spa drops 100 m from Les Combes to Eau Rouge, so the whole bottom half of the
 *  circuit sat underneath the ground plane and simply was not drawn. It read as
 *  a camera or fog problem for a long time. */
export function Ground({ radius, geo }: { radius: number; geo: Geometry }) {
  const y = useMemo(() => {
    if (!geo.has_elevation) return -1.2;
    let lo = Infinity;
    for (const z of geo.z) lo = Math.min(lo, z);
    return (lo * S) - 3.0;
  }, [geo]);
  return (
    <mesh rotation={[-Math.PI / 2, 0, 0]} position={[0, y, 0]}>
      <circleGeometry args={[radius * 3.4, 64]} />
      <meshStandardMaterial color="#0D0D12" roughness={1} metalness={0} />
    </mesh>
  );
}
