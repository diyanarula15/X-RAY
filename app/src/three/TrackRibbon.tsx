import { useMemo } from 'react';
import * as THREE from 'three';
import type { Geometry } from '../lib/api';
import { C } from '../lib/theme';

export type PaintMode = 'neutral' | 'deployment' | 'observability' | 'refusal';

const col = (hex: string) => new THREE.Color(hex);
const ASPHALT = col('#4A4A58');   // lifted off the grid colour: at tactical
                                  // zoom under a dark sky, #26262E reads black
const AMBER = col(C.amber);
const RED = col(C.red);
const GREEN = col(C.green);

/**
 * The ribbon is not decoration, it is the primary data surface, so it is a
 * custom BufferGeometry with per-vertex colours rather than a textured mesh.
 * Built once from the real centreline (including the real Z), cached, and
 * repainted by swapping the colour attribute — never rebuilt per frame.
 */
/** 20 m rather than a true 12-15 m: the ribbon is the primary data surface and
 *  the paint modes have to be readable from the tactical camera, where a true
 *  width is sub-pixel. A schematic widening, and the only one in the scene. */
export function useRibbon(geo: Geometry, width = 20) {
  return useMemo(() => {
    const n = geo.s.length;
    const pos = new Float32Array(n * 2 * 3);
    const idx: number[] = [];
    const scale = 1 / 12; // metres -> scene units, keeps the circuit a sane size

    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n, k = (i - 1 + n) % n;
      const tx = geo.x[j] - geo.x[k], ty = geo.y[j] - geo.y[k];
      const L = Math.hypot(tx, ty) || 1;
      const nx = -ty / L, ny = tx / L;             // left normal, on the surface
      const x = geo.x[i] * scale, y = geo.y[i] * scale;
      const z = (geo.has_elevation ? geo.z[i] : 0) * scale;
      const w = (width * scale) / 2;
      pos[i * 6 + 0] = x + nx * w; pos[i * 6 + 1] = z; pos[i * 6 + 2] = y + ny * w;
      pos[i * 6 + 3] = x - nx * w; pos[i * 6 + 4] = z; pos[i * 6 + 5] = y - ny * w;
      const a = i * 2, b = i * 2 + 1, c = ((i + 1) % n) * 2, d = ((i + 1) % n) * 2 + 1;
      idx.push(a, b, c, b, d, c);
    }
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(pos, 3));
    g.setAttribute('color', new THREE.BufferAttribute(new Float32Array(n * 2 * 3), 3));
    g.setIndex(idx);
    g.computeVertexNormals();
    return { geometry: g, n, scale };
  }, [geo]);
}

/** Repaint in place: write into the existing colour attribute, no rebuild. */
export function paintRibbon(
  g: THREE.BufferGeometry, n: number, mode: PaintMode,
  deployment: Float32Array | null, observability: Float32Array | null,
  refusal: Uint8Array | null,
) {
  const attr = g.getAttribute('color') as THREE.BufferAttribute;
  const arr = attr.array as Float32Array;
  const t = new THREE.Color();
  for (let i = 0; i < n; i++) {
    if (mode === 'deployment' && deployment) {
      const f = Math.min(Math.max(deployment[i], 0), 1);
      t.copy(ASPHALT).lerp(AMBER, f);
    } else if (mode === 'observability' && observability) {
      const f = Math.min(Math.max(observability[i], 0), 1);
      t.copy(RED).lerp(GREEN, f);
    } else if (mode === 'refusal' && refusal) {
      t.copy(refusal[i] ? RED : ASPHALT);
      if (refusal[i] && i % 6 < 3) t.multiplyScalar(0.35);   // hatch
    } else {
      t.copy(ASPHALT);
    }
    arr[i * 6] = t.r; arr[i * 6 + 1] = t.g; arr[i * 6 + 2] = t.b;
    arr[i * 6 + 3] = t.r; arr[i * 6 + 4] = t.g; arr[i * 6 + 5] = t.b;
  }
  attr.needsUpdate = true;
}

export function TrackMesh({ geometry }: { geometry: THREE.BufferGeometry }) {
  return (
    <mesh geometry={geometry} receiveShadow>
      <meshStandardMaterial
        vertexColors side={THREE.DoubleSide}
        roughness={0.88} metalness={0.04}
        emissive={new THREE.Color('#20202A')} emissiveIntensity={0.6}
      />
    </mesh>
  );
}

/** White edge lines. Cheap, and they make the ribbon read as a racetrack. */
export function TrackEdges({ geo, width = 20 }: { geo: Geometry; width?: number }) {
  const lines = useMemo(() => {
    const scale = 1 / 12, n = geo.s.length;
    const mk = (sign: number) => {
      const p: number[] = [];
      for (let i = 0; i < n; i++) {
        const j = (i + 1) % n, k = (i - 1 + n) % n;
        const tx = geo.x[j] - geo.x[k], ty = geo.y[j] - geo.y[k];
        const L = Math.hypot(tx, ty) || 1;
        const w = (width * scale) / 2;
        p.push(geo.x[i] * scale + (-ty / L) * w * sign,
               (geo.has_elevation ? geo.z[i] : 0) * scale + 0.01,
               geo.y[i] * scale + (tx / L) * w * sign);
      }
      p.push(p[0], p[1], p[2]);
      return new Float32Array(p);
    };
    return [mk(1), mk(-1)];
  }, [geo, width]);
  const objs = useMemo(() => lines.map((p) => {
    const g = new THREE.BufferGeometry();
    g.setAttribute('position', new THREE.BufferAttribute(p, 3));
    return new THREE.Line(g, new THREE.LineBasicMaterial({
      color: C.white, transparent: true, opacity: 0.32 }));
  }), [lines]);
  return <>{objs.map((o, i) => <primitive key={i} object={o} />)}</>;
}
