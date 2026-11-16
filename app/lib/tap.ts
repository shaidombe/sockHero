/**
 * Segment the object under a tap: grow a region from the tapped point in Lab
 * space at several tolerances and keep the most stable one (the tolerance
 * range over which the region barely changes is where the object's own
 * boundary sits). Works on a small RGBA frame; returns a mask or null.
 */

export interface TapResult {
  /** 1 where the object is, size w*h. */
  mask: Uint8Array;
  bbox: [number, number, number, number]; // x0,y0,x1,y1 in pixels (inclusive)
  area: number;
  tolerance: number;
}

const lin = new Float32Array(256);
for (let i = 0; i < 256; i++) { const c = i / 255; lin[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4); }
const f = (t: number) => (t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116);

function lab(rgba: Uint8ClampedArray, n: number) {
  const out = new Float32Array(n * 3);
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const r = lin[rgba[p]], g = lin[rgba[p + 1]], b = lin[rgba[p + 2]];
    const fx = f((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047);
    const fy = f(0.2126729 * r + 0.7151522 * g + 0.072175 * b);
    const fz = f((0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883);
    out[i * 3] = 116 * fy - 16; out[i * 3 + 1] = 500 * (fx - fy); out[i * 3 + 2] = 200 * (fy - fz);
  }
  return out;
}

function blur(src: Float32Array, w: number, h: number, r: number) {
  const tmp = new Float32Array(src.length), out = new Float32Array(src.length);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const a = Math.max(0, x - r), b = Math.min(w - 1, x + r);
    for (let c = 0; c < 3; c++) { let s = 0; for (let k = a; k <= b; k++) s += src[(y * w + k) * 3 + c]; tmp[(y * w + x) * 3 + c] = s / (b - a + 1); }
  }
  for (let y = 0; y < h; y++) {
    const a = Math.max(0, y - r), b = Math.min(h - 1, y + r);
    for (let x = 0; x < w; x++) for (let c = 0; c < 3; c++) { let s = 0; for (let k = a; k <= b; k++) s += tmp[(k * w + x) * 3 + c]; out[(y * w + x) * 3 + c] = s / (b - a + 1); }
  }
  return out;
}

function grow(L: Float32Array, w: number, h: number, sx: number, sy: number, tol: number, mean: number[], mask: Uint8Array, stack: Int32Array) {
  mask.fill(0);
  let sp = 0, count = 0;
  const s = sy * w + sx;
  mask[s] = 1; stack[sp++] = s;
  const t2 = tol * tol;
  while (sp > 0) {
    const i = stack[--sp]; count++;
    const x = i % w, y = (i / w) | 0;
    for (let k = 0; k < 4; k++) {
      const nx = k === 0 ? x - 1 : k === 1 ? x + 1 : x, ny = k === 2 ? y - 1 : k === 3 ? y + 1 : y;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const j = ny * w + nx;
      if (mask[j]) continue;
      const dl = L[j * 3] - mean[0], da = L[j * 3 + 1] - mean[1], db = L[j * 3 + 2] - mean[2];
      if (dl * dl * 0.5 + da * da + db * db <= t2) { mask[j] = 1; stack[sp++] = j; }
    }
  }
  return count;
}

export function segmentAtPoint(rgba: Uint8ClampedArray, w: number, h: number, px: number, py: number): TapResult | null {
  const n = w * h;
  const sx = Math.min(w - 1, Math.max(0, Math.round(px))), sy = Math.min(h - 1, Math.max(0, Math.round(py)));
  const L = blur(lab(rgba, n), w, h, Math.max(1, Math.round(Math.min(w, h) / 120)));

  // Seed colour: median-ish average of a small neighbourhood around the tap.
  const r = Math.max(2, Math.round(Math.min(w, h) / 60));
  const mean = [0, 0, 0]; let c = 0;
  for (let y = Math.max(0, sy - r); y <= Math.min(h - 1, sy + r); y++) for (let x = Math.max(0, sx - r); x <= Math.min(w - 1, sx + r); x++) {
    const i = y * w + x; mean[0] += L[i * 3]; mean[1] += L[i * 3 + 1]; mean[2] += L[i * 3 + 2]; c++;
  }
  mean[0] /= c; mean[1] /= c; mean[2] /= c;

  const stack = new Int32Array(n), mask = new Uint8Array(n);
  const tols = [4, 6, 8, 11, 14, 18, 23, 29, 36];
  const areas: number[] = [];
  for (const t of tols) areas.push(grow(L, w, h, sx, sy, t, mean, mask, stack));

  // Most stable tolerance: smallest relative area change to its neighbours,
  // ignoring regions that are tiny or swallow most of the frame.
  let best = -1, bestScore = Infinity;
  for (let i = 0; i < tols.length; i++) {
    if (areas[i] < n * 0.004 || areas[i] > n * 0.5) continue;
    const prev = i > 0 ? areas[i - 1] : areas[i], next = i < tols.length - 1 ? areas[i + 1] : areas[i] * 2;
    const score = (next - prev) / areas[i];
    if (score < bestScore) { bestScore = score; best = i; }
  }
  if (best < 0) return null;
  grow(L, w, h, sx, sy, tols[best], mean, mask, stack);

  let x0 = w, y0 = h, x1 = 0, y1 = 0, area = 0;
  for (let i = 0; i < n; i++) if (mask[i]) { const x = i % w, y = (i / w) | 0; if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y; area++; }
  return { mask, bbox: [x0, y0, x1, y1], area, tolerance: tols[best] };
}
