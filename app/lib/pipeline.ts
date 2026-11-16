/**
 * SockHero vision pipeline.
 *
 * Pure, dependency-free and DOM-free so it runs unchanged in a Web Worker and
 * in Node tests. Everything works on typed arrays at a small working
 * resolution (the caller downsizes the frame), so a full analysis takes a few
 * milliseconds.
 *
 * Stages:
 *   1. RGB -> CIE Lab.
 *   2. Learn the surface (background) as a few colour clusters, first from the
 *      image border and then from everything that was not foreground.
 *   3. Foreground = pixels statistically far from every surface cluster,
 *      cleaned up with morphology and split into connected blobs (socks).
 *   4. Describe each sock: colour histogram, colour layout along its main
 *      axis, texture and size. All descriptors are rotation/mirror invariant.
 *   5. Pick the globally best set of pairs (exact for up to 16 socks).
 */

export interface AnalyzeOptions {
  /** 0 = pair generously, 1 = only near-identical socks. Default 0.5. */
  strictness?: number;
  /** Include the sock-to-sock distance matrix in the result (for tuning). */
  debug?: boolean;
}

export interface Sock {
  id: number;
  /** Normalised [x0, y0, x1, y1] in 0..1 image coordinates. */
  bbox: [number, number, number, number];
  /** Normalised centroid. */
  cx: number;
  cy: number;
  /** Outline as flat normalised [x0, y0, x1, y1, ...]. */
  contour: number[];
  /** Average sRGB colour. */
  color: [number, number, number];
  /** Fraction of the image covered. */
  area: number;
}

export interface Pair {
  a: number;
  b: number;
  /** 0..1, how far below the match threshold the pair's distance is. */
  confidence: number;
}

export interface Analysis {
  socks: Sock[];
  pairs: Pair[];
  /** Ids of socks left without a partner. */
  singles: number[];
  width: number;
  height: number;
  ms: number;
  /** Only with `debug`: distances[i][j] between socks i and j. */
  distances?: number[][];
  /** Only with `debug`: the distance below which two socks may pair. */
  threshold?: number;
  /** Only with `debug`: raw foreground mask and per-stage timings. */
  mask?: Uint8Array;
  timings?: Record<string, number>;
}

// ---------------------------------------------------------------------------
// Colour conversion
// ---------------------------------------------------------------------------

const SRGB_TO_LINEAR = new Float32Array(256);
for (let i = 0; i < 256; i++) {
  const c = i / 255;
  SRGB_TO_LINEAR[i] = c <= 0.04045 ? c / 12.92 : Math.pow((c + 0.055) / 1.055, 2.4);
}

const LAB_F_SIZE = 4096;
const LAB_F = new Float32Array(LAB_F_SIZE + 1);
for (let i = 0; i <= LAB_F_SIZE; i++) {
  const t = (i / LAB_F_SIZE) * 1.1;
  LAB_F[i] = t > 0.008856 ? Math.cbrt(t) : 7.787 * t + 16 / 116;
}
const labF = (t: number) => LAB_F[Math.min(LAB_F_SIZE, (t * (LAB_F_SIZE / 1.1)) | 0)];

function toLab(rgba: Uint8ClampedArray, n: number, L: Float32Array, A: Float32Array, B: Float32Array, gain?: Float32Array) {
  for (let i = 0, p = 0; i < n; i++, p += 4) {
    const k = gain ? gain[i] : 1;
    const r = SRGB_TO_LINEAR[rgba[p]] * k;
    const g = SRGB_TO_LINEAR[rgba[p + 1]] * k;
    const b = SRGB_TO_LINEAR[rgba[p + 2]] * k;
    const fx = labF((0.4124564 * r + 0.3575761 * g + 0.1804375 * b) / 0.95047);
    const fy = labF(0.2126729 * r + 0.7151522 * g + 0.072175 * b);
    const fz = labF((0.0193339 * r + 0.119192 * g + 0.9503041 * b) / 1.08883);
    L[i] = 116 * fy - 16;
    A[i] = 500 * (fx - fy);
    B[i] = 200 * (fy - fz);
  }
}

/**
 * Estimate uneven lighting from the surface itself. The surface is assumed to
 * have roughly one brightness, so any smooth change in its luminance across
 * the frame is lighting. Fits log-luminance of surface pixels with a quadratic
 * in (x, y) and returns the per-pixel gain that flattens it, or null when
 * there is too little surface to trust.
 */
function lightingGain(rgba: Uint8ClampedArray, w: number, h: number, foreground: Uint8Array): Float32Array | null {
  const n = w * h;
  const stride = Math.max(1, Math.floor(n / 5000));
  const ata = new Float64Array(36), atb = new Float64Array(6), f = new Float64Array(6);
  let count = 0;
  for (let i = (stride >> 1); i < n; i += stride) {
    if (foreground[i]) continue;
    const p = i * 4;
    const lum = 0.2126729 * SRGB_TO_LINEAR[rgba[p]] + 0.7151522 * SRGB_TO_LINEAR[rgba[p + 1]] + 0.072175 * SRGB_TO_LINEAR[rgba[p + 2]];
    const x = ((i % w) / w) * 2 - 1, y = (((i / w) | 0) / h) * 2 - 1;
    f[0] = 1; f[1] = x; f[2] = y; f[3] = x * x; f[4] = x * y; f[5] = y * y;
    const v = Math.log(lum + 0.003);
    for (let a = 0; a < 6; a++) { atb[a] += f[a] * v; for (let b = 0; b < 6; b++) ata[a * 6 + b] += f[a] * f[b]; }
    count++;
  }
  if (count < 300) return null;
  // Light ridge on the curvature terms keeps the fit tame where the surface
  // is only visible on one side of the frame.
  for (let a = 3; a < 6; a++) ata[a * 6 + a] += count * 0.02;

  // Solve the 6x6 normal equations by Gaussian elimination with pivoting.
  const c = new Float64Array(6);
  for (let col = 0; col < 6; col++) {
    let piv = col;
    for (let r = col + 1; r < 6; r++) if (Math.abs(ata[r * 6 + col]) > Math.abs(ata[piv * 6 + col])) piv = r;
    if (Math.abs(ata[piv * 6 + col]) < 1e-9) return null;
    if (piv !== col) {
      for (let k = 0; k < 6; k++) { const t = ata[col * 6 + k]; ata[col * 6 + k] = ata[piv * 6 + k]; ata[piv * 6 + k] = t; }
      const t = atb[col]; atb[col] = atb[piv]; atb[piv] = t;
    }
    for (let r = col + 1; r < 6; r++) {
      const m = ata[r * 6 + col] / ata[col * 6 + col];
      for (let k = col; k < 6; k++) ata[r * 6 + k] -= m * ata[col * 6 + k];
      atb[r] -= m * atb[col];
    }
  }
  for (let r = 5; r >= 0; r--) {
    let v = atb[r];
    for (let k = r + 1; k < 6; k++) v -= ata[r * 6 + k] * c[k];
    c[r] = v / ata[r * 6 + r];
  }

  // Normalise so the frame centre keeps its brightness; clamp to stay sane.
  const gain = new Float32Array(n);
  let spread = 0;
  for (let yy = 0; yy < h; yy++) {
    const y = (yy / h) * 2 - 1;
    for (let xx = 0; xx < w; xx++) {
      const x = (xx / w) * 2 - 1;
      const fit = c[1] * x + c[2] * y + c[3] * x * x + c[4] * x * y + c[5] * y * y;
      const g = Math.exp(-fit);
      const clamped = g < 0.55 ? 0.55 : g > 1.8 ? 1.8 : g;
      gain[yy * w + xx] = clamped;
      spread = Math.max(spread, Math.abs(clamped - 1));
    }
  }
  return spread > 0.03 ? gain : null; // already even: skip the extra work
}

/** 3x3 box blur, used only for segmentation to suppress sensor noise. */
function blur3(src: Float32Array, w: number, h: number, tmp: Float32Array, dst: Float32Array) {
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      const xl = x > 0 ? x - 1 : x;
      const xr = x < w - 1 ? x + 1 : x;
      tmp[row + x] = src[row + xl] + src[row + x] + src[row + xr];
    }
  }
  for (let y = 0; y < h; y++) {
    const row = y * w;
    const up = (y > 0 ? y - 1 : y) * w;
    const dn = (y < h - 1 ? y + 1 : y) * w;
    for (let x = 0; x < w; x++) dst[row + x] = (tmp[up + x] + tmp[row + x] + tmp[dn + x]) / 9;
  }
}

// ---------------------------------------------------------------------------
// Background model
// ---------------------------------------------------------------------------

interface Cluster {
  mL: number;
  mA: number;
  mB: number;
  /** Inverse variances (with floors) for a diagonal Mahalanobis distance. */
  iL: number;
  iA: number;
  iB: number;
}

const VAR_FLOOR_L = 36; // sigma 6: tolerate soft shading on the surface
const VAR_FLOOR_AB = 9; // sigma 3

/**
 * Small deterministic k-means over Lab samples. Farthest-point seeding puts
 * outliers (e.g. a sock poking into the border) in their own cluster, which
 * is then rejected for being too small.
 */
function surfaceClusters(s: Float32Array, n: number, k: number, minFraction: number): Cluster[] {
  if (n === 0) return [];
  const mean = new Float64Array(k * 3);
  const assign = new Uint8Array(n);
  const dist = new Float32Array(n).fill(Infinity);

  // Seed 0: the sample closest to the overall mean.
  let sL = 0, sA = 0, sB = 0;
  for (let i = 0; i < n; i++) { sL += s[i * 3]; sA += s[i * 3 + 1]; sB += s[i * 3 + 2]; }
  sL /= n; sA /= n; sB /= n;
  let seed = 0, best = Infinity;
  for (let i = 0; i < n; i++) {
    const d = (s[i * 3] - sL) ** 2 + (s[i * 3 + 1] - sA) ** 2 + (s[i * 3 + 2] - sB) ** 2;
    if (d < best) { best = d; seed = i; }
  }
  for (let c = 0; c < k; c++) {
    mean[c * 3] = s[seed * 3]; mean[c * 3 + 1] = s[seed * 3 + 1]; mean[c * 3 + 2] = s[seed * 3 + 2];
    let far = 0, farD = -1;
    for (let i = 0; i < n; i++) {
      const d = (s[i * 3] - mean[c * 3]) ** 2 + (s[i * 3 + 1] - mean[c * 3 + 1]) ** 2 + (s[i * 3 + 2] - mean[c * 3 + 2]) ** 2;
      if (d < dist[i]) dist[i] = d;
      if (dist[i] > farD) { farD = dist[i]; far = i; }
    }
    seed = far;
  }

  const sum = new Float64Array(k * 3);
  const count = new Int32Array(k);
  for (let iter = 0; iter < 8; iter++) {
    sum.fill(0); count.fill(0);
    for (let i = 0; i < n; i++) {
      const l = s[i * 3], a = s[i * 3 + 1], b = s[i * 3 + 2];
      let bc = 0, bd = Infinity;
      for (let c = 0; c < k; c++) {
        const d = (l - mean[c * 3]) ** 2 + (a - mean[c * 3 + 1]) ** 2 + (b - mean[c * 3 + 2]) ** 2;
        if (d < bd) { bd = d; bc = c; }
      }
      assign[i] = bc; count[bc]++;
      sum[bc * 3] += l; sum[bc * 3 + 1] += a; sum[bc * 3 + 2] += b;
    }
    for (let c = 0; c < k; c++) {
      if (!count[c]) continue;
      mean[c * 3] = sum[c * 3] / count[c]; mean[c * 3 + 1] = sum[c * 3 + 1] / count[c]; mean[c * 3 + 2] = sum[c * 3 + 2] / count[c];
    }
  }

  const variance = new Float64Array(k * 3);
  for (let i = 0; i < n; i++) {
    const c = assign[i];
    variance[c * 3] += (s[i * 3] - mean[c * 3]) ** 2;
    variance[c * 3 + 1] += (s[i * 3 + 1] - mean[c * 3 + 1]) ** 2;
    variance[c * 3 + 2] += (s[i * 3 + 2] - mean[c * 3 + 2]) ** 2;
  }

  let largest = 0;
  for (let c = 1; c < k; c++) if (count[c] > count[largest]) largest = c;
  const out: Cluster[] = [];
  for (let c = 0; c < k; c++) {
    if (count[c] < minFraction * n && c !== largest) continue;
    out.push({
      mL: mean[c * 3], mA: mean[c * 3 + 1], mB: mean[c * 3 + 2],
      iL: 1 / Math.max(VAR_FLOOR_L, variance[c * 3] / count[c]),
      iA: 1 / Math.max(VAR_FLOOR_AB, variance[c * 3 + 1] / count[c]),
      iB: 1 / Math.max(VAR_FLOOR_AB, variance[c * 3 + 2] / count[c]),
    });
  }
  return out;
}

/** A cast shadow is the surface, only darker: forgive this much lightness. */
const SHADOW_ALLOWANCE = 12;

/**
 * Squared Mahalanobis distance of every pixel to its nearest surface cluster.
 * Pixels darker than the surface get SHADOW_ALLOWANCE of lightness for free,
 * so the soft shadow around a sock is not mistaken for part of it.
 */
function surfaceDistance(L: Float32Array, A: Float32Array, B: Float32Array, n: number, model: Cluster[], out: Float32Array) {
  for (let i = 0; i < n; i++) {
    let d = Infinity;
    for (let c = 0; c < model.length; c++) {
      const m = model[c];
      let dl = L[i] - m.mL;
      if (dl < 0) dl = dl < -SHADOW_ALLOWANCE ? dl + SHADOW_ALLOWANCE : 0;
      const da = A[i] - m.mA, db = B[i] - m.mB;
      const v = dl * dl * m.iL + da * da * m.iA + db * db * m.iB;
      if (v < d) d = v;
    }
    out[i] = d;
  }
}

// ---------------------------------------------------------------------------
// Binary morphology
// ---------------------------------------------------------------------------

/** Separable square dilation (grow=1) or erosion (grow=0) with radius r. */
function morph(src: Uint8Array, w: number, h: number, r: number, grow: 0 | 1, tmp: Uint8Array, dst: Uint8Array) {
  const stop = grow;
  for (let y = 0; y < h; y++) {
    const row = y * w;
    for (let x = 0; x < w; x++) {
      let v = 1 - stop;
      const x0 = x - r < 0 ? 0 : x - r, x1 = x + r >= w ? w - 1 : x + r;
      for (let xx = x0; xx <= x1; xx++) if (src[row + xx] === stop) { v = stop; break; }
      tmp[row + x] = v;
    }
  }
  for (let y = 0; y < h; y++) {
    const y0 = y - r < 0 ? 0 : y - r, y1 = y + r >= h ? h - 1 : y + r;
    for (let x = 0; x < w; x++) {
      let v = 1 - stop;
      for (let yy = y0; yy <= y1; yy++) if (tmp[yy * w + x] === stop) { v = stop; break; }
      dst[y * w + x] = v;
    }
  }
}

/**
 * Fill small enclosed background regions (pattern details the same colour as
 * the surface). Large enclosed regions are real gaps between socks and stay.
 */
function fillHoles(mask: Uint8Array, w: number, h: number, stack: Int32Array, maxHole: number) {
  const n = w * h;
  let sp = 0;
  const push = (i: number) => { if (mask[i] === 0) { mask[i] = 2; stack[sp++] = i; } };
  const flood = () => {
    while (sp > 0) {
      const i = stack[--sp];
      const x = i % w;
      if (x > 0) push(i - 1);
      if (x < w - 1) push(i + 1);
      if (i >= w) push(i - w);
      if (i < n - w) push(i + w);
    }
  };
  for (let x = 0; x < w; x++) { push(x); push(n - w + x); }
  for (let y = 0; y < h; y++) { push(y * w); push(y * w + w - 1); }
  flood();
  // Anything still 0 is enclosed. Collect each hole (marking it 3) and fill it
  // only if it is small.
  for (let seed = 0; seed < n; seed++) {
    if (mask[seed] !== 0) continue;
    let head = 0, tail = 0;
    stack[tail++] = seed; mask[seed] = 3;
    while (head < tail) {
      const i = stack[head++];
      const x = i % w;
      if (x > 0 && mask[i - 1] === 0) { mask[i - 1] = 3; stack[tail++] = i - 1; }
      if (x < w - 1 && mask[i + 1] === 0) { mask[i + 1] = 3; stack[tail++] = i + 1; }
      if (i >= w && mask[i - w] === 0) { mask[i - w] = 3; stack[tail++] = i - w; }
      if (i < n - w && mask[i + w] === 0) { mask[i + w] = 3; stack[tail++] = i + w; }
    }
    const fill = tail <= maxHole ? 1 : 2;
    for (let k = 0; k < tail; k++) mask[stack[k]] = fill;
  }
  for (let i = 0; i < n; i++) if (mask[i] === 2) mask[i] = 0;
}

/** Chamfer (3-4) distance to the nearest background pixel, in 1/3 pixels. */
function distanceTransform(mask: Uint8Array, w: number, h: number, out: Uint16Array) {
  const BIG = 60000;
  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      const i = y * w + x;
      if (!mask[i]) { out[i] = 0; continue; }
      let d = BIG;
      if (x === 0 || y === 0) d = 3;
      else {
        d = Math.min(d, out[i - 1] + 3, out[i - w] + 3, out[i - w - 1] + 4);
        if (x < w - 1) d = Math.min(d, out[i - w + 1] + 4);
      }
      out[i] = d;
    }
  }
  for (let y = h - 1; y >= 0; y--) {
    for (let x = w - 1; x >= 0; x--) {
      const i = y * w + x;
      if (!mask[i]) continue;
      let d = out[i];
      if (x === w - 1 || y === h - 1) d = Math.min(d, 3);
      else {
        d = Math.min(d, out[i + 1] + 3, out[i + w] + 3, out[i + w + 1] + 4);
        if (x > 0) d = Math.min(d, out[i + w - 1] + 4);
      }
      out[i] = d;
    }
  }
}

/**
 * Split a blob where two socks merely touch. The thick "core" of each sock
 * survives a deep erosion as its own island; a thin contact bridge does not.
 * Each core then reclaims the surrounding pixels. Relabels `labels` and
 * regroups the blob's slice of `order` so every part is contiguous; returns
 * the [start, end, label] of each part.
 */
function splitAtNecks(
  start: number, end: number, label: number, nextLabel: () => number,
  order: Int32Array, labels: Int32Array, dt: Uint16Array, w: number, h: number,
  queue: Int32Array, scratch: Int32Array,
): [number, number, number][] {
  let maxD = 0;
  for (let k = start; k < end; k++) if (dt[order[k]] > maxD) maxD = dt[order[k]];
  if (maxD < 12) return [[start, end, label]]; // thinner than ~8px: nothing to split
  const coreT = maxD * 0.6;

  // Label core islands. Core pixels are temporarily marked with negative ids.
  const coreSizes: number[] = [];
  for (let k = start; k < end; k++) {
    const seed = order[k];
    if (labels[seed] !== label || dt[seed] <= coreT) continue;
    const id = -(coreSizes.length + 1);
    let head = 0, tail = 0;
    queue[tail++] = seed; labels[seed] = id;
    while (head < tail) {
      const i = queue[head++];
      const x = i % w, y = (i / w) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          if (labels[j] === label && dt[j] > coreT) { labels[j] = id; queue[tail++] = j; }
        }
      }
    }
    coreSizes.push(tail);
  }
  const biggest = Math.max(...coreSizes);
  const keep = coreSizes.map((size) => size >= biggest * 0.2);
  const kept = keep.filter(Boolean).length;
  if (kept < 2) {
    for (let k = start; k < end; k++) labels[order[k]] = label;
    return [[start, end, label]];
  }

  // Grow all kept cores outwards at the same speed over the blob.
  const partLabel = coreSizes.map((_, c) => (keep[c] ? nextLabel() : 0));
  let head = 0, tail = 0;
  for (let k = start; k < end; k++) {
    const i = order[k];
    if (labels[i] < 0) {
      const c = -labels[i] - 1;
      if (keep[c]) { labels[i] = partLabel[c]; queue[tail++] = i; } else labels[i] = label;
    }
  }
  while (head < tail) {
    const i = queue[head++];
    const x = i % w, me = labels[i];
    if (x > 0 && labels[i - 1] === label) { labels[i - 1] = me; queue[tail++] = i - 1; }
    if (x < w - 1 && labels[i + 1] === label) { labels[i + 1] = me; queue[tail++] = i + 1; }
    if (i >= w && labels[i - w] === label) { labels[i - w] = me; queue[tail++] = i - w; }
    if (i + w < w * h && labels[i + w] === label) { labels[i + w] = me; queue[tail++] = i + w; }
  }

  // Regroup the slice of `order` by part. Pixels only reachable diagonally
  // (never claimed above) are dropped by pushing them into a trailing reject
  // group that the caller ignores.
  const parts: [number, number, number][] = [];
  let write = start;
  for (let k = start; k < end; k++) scratch[k] = order[k];
  for (let c = 0; c < partLabel.length; c++) {
    if (!keep[c]) continue;
    const from = write;
    for (let k = start; k < end; k++) if (labels[scratch[k]] === partLabel[c]) order[write++] = scratch[k];
    parts.push([from, write, partLabel[c]]);
  }
  for (let k = start; k < end; k++) if (labels[scratch[k]] === label) { labels[scratch[k]] = 0; order[write++] = scratch[k]; }
  return parts;
}

// ---------------------------------------------------------------------------
// Blobs and descriptors
// ---------------------------------------------------------------------------

const HIST_L = 8, HIST_A = 9, HIST_B = 9;
const HIST_SIZE = HIST_L * HIST_A * HIST_B;
const AB_RANGE = 72;
const SEGMENTS = 5;

interface Blob {
  label: number;
  start: number; // range in the shared pixel-order array
  end: number;
  area: number;
  x0: number; y0: number; x1: number; y1: number;
  cx: number; cy: number;
  ux: number; uy: number; // unit vector of the principal axis
  length: number; // extent along the principal axis
  hist: Float32Array;
  mean: [number, number, number];
  segments: Float32Array; // SEGMENTS x 3 Lab means along the axis
  texture: Float32Array;
  rgb: [number, number, number];
}

function describe(
  blob: Blob, order: Int32Array, labels: Int32Array, w: number, h: number,
  L: Float32Array, A: Float32Array, B: Float32Array, rgba: Uint8ClampedArray,
) {
  const { start, end, label } = blob;
  const area = end - start;

  // Centroid and principal axis from second moments.
  let sx = 0, sy = 0;
  for (let k = start; k < end; k++) { const i = order[k]; sx += i % w; sy += (i / w) | 0; }
  const cx = sx / area, cy = sy / area;
  let cxx = 0, cyy = 0, cxy = 0;
  for (let k = start; k < end; k++) {
    const i = order[k];
    const dx = (i % w) - cx, dy = ((i / w) | 0) - cy;
    cxx += dx * dx; cyy += dy * dy; cxy += dx * dy;
  }
  const theta = 0.5 * Math.atan2(2 * cxy, cxx - cyy);
  const ux = Math.cos(theta), uy = Math.sin(theta);
  let tMin = Infinity, tMax = -Infinity;
  for (let k = start; k < end; k++) {
    const i = order[k];
    const t = ((i % w) - cx) * ux + (((i / w) | 0) - cy) * uy;
    if (t < tMin) tMin = t;
    if (t > tMax) tMax = t;
  }
  blob.cx = cx; blob.cy = cy; blob.ux = ux; blob.uy = uy;
  blob.length = tMax - tMin + 1;

  const hist = new Float32Array(HIST_SIZE);
  const seg = new Float64Array(SEGMENTS * 3);
  const segCount = new Float64Array(SEGMENTS);
  const tex = new Float64Array(4);
  let mL = 0, mA = 0, mB = 0, r = 0, g = 0, b = 0;
  let used = 0, gradSum = 0, gradCount = 0;
  const segScale = SEGMENTS / (tMax - tMin + 1e-6);

  // Two passes at most: prefer interior pixels (edge pixels are blended with
  // the surface); fall back to all pixels for very thin blobs.
  for (let pass = 0; pass < 2 && used < area * 0.3; pass++) {
    if (pass === 1) { hist.fill(0); seg.fill(0); segCount.fill(0); tex.fill(0); mL = mA = mB = r = g = b = 0; used = 0; gradSum = 0; gradCount = 0; }
    for (let k = start; k < end; k++) {
      const i = order[k];
      const x = i % w, y = (i / w) | 0;
      const interior = x > 0 && x < w - 1 && y > 0 && y < h - 1 &&
        labels[i - 1] === label && labels[i + 1] === label && labels[i - w] === label && labels[i + w] === label;
      if (pass === 0 && !interior) continue;
      used++;
      const l = L[i], a = A[i], bb = B[i];
      mL += l; mA += a; mB += bb;
      r += rgba[i * 4]; g += rgba[i * 4 + 1]; b += rgba[i * 4 + 2];

      // Trilinear soft-binned Lab histogram.
      const fl = Math.min(HIST_L - 1.001, Math.max(0, (l / 100) * HIST_L - 0.5));
      const fa = Math.min(HIST_A - 1.001, Math.max(0, ((a + AB_RANGE) / (2 * AB_RANGE)) * HIST_A - 0.5));
      const fb = Math.min(HIST_B - 1.001, Math.max(0, ((bb + AB_RANGE) / (2 * AB_RANGE)) * HIST_B - 0.5));
      const il = fl | 0, ia = fa | 0, ib = fb | 0;
      const wl = fl - il, wa = fa - ia, wb = fb - ib;
      const base = (il * HIST_A + ia) * HIST_B + ib;
      const dL = HIST_A * HIST_B, dA = HIST_B;
      hist[base] += (1 - wl) * (1 - wa) * (1 - wb);
      hist[base + 1] += (1 - wl) * (1 - wa) * wb;
      hist[base + dA] += (1 - wl) * wa * (1 - wb);
      hist[base + dA + 1] += (1 - wl) * wa * wb;
      hist[base + dL] += wl * (1 - wa) * (1 - wb);
      hist[base + dL + 1] += wl * (1 - wa) * wb;
      hist[base + dL + dA] += wl * wa * (1 - wb);
      hist[base + dL + dA + 1] += wl * wa * wb;

      // Colour layout along the principal axis.
      const t = (x - cx) * ux + (y - cy) * uy;
      const si = Math.min(SEGMENTS - 1, ((t - tMin) * segScale) | 0);
      seg[si * 3] += l; seg[si * 3 + 1] += a; seg[si * 3 + 2] += bb;
      segCount[si]++;

      // Texture: gradient energy split by direction relative to the axis
      // (folded to 0..90 degrees so it is rotation and mirror invariant).
      if (interior) {
        const gx = (L[i + 1] - L[i - 1]) * 0.5, gy = (L[i + w] - L[i - w]) * 0.5;
        const mag = Math.sqrt(gx * gx + gy * gy);
        gradSum += mag; gradCount++;
        if (mag > 1.5) {
          const along = Math.abs(gx * ux + gy * uy), across = Math.abs(-gx * uy + gy * ux);
          const ang = Math.atan2(across, along) / (Math.PI / 2); // 0 = along axis
          const f = Math.min(2.999, ang * 3);
          const bi = f | 0;
          tex[bi] += mag * (1 - (f - bi));
          tex[bi + 1] += mag * (f - bi);
        }
      }
    }
  }

  const inv = 1 / Math.max(1, used);
  for (let i = 0; i < HIST_SIZE; i++) hist[i] *= inv;
  blob.hist = hist;
  blob.mean = [mL * inv, mA * inv, mB * inv];
  blob.rgb = [Math.round(r * inv), Math.round(g * inv), Math.round(b * inv)];

  const segments = new Float32Array(SEGMENTS * 3);
  for (let s = 0; s < SEGMENTS; s++) {
    if (segCount[s] > 0) {
      segments[s * 3] = seg[s * 3] / segCount[s];
      segments[s * 3 + 1] = seg[s * 3 + 1] / segCount[s];
      segments[s * 3 + 2] = seg[s * 3 + 2] / segCount[s];
    } else {
      segments[s * 3] = blob.mean[0]; segments[s * 3 + 1] = blob.mean[1]; segments[s * 3 + 2] = blob.mean[2];
    }
  }
  blob.segments = segments;

  // Texture vector: overall strength, then how the gradient energy is spread
  // over four directions from "along the sock" to "across the sock".
  const meanGrad = gradCount ? gradSum / gradCount : 0;
  const strength = meanGrad / (meanGrad + 4);
  const texTotal = tex[0] + tex[1] + tex[2] + tex[3] || 1;
  blob.texture = new Float32Array([
    strength,
    (tex[0] / texTotal) * strength,
    (tex[1] / texTotal) * strength,
    (tex[2] / texTotal) * strength,
    (tex[3] / texTotal) * strength,
  ]);
}

const CONTOUR_DX = [-1, -1, 0, 1, 1, 1, 0, -1];
const CONTOUR_DY = [0, -1, -1, -1, 0, 1, 1, 1];
const CONTOUR_BACK = [6, 6, 0, 0, 2, 2, 4, 4];

/** Moore-neighbour boundary trace, subsampled to at most maxPoints. */
function traceContour(blob: Blob, order: Int32Array, labels: Int32Array, w: number, h: number, maxPoints: number): number[] {
  // The first pixel in raster order has no blob pixel to its west or north.
  let startIdx = order[blob.start];
  for (let k = blob.start + 1; k < blob.end; k++) if (order[k] < startIdx) startIdx = order[k];
  const sx = startIdx % w, sy = (startIdx / w) | 0;
  const label = blob.label;
  const pts: number[] = [sx, sy];
  let x = sx, y = sy, back = 0;
  const limit = blob.area * 4 + 16;
  for (let step = 0; step < limit; step++) {
    let moved = false;
    for (let i = 1; i <= 8; i++) {
      const d = (back + i) & 7;
      const nx = x + CONTOUR_DX[d], ny = y + CONTOUR_DY[d];
      if (nx < 0 || ny < 0 || nx >= w || ny >= h || labels[ny * w + nx] !== label) continue;
      x = nx; y = ny; back = CONTOUR_BACK[d]; moved = true;
      break;
    }
    if (!moved || (x === sx && y === sy)) break;
    pts.push(x, y);
  }
  const count = pts.length / 2;
  const stride = Math.max(1, Math.ceil(count / maxPoints));
  const out: number[] = [];
  for (let i = 0; i < count; i += stride) out.push((pts[i * 2] + 0.5) / w, (pts[i * 2 + 1] + 0.5) / h);
  return out;
}

// ---------------------------------------------------------------------------
// Matching
// ---------------------------------------------------------------------------

const L_WEIGHT = 0.7; // lightness is less reliable than chroma under uneven light

function labDelta(a: ArrayLike<number>, ai: number, b: ArrayLike<number>, bi: number) {
  const dl = (a[ai] - b[bi]) * L_WEIGHT, da = a[ai + 1] - b[bi + 1], db = a[ai + 2] - b[bi + 2];
  return Math.sqrt(dl * dl + da * da + db * db);
}

/** 0 = identical, grows with dissimilarity; Infinity = cannot be a pair. */
function sockDistance(p: Blob, q: Blob): number {
  const areaRatio = Math.min(p.area, q.area) / Math.max(p.area, q.area);
  if (areaRatio < 0.4) return Infinity;
  const lengthRatio = Math.min(p.length, q.length) / Math.max(p.length, q.length);

  let overlap = 0;
  for (let i = 0; i < HIST_SIZE; i++) overlap += p.hist[i] < q.hist[i] ? p.hist[i] : q.hist[i];
  const dHist = 1 - overlap;

  const dMean = Math.min(1, labDelta(p.mean, 0, q.mean, 0) / 30);

  let fwd = 0, rev = 0, worstFwd = 0, worstRev = 0;
  for (let s = 0; s < SEGMENTS; s++) {
    const f = labDelta(p.segments, s * 3, q.segments, s * 3);
    const r = labDelta(p.segments, s * 3, q.segments, (SEGMENTS - 1 - s) * 3);
    fwd += f; rev += r;
    if (f > worstFwd) worstFwd = f;
    if (r > worstRev) worstRev = r;
  }
  const useFwd = fwd <= rev;
  const dLayout = Math.min(1, (useFwd ? fwd : rev) / SEGMENTS / 25);
  // One clearly different band (e.g. a red toe vs a blue toe) should hurt even
  // though it barely moves the averages.
  const dWorst = Math.min(1, (useFwd ? worstFwd : worstRev) / 45);

  let dTex = 0;
  for (let i = 0; i < p.texture.length; i++) dTex += Math.abs(p.texture[i] - q.texture[i]);
  dTex = Math.min(1, dTex);

  // Small differences are normal (a sock lying bunched up); large ones mean
  // a different size of sock.
  const dSize = Math.min(1, Math.max(0, (0.8 - Math.min(areaRatio, lengthRatio)) / 0.35));

  return 0.3 * dHist + 0.2 * dMean + 0.15 * dLayout + 0.15 * dWorst + 0.12 * dTex + 0.2 * dSize;
}

/**
 * Choose disjoint pairs maximising the total margin below the threshold.
 * Exact bitmask DP for up to 16 candidates, greedy beyond that.
 */
function matchPairs(dist: Float32Array, n: number, threshold: number): [number, number, number][] {
  const gain = (i: number, j: number) => threshold - dist[i * n + j];
  const result: [number, number, number][] = [];

  if (n <= 16) {
    const size = 1 << n;
    const best = new Float32Array(size);
    const choice = new Int8Array(size).fill(-1);
    for (let mask = 1; mask < size; mask++) {
      const i = 31 - Math.clz32(mask & -mask); // lowest set bit
      const rest = mask & ~(1 << i);
      let b = best[rest], c = -1; // leave sock i single
      for (let j = i + 1; j < n; j++) {
        if (!(rest & (1 << j))) continue;
        const gn = gain(i, j);
        if (gn <= 0) continue;
        const v = gn + best[rest & ~(1 << j)];
        if (v > b) { b = v; c = j; }
      }
      best[mask] = b; choice[mask] = c;
    }
    let mask = size - 1;
    while (mask) {
      const i = 31 - Math.clz32(mask & -mask);
      const j = choice[mask];
      mask &= ~(1 << i);
      if (j >= 0) { result.push([i, j, dist[i * n + j]]); mask &= ~(1 << j); }
    }
    return result;
  }

  const candidates: [number, number, number][] = [];
  for (let i = 0; i < n; i++) for (let j = i + 1; j < n; j++) if (gain(i, j) > 0) candidates.push([i, j, dist[i * n + j]]);
  candidates.sort((a, b) => a[2] - b[2]);
  const taken = new Uint8Array(n);
  for (const c of candidates) {
    if (taken[c[0]] || taken[c[1]]) continue;
    taken[c[0]] = taken[c[1]] = 1;
    result.push(c);
  }
  return result;
}

// ---------------------------------------------------------------------------
// Entry point
// ---------------------------------------------------------------------------

const T_LOW = 2.5 ** 2; // candidate foreground
const T_HIGH = 3.0 ** 2; // confident foreground
const MIN_AREA = 0.0015;
const MAX_AREA = 0.45;
const MAX_SOCKS = 40;

export function analyze(rgba: Uint8ClampedArray, width: number, height: number, options: AnalyzeOptions = {}): Analysis {
  const t0 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const w = width, h = height, n = w * h;
  const strictness = Math.min(1, Math.max(0, options.strictness ?? 0.5));
  const threshold = 0.26 - 0.16 * strictness;

  const L = new Float32Array(n), A = new Float32Array(n), B = new Float32Array(n);
  toLab(rgba, n, L, A, B);
  const tmpF = new Float32Array(n);
  const sL = new Float32Array(n), sA = new Float32Array(n), sB = new Float32Array(n);
  blur3(L, w, h, tmpF, sL); blur3(A, w, h, tmpF, sA); blur3(B, w, h, tmpF, sB);

  // --- Surface model, pass 1: the image border -----------------------------
  const band = Math.max(2, Math.round(Math.min(w, h) * 0.04));
  const borderCount = 2 * band * w + 2 * band * Math.max(0, h - 2 * band);
  const stride1 = Math.max(1, Math.floor(borderCount / 4000));
  const maxSamples = 8000;
  const samples = new Float32Array(maxSamples * 3);
  let ns = 0, seen = 0;
  for (let y = 0; y < h; y++) {
    const edgeRow = y < band || y >= h - band;
    for (let x = 0; x < w; x++) {
      if (!edgeRow && x >= band && x < w - band) { x = w - band - 1; continue; }
      if (seen++ % stride1 === 0 && ns < maxSamples) {
        const i = y * w + x;
        samples[ns * 3] = sL[i]; samples[ns * 3 + 1] = sA[i]; samples[ns * 3 + 2] = sB[i]; ns++;
      }
    }
  }
  let model = surfaceClusters(samples, ns, 3, 0.12);
  const dist = new Float32Array(n);
  surfaceDistance(sL, sA, sB, n, model, dist);

  const mask = new Uint8Array(n), tmpM = new Uint8Array(n), tmpM2 = new Uint8Array(n);
  let fg = 0;
  for (let i = 0; i < n; i++) { mask[i] = dist[i] > T_LOW ? 1 : 0; fg += mask[i]; }

  // --- Pass 2: flatten the lighting, then relearn the surface from
  // everything that is not foreground ---------------------------------------
  const radius = Math.max(1, Math.round(Math.min(w, h) / 180));
  if (fg < n * 0.6) {
    morph(mask, w, h, radius + 1, 1, tmpM, tmpM2);
    const gain = lightingGain(rgba, w, h, tmpM2);
    if (gain) {
      toLab(rgba, n, L, A, B, gain);
      blur3(L, w, h, tmpF, sL); blur3(A, w, h, tmpF, sA); blur3(B, w, h, tmpF, sB);
    }
    const stride2 = Math.max(1, Math.floor((n - fg) / 6000));
    ns = 0; seen = 0;
    for (let i = 0; i < n; i++) {
      if (tmpM2[i]) continue;
      if (seen++ % stride2 === 0 && ns < maxSamples) {
        samples[ns * 3] = sL[i]; samples[ns * 3 + 1] = sA[i]; samples[ns * 3 + 2] = sB[i]; ns++;
      }
    }
    if (ns > 200) {
      model = surfaceClusters(samples, ns, 4, 0.04);
      surfaceDistance(sL, sA, sB, n, model, dist);
      for (let i = 0; i < n; i++) mask[i] = dist[i] > T_LOW ? 1 : 0;
    }
  }

  // --- Clean up: close hairline gaps, fill small holes, drop specks -------
  const stack = new Int32Array(n);
  morph(mask, w, h, 1, 1, tmpM, tmpM2);
  morph(tmpM2, w, h, 1, 0, tmpM, mask);
  fillHoles(mask, w, h, stack, Math.round(n * 0.002));
  morph(mask, w, h, radius, 0, tmpM, tmpM2);
  morph(tmpM2, w, h, radius, 1, tmpM, mask);

  // --- Connected components (8-connectivity), split where socks touch ------
  const dt = new Uint16Array(n);
  distanceTransform(mask, w, h, dt);
  const labels = new Int32Array(n);
  const order = new Int32Array(n);
  const scratch = new Int32Array(n);
  let filled = 0, labelCount = 0;
  const nextLabel = () => ++labelCount;
  const blobs: Blob[] = [];
  const minArea = Math.max(24, n * MIN_AREA), maxArea = n * MAX_AREA;
  for (let seedIdx = 0; seedIdx < n; seedIdx++) {
    if (!mask[seedIdx] || labels[seedIdx]) continue;
    const label = nextLabel();
    const start = filled;
    let sp = 0;
    stack[sp++] = seedIdx; labels[seedIdx] = label;
    while (sp > 0) {
      const i = stack[--sp];
      order[filled++] = i;
      const x = i % w, y = (i / w) | 0;
      for (let dy = -1; dy <= 1; dy++) {
        const yy = y + dy;
        if (yy < 0 || yy >= h) continue;
        for (let dx = -1; dx <= 1; dx++) {
          const xx = x + dx;
          if (xx < 0 || xx >= w) continue;
          const j = yy * w + xx;
          if (mask[j] && !labels[j]) { labels[j] = label; stack[sp++] = j; }
        }
      }
    }
    if (filled - start < minArea) continue;

    for (const [from, to, partLabel] of splitAtNecks(start, filled, label, nextLabel, order, labels, dt, w, h, stack, scratch)) {
      const area = to - from;
      if (area < minArea || area > maxArea) continue;
      let x0 = w, y0 = h, x1 = 0, y1 = 0, strong = 0, edge = 0;
      for (let k = from; k < to; k++) {
        const i = order[k];
        const x = i % w, y = (i / w) | 0;
        if (x < x0) x0 = x; if (x > x1) x1 = x; if (y < y0) y0 = y; if (y > y1) y1 = y;
        if (dist[i] > T_HIGH) strong++;
        if (x === 0 || y === 0 || x === w - 1 || y === h - 1) edge++;
      }
      if (strong < area * 0.2) continue; // faint blob: shading, not a sock
      if (edge > 0.6 * Math.sqrt(area)) continue; // cut off by the frame
      blobs.push({
        label: partLabel, start: from, end: to, area, x0, y0, x1, y1,
        cx: 0, cy: 0, ux: 1, uy: 0, length: 0,
        hist: null as unknown as Float32Array, mean: [0, 0, 0],
        segments: null as unknown as Float32Array, texture: null as unknown as Float32Array, rgb: [0, 0, 0],
      });
    }
  }
  blobs.sort((p, q) => q.area - p.area);
  if (blobs.length > MAX_SOCKS) blobs.length = MAX_SOCKS;

  // --- Describe and match --------------------------------------------------
  for (const blob of blobs) describe(blob, order, labels, w, h, L, A, B, rgba);
  const m = blobs.length;
  const pairDist = new Float32Array(m * m);
  for (let i = 0; i < m; i++) for (let j = i + 1; j < m; j++) pairDist[i * m + j] = pairDist[j * m + i] = sockDistance(blobs[i], blobs[j]);
  const matches = matchPairs(pairDist, m, threshold);

  const socks: Sock[] = blobs.map((blob, id) => ({
    id,
    bbox: [blob.x0 / w, blob.y0 / h, (blob.x1 + 1) / w, (blob.y1 + 1) / h],
    cx: (blob.cx + 0.5) / w,
    cy: (blob.cy + 0.5) / h,
    contour: traceContour(blob, order, labels, w, h, 72),
    color: blob.rgb,
    area: blob.area / n,
  }));
  const paired = new Uint8Array(m);
  const pairs: Pair[] = matches
    .sort((p, q) => p[2] - q[2])
    .map(([a, b, d]) => { paired[a] = paired[b] = 1; return { a, b, confidence: Math.min(1, Math.max(0, 1 - d / threshold)) }; });
  const singles: number[] = [];
  for (let i = 0; i < m; i++) if (!paired[i]) singles.push(i);

  const t1 = typeof performance !== 'undefined' ? performance.now() : Date.now();
  const result: Analysis = { socks, pairs, singles, width: w, height: h, ms: t1 - t0 };
  if (options.debug) {
    result.threshold = threshold;
    result.mask = Uint8Array.from(labels, (v) => (v > 0 ? 1 + (v % 250) : 0));
    result.distances = blobs.map((_, i) => blobs.map((__, j) => (i === j ? 0 : pairDist[i * m + j])));
  }
  return result;
}
