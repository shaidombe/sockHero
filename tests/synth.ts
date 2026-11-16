/**
 * Procedural test scenes: socks of various colours and patterns scattered at
 * random angles on a surface, with sensor noise, uneven lighting and soft
 * shadows. Deterministic for a given seed.
 */

export type RGB = [number, number, number];

export interface SockStyle {
  base: RGB;
  /** Stripes across the leg. */
  stripe?: RGB;
  stripePeriod?: number;
  /** Polka dots. */
  dot?: RGB;
  /** Contrasting toe + heel + cuff. */
  accent?: RGB;
  /** Relative size, 1 = adult. */
  scale?: number;
}

export interface PlacedSock {
  style: number; // index into the styles list; socks sharing it are a pair
  x: number;
  y: number;
  angle: number;
  mirror: boolean;
}

export interface SceneOptions {
  width: number;
  height: number;
  surface: RGB;
  /** Second surface colour for a woven/checked texture. */
  surfaceAlt?: RGB;
  styles: SockStyle[];
  /** How many socks to place per style (2 = a pair, 1 = a single). */
  counts: number[];
  seed: number;
  noise?: number;
  /** 0..1 strength of the left-to-right lighting falloff. */
  lighting?: number;
  shadow?: boolean;
}

export function rng(seed: number) {
  let s = seed >>> 0 || 1;
  return () => {
    s ^= s << 13; s >>>= 0;
    s ^= s >>> 17;
    s ^= s << 5; s >>>= 0;
    return s / 4294967296;
  };
}

function segmentDistance(px: number, py: number, ax: number, ay: number, bx: number, by: number) {
  const vx = bx - ax, vy = by - ay;
  const t = Math.max(0, Math.min(1, ((px - ax) * vx + (py - ay) * vy) / (vx * vx + vy * vy)));
  const dx = px - (ax + t * vx), dy = py - (ay + t * vy);
  return { d: Math.sqrt(dx * dx + dy * dy), t };
}

export function renderScene(opts: SceneOptions): { rgba: Uint8ClampedArray; width: number; height: number; socks: PlacedSock[] } {
  const { width: w, height: h, styles, counts } = opts;
  const rand = rng(opts.seed);
  const noise = opts.noise ?? 6;
  const lighting = opts.lighting ?? 0.15;
  const unit = Math.min(w, h);

  // Place socks on a jittered grid so they never touch.
  const order: number[] = [];
  counts.forEach((c, s) => { for (let i = 0; i < c; i++) order.push(s); });
  for (let i = order.length - 1; i > 0; i--) { const j = Math.floor(rand() * (i + 1)); [order[i], order[j]] = [order[j], order[i]]; }
  const cols = Math.ceil(Math.sqrt(order.length * (w / h)));
  const rows = Math.ceil(order.length / cols);
  const cellW = w / cols, cellH = h / rows;
  const cell = Math.min(cellW, cellH);
  const socks: PlacedSock[] = order.map((style, i) => ({
    style,
    x: (i % cols + 0.5) * cellW + (rand() - 0.5) * cellW * 0.12,
    y: (Math.floor(i / cols) + 0.5) * cellH + (rand() - 0.5) * cellH * 0.12,
    angle: rand() * Math.PI * 2,
    mirror: rand() < 0.5,
  }));

  const rgba = new Uint8ClampedArray(w * h * 4);
  const shadowOffset = unit * 0.012;

  for (let y = 0; y < h; y++) {
    for (let x = 0; x < w; x++) {
      let color: RGB = opts.surface;
      if (opts.surfaceAlt && ((Math.floor(x / (unit * 0.02)) + Math.floor(y / (unit * 0.02))) & 1)) color = opts.surfaceAlt;
      let shade = 1;

      for (const sock of socks) {
        const st = styles[sock.style];
        const size = cell * 0.36 * (st.scale ?? 1);
        const leg = size, foot = size * 0.75, rad = size * 0.2;
        const cos = Math.cos(-sock.angle), sin = Math.sin(-sock.angle);
        const test = (px: number, py: number) => {
          let u = (px - sock.x) * cos - (py - sock.y) * sin;
          const v = (px - sock.x) * sin + (py - sock.y) * cos;
          if (sock.mirror) u = -u;
          // Leg: (0,-leg)->(0,0), foot: (0,0)->(foot*0.8, foot*0.6), shifted to centre the shape.
          const uu = u + foot * 0.3, vv = v - leg * 0.3;
          const a = segmentDistance(uu, vv, 0, -leg, 0, 0);
          const b = segmentDistance(uu, vv, 0, 0, foot * 0.8, foot * 0.6);
          return { a, b, uu, vv };
        };
        if (Math.abs(x - sock.x) > size * 1.6 || Math.abs(y - sock.y) > size * 1.6) continue;

        const p = test(x, y);
        const inLeg = p.a.d <= rad, inFoot = p.b.d <= rad;
        if (inLeg || inFoot) {
          let c = st.base;
          const along = inLeg ? p.a.t * leg : leg + p.b.t * foot;
          if (st.stripe && Math.floor(along / (size * (st.stripePeriod ?? 0.14))) % 2 === 0) c = st.stripe;
          if (st.dot) {
            const g = size * 0.16;
            const du = ((p.uu % g) + g) % g - g / 2, dv = ((p.vv % g) + g) % g - g / 2;
            if (du * du + dv * dv < (g * 0.28) ** 2) c = st.dot;
          }
          if (st.accent) {
            const cuff = inLeg && p.a.t < 0.14;
            const toe = !inLeg && p.b.t > 0.7;
            const heel = p.a.t > 0.9 && p.b.t < 0.12 && p.uu < 0;
            if (cuff || toe || heel) c = st.accent;
          }
          // A hint of fabric shading across the sock.
          const across = inLeg ? p.a.d / rad : p.b.d / rad;
          shade = 1 - 0.1 * across * across;
          color = c;
          break;
        } else if (opts.shadow) {
          const q = test(x - shadowOffset, y - shadowOffset);
          const d = Math.min(q.a.d, q.b.d) - rad;
          if (d < 0) shade = Math.min(shade, 0.88);
          else if (d < rad * 0.25) shade = Math.min(shade, 0.88 + 0.12 * (d / (rad * 0.25)));
        }
      }

      const light = 1 - lighting * (x / w) - lighting * 0.4 * (y / h);
      const i = (y * w + x) * 4;
      for (let ch = 0; ch < 3; ch++) {
        const nz = (rand() + rand() + rand() - 1.5) * noise;
        rgba[i + ch] = color[ch] * light * shade + nz;
      }
      rgba[i + 3] = 255;
    }
  }
  return { rgba, width: w, height: h, socks };
}

export const STYLES: Record<string, SockStyle> = {
  black: { base: [28, 28, 32] },
  navy: { base: [30, 42, 86] },
  white: { base: [238, 238, 232] },
  grey: { base: [128, 130, 134] },
  red: { base: [196, 40, 44] },
  green: { base: [38, 128, 72] },
  yellow: { base: [236, 196, 48] },
  pink: { base: [236, 128, 170] },
  navyStripes: { base: [30, 42, 86], stripe: [228, 228, 228] },
  redStripes: { base: [196, 40, 44], stripe: [228, 228, 228] },
  blackRedToe: { base: [28, 28, 32], accent: [200, 44, 48] },
  blackBlueToe: { base: [28, 28, 32], accent: [52, 96, 220] },
  greyDots: { base: [128, 130, 134], dot: [240, 210, 60] },
  greyDotsPink: { base: [128, 130, 134], dot: [236, 110, 170] },
  kidsBlack: { base: [28, 28, 32], scale: 0.62 },
};
