import { describe, expect, it } from 'vitest';
import { analyze, type Analysis } from '../app/lib/pipeline';
import { renderScene, STYLES, type PlacedSock, type RGB, type SockStyle } from './synth';

const SURFACES: Record<string, RGB> = {
  wood: [176, 138, 96],
  beige: [222, 210, 190],
  teal: [70, 150, 150],
  white: [244, 244, 240],
  dark: [52, 50, 56],
};

interface Scored { found: number; correctPairs: number; wrongPairs: number; expectedPairs: number; ms: number }

/** Map every detected sock to the ground-truth sock under its centroid. */
function score(result: Analysis, truth: PlacedSock[], w: number, h: number, counts: number[]): Scored {
  const styleOf = result.socks.map((s) => {
    let best = -1, bestD = Infinity;
    truth.forEach((t, i) => {
      const d = Math.hypot(s.cx * w - t.x, s.cy * h - t.y);
      if (d < bestD) { bestD = d; best = i; }
    });
    return bestD < Math.min(w, h) * 0.15 ? truth[best].style : -1;
  });
  let correct = 0, wrong = 0;
  for (const p of result.pairs) {
    if (styleOf[p.a] >= 0 && styleOf[p.a] === styleOf[p.b]) correct++;
    else wrong++;
  }
  return {
    found: result.socks.length,
    correctPairs: correct,
    wrongPairs: wrong,
    expectedPairs: counts.reduce((a, c) => a + Math.floor(c / 2), 0),
    ms: result.ms,
  };
}

function run(surface: RGB, styles: SockStyle[], counts: number[], seed: number, extra: Partial<Parameters<typeof renderScene>[0]> = {}) {
  const scene = renderScene({ width: 480, height: 360, surface, styles, counts, seed, shadow: true, ...extra });
  const result = analyze(scene.rgba, scene.width, scene.height);
  return { result, scored: score(result, scene.socks, scene.width, scene.height, counts), total: counts.reduce((a, c) => a + c, 0) };
}

describe('segmentation', () => {
  it.each(Object.keys(SURFACES))('finds every sock on a %s surface', (name) => {
    const surface = SURFACES[name];
    // Skip socks that are the same colour as the surface: invisible by design.
    const styles = [STYLES.red, STYLES.navy, STYLES.green, STYLES.yellow, STYLES.pink, name === 'dark' ? STYLES.white : STYLES.black];
    for (let seed = 1; seed <= 5; seed++) {
      const { result, total } = run(surface, styles, styles.map(() => 2), seed);
      expect(result.socks.length, `seed ${seed}`).toBe(total);
    }
  });

  it('copes with a two-tone checked surface', () => {
    const styles = [STYLES.red, STYLES.navy, STYLES.black, STYLES.yellow];
    const { result, scored } = run([214, 200, 180], styles, [2, 2, 2, 2], 3, { surfaceAlt: [196, 182, 160] });
    expect(result.socks.length).toBe(8);
    expect(scored.correctPairs).toBe(4);
  });

  it('reports nothing on an empty surface', () => {
    const { result } = run(SURFACES.wood, [STYLES.red], [0], 1);
    expect(result.socks.length).toBe(0);
    expect(result.pairs.length).toBe(0);
  });
});

describe('pairing', () => {
  it('pairs plain socks of distinct colours at any angle', () => {
    const styles = [STYLES.red, STYLES.navy, STYLES.green, STYLES.yellow, STYLES.pink, STYLES.white, STYLES.black];
    for (let seed = 1; seed <= 10; seed++) {
      const { scored } = run(SURFACES.wood, styles, styles.map(() => 2), seed);
      expect(scored, `seed ${seed}`).toMatchObject({ correctPairs: 7, wrongPairs: 0 });
    }
  });

  it('tells apart socks that differ only in pattern or accent colour', () => {
    const styles = [STYLES.navy, STYLES.navyStripes, STYLES.red, STYLES.redStripes, STYLES.blackRedToe, STYLES.blackBlueToe, STYLES.greyDots, STYLES.greyDotsPink];
    for (let seed = 1; seed <= 10; seed++) {
      const { scored } = run(SURFACES.beige, styles, styles.map(() => 2), seed);
      expect(scored, `seed ${seed}`).toMatchObject({ correctPairs: 8, wrongPairs: 0 });
    }
  });

  it('separates black from navy and adult from kids sizes', () => {
    const styles = [STYLES.black, STYLES.navy, STYLES.kidsBlack, STYLES.grey];
    for (let seed = 1; seed <= 10; seed++) {
      const { scored } = run(SURFACES.beige, styles, styles.map(() => 2), seed);
      expect(scored, `seed ${seed}`).toMatchObject({ correctPairs: 4, wrongPairs: 0 });
    }
  });

  it('leaves odd socks single instead of forcing a match', () => {
    const styles = [STYLES.red, STYLES.navy, STYLES.green, STYLES.blackRedToe, STYLES.blackBlueToe, STYLES.navyStripes];
    const counts = [2, 2, 1, 1, 1, 1];
    for (let seed = 1; seed <= 10; seed++) {
      const { result, scored } = run(SURFACES.wood, styles, counts, seed);
      expect(scored, `seed ${seed}`).toMatchObject({ correctPairs: 2, wrongPairs: 0 });
      expect(result.singles.length, `seed ${seed}`).toBe(4);
    }
  });

  it('survives strong uneven lighting and noise', () => {
    const styles = [STYLES.red, STYLES.navy, STYLES.green, STYLES.yellow, STYLES.grey, STYLES.navyStripes];
    for (let seed = 1; seed <= 10; seed++) {
      const { scored } = run(SURFACES.beige, styles, styles.map(() => 2), seed, { lighting: 0.3, noise: 14 });
      expect(scored, `seed ${seed}`).toMatchObject({ correctPairs: 6, wrongPairs: 0 });
    }
  });

  it('pairs up a drawer of identical socks', () => {
    const { scored } = run(SURFACES.wood, [STYLES.black], [8], 2);
    expect(scored).toMatchObject({ correctPairs: 4, wrongPairs: 0 });
  });
});

describe('performance', () => {
  it('analyses a 640x360 frame with 16 socks in well under a frame budget', () => {
    const styles = Object.values(STYLES).slice(0, 8);
    const scene = renderScene({ width: 640, height: 360, surface: SURFACES.wood, styles, counts: styles.map(() => 2), seed: 7, shadow: true });
    analyze(scene.rgba, scene.width, scene.height); // warm up the JIT
    let best = Infinity;
    for (let i = 0; i < 5; i++) best = Math.min(best, analyze(scene.rgba, scene.width, scene.height).ms);
    expect(best).toBeLessThan(60);
  });
});
