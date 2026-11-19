import sys, glob, os, itertools
import numpy as np, cv2, torch, timm

def select_socks(masks, n_expected=None):
    n = len(masks)
    area = masks.reshape(n, -1).mean(1)
    def border_frac(m):
        e = np.concatenate([m[0], m[-1], m[:, 0], m[:, -1]])
        return e.mean()
    idx = [i for i in range(n) if 0.004 < area[i] < 0.2 and border_frac(masks[i]) < 0.02]
    inter = {}
    def frac_in(i, j):  # fraction of i inside j
        k = (i, j)
        if k not in inter: inter[k] = (masks[i] & masks[j]).sum() / max(1, masks[i].sum())
        return inter[k]
    # composites: contain >=2 other candidates that each make up >=30% of it
    composite = set()
    for j in idx:
        inside = [i for i in idx if i != j and area[i] < area[j] and frac_in(i, j) > 0.85 and area[i] > 0.3 * area[j]]
        if len(inside) >= 2: composite.add(j)
    idx = [i for i in idx if i not in composite]
    keep = []
    for i in idx:
        if any(j != i and area[j] > area[i] * 1.15 and frac_in(i, j) > 0.85 for j in idx): continue
        keep.append(i)
    out = []
    for i in sorted(keep, key=lambda k: -area[k]):
        if all(((masks[i] & masks[j]).sum() / (masks[i] | masks[j]).sum()) < 0.7 for j in out): out.append(i)
    if n_expected and len(out) > n_expected: out = out[:n_expected]
    return out

def canonical_crops(img, mask, size=224):
    ys, xs = np.nonzero(mask)
    pts = np.stack([xs, ys], 1).astype(np.float32); c = pts.mean(0)
    u, s, vt = np.linalg.svd(pts - c, full_matrices=False); ax = vt[0]
    ang = np.degrees(np.arctan2(ax[1], ax[0])) - 90
    M = cv2.getRotationMatrix2D((float(c[0]), float(c[1])), ang, 1.0)
    H, W = mask.shape; Wd = int(np.hypot(H, W)) + 4
    M[0, 2] += Wd / 2 - c[0]; M[1, 2] += Wd / 2 - c[1]
    im = cv2.warpAffine(img, M, (Wd, Wd), flags=cv2.INTER_LINEAR, borderValue=(128, 128, 128))
    mk = cv2.warpAffine(mask.astype(np.uint8), M, (Wd, Wd), flags=cv2.INTER_NEAREST)
    im = np.where(mk[..., None] > 0, im, 128).astype(np.uint8)
    ys, xs = np.nonzero(mk); x0, x1, y0, y1 = xs.min(), xs.max() + 1, ys.min(), ys.max() + 1
    crop = im[y0:y1, x0:x1]; h, w = crop.shape[:2]; sc = size / max(h, w)
    crop = cv2.resize(crop, (max(1, int(w * sc)), max(1, int(h * sc))), interpolation=cv2.INTER_AREA)
    canvas = np.full((size, size, 3), 128, np.uint8); oy, ox = (size - crop.shape[0]) // 2, (size - crop.shape[1]) // 2
    canvas[oy:oy + crop.shape[0], ox:ox + crop.shape[1]] = crop
    return canvas

_model = None
def get_model():
    global _model
    if _model is None:
        m = timm.create_model('resnet50', pretrained=False, num_classes=0)
        sd = torch.load('resnet50_a1.pth', map_location='cpu', weights_only=True)
        sd = {k: v for k, v in sd.items() if not k.startswith('fc.')}
        print(m.load_state_dict(sd, strict=False)); m.eval(); _model = m
    return _model

MEAN = np.array([0.485, 0.456, 0.406], np.float32); STD = np.array([0.229, 0.224, 0.225], np.float32)
@torch.no_grad()
def embed(crops):
    m = get_model(); variants = []
    for c in crops:
        rgb = cv2.cvtColor(c, cv2.COLOR_BGR2RGB)
        vs = [rgb, rgb[::-1, ::-1], rgb[:, ::-1], rgb[::-1, :]]  # 0, 180, mirror, flip
        variants.append(np.stack(vs))
    x = np.concatenate(variants).astype(np.float32) / 255.0
    x = torch.from_numpy(((x - MEAN) / STD).transpose(0, 3, 1, 2))
    f = m(x).numpy(); f /= np.linalg.norm(f, axis=1, keepdims=True) + 1e-8
    return f.reshape(len(crops), 4, -1)

def similarity(F):
    n = len(F); S = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            if i != j: S[i, j] = max(F[i, 0] @ F[j, v] for v in range(4))
    return S

def best_matching(S):
    n = len(S); best = {0: (0.0, [])}
    from functools import lru_cache
    @lru_cache(None)
    def go(mask):
        if mask == (1 << n) - 1: return 0.0, ()
        i = next(k for k in range(n) if not mask >> k & 1)
        b = go(mask | 1 << i)
        for j in range(i + 1, n):
            if mask >> j & 1: continue
            s, p = go(mask | 1 << i | 1 << j)
            if s + S[i, j] > b[0]: b = (s + S[i, j], ((i, j),) + p)
        return b
    return list(go(0)[1])

def expected(kind):
    return {(1, 2), (3, 4), (5, 6), (7, 8), (9, 10)} if kind == 'straight_line' else {(1, 10), (2, 9), (3, 8), (4, 7), (5, 6)}

if __name__ == '__main__':
    tot_ok = tot = 0; rows = []
    for npz in sorted(glob.glob('*.npz')):
        name = npz[:-4]; kind = name.split('__')[0]
        masks = np.load(npz)['masks']; img = cv2.imread(name + '.jpg')
        sel = select_socks(masks, 10)
        if len(sel) != 10: rows.append((name, len(sel), None)); continue
        crops = [canonical_crops(img, masks[i]) for i in sel]
        F = embed(crops); S = similarity(F)
        pairs = best_matching(S)
        # map to order by y-centroid
        cy = [np.nonzero(masks[i])[0].mean() for i in sel]; order = np.argsort(cy); pos = {int(k): r + 1 for r, k in enumerate(order)}
        got = {tuple(sorted((pos[a], pos[b]))) for a, b in pairs}
        ok = len(got & expected(kind)); tot_ok += ok; tot += 5
        rows.append((name, len(sel), ok))
    for r in rows: print(r)
    print(f'correct pairs {tot_ok}/{tot}')
