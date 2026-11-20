"""Better sock selection from SAM "segment everything" masks.

v1 only used absolute area limits and containment. v2 adds:
  * shape and contrast filters (drop quilt patches, prints on a sock, floor/furniture),
  * size relative to the other candidates instead of a fixed fraction of the frame,
  * splitting of one mask that actually covers two touching socks (distance-transform
    watershed).
"""
import numpy as np
import cv2


def _stats(img_lab, masks):
    n, H, W = masks.shape
    out = []
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2 * max(3, int(0.012 * max(H, W))) + 1,) * 2)
    for i in range(n):
        m = masks[i].astype(np.uint8)
        area = m.sum() / (H * W)
        if area < 0.002:
            out.append(None)
            continue
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        c = max(cs, key=cv2.contourArea)
        hull = cv2.convexHull(c)
        solidity = cv2.contourArea(c) / max(1.0, cv2.contourArea(hull))
        edge = np.concatenate([masks[i][0], masks[i][-1], masks[i][:, 0], masks[i][:, -1]]).mean()
        ring = cv2.dilate(m, kernel) - m
        ring_px = img_lab[ring > 0]
        in_px = img_lab[m > 0]
        contrast = float(np.linalg.norm(in_px.mean(0) - ring_px.mean(0))) if len(ring_px) else 0.0
        out.append(dict(area=area, solidity=solidity, edge=edge, contrast=contrast))
    return out


def _split_touching(mask):
    """Split one mask into several if its distance transform has several strong peaks."""
    m = mask.astype(np.uint8)
    dt = cv2.distanceTransform(m, cv2.DIST_L2, 5)
    if dt.max() < 6:
        return [mask]
    core = (dt > 0.55 * dt.max()).astype(np.uint8)
    n, lab, st, _ = cv2.connectedComponentsWithStats(core)
    big = [k for k in range(1, n) if st[k, cv2.CC_STAT_AREA] > 0.25 * max(st[1:, cv2.CC_STAT_AREA])]
    if len(big) < 2:
        return [mask]
    markers = np.zeros(m.shape, np.int32)
    for idx, k in enumerate(big, 1):
        markers[lab == k] = idx
    markers[(m == 0)] = len(big) + 1
    ws = cv2.watershed(cv2.cvtColor((dt / dt.max() * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR), markers)
    parts = [(ws == idx) & mask for idx in range(1, len(big) + 1)]
    return [p for p in parts if p.sum() > 0.15 * mask.sum()] or [mask]


def select_socks_v2(img_bgr, masks, n_expected=None, return_info=False):
    n, H, W = masks.shape
    lab = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2LAB).astype(np.float32)
    st = _stats(lab, masks)
    cand = [i for i in range(n) if st[i] and 0.003 < st[i]['area'] < 0.5 and st[i]['edge'] < 0.02
            and st[i]['contrast'] > 3 and st[i]['solidity'] > 0.6]
    # composites (cover >=2 other candidates) and parts (sit inside a bigger candidate)
    def inside(i, j):
        return (masks[i] & masks[j]).sum() / max(1, masks[i].sum())
    composite = set()
    for j in cand:
        parts = [i for i in cand if i != j and st[i]['area'] < st[j]['area'] and inside(i, j) > 0.85
                 and st[i]['area'] > 0.3 * st[j]['area']]
        if len(parts) >= 2:
            composite.add(j)
    cand = [i for i in cand if i not in composite]
    cand = [i for i in cand if not any(j != i and st[j]['area'] > st[i]['area'] * 1.15 and inside(i, j) > 0.85
                                       for j in cand)]
    # size relative to the other candidates: a print or patch is far smaller than a sock
    if len(cand) >= 3:
        ref = np.percentile([st[i]['area'] for i in cand], 75)
        cand = [i for i in cand if st[i]['area'] > 0.2 * ref]
    # near duplicates
    out = []
    for i in sorted(cand, key=lambda k: -st[k]['area']):
        if all((masks[i] & masks[j]).sum() / (masks[i] | masks[j]).sum() < 0.7 for j in out):
            out.append(i)
    result = []
    for i in out:
        s = st[i]
        if s['solidity'] < 0.88 or (len(out) >= 2 and s['area'] > 1.7 * np.median([st[k]['area'] for k in out])):
            result.extend(_split_touching(masks[i]))
        else:
            result.append(masks[i])
    if n_expected and len(result) > n_expected:
        result = sorted(result, key=lambda m: -m.sum())[:n_expected]
    return np.stack(result) if result else np.zeros((0, H, W), bool)
