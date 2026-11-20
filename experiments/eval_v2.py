import sys, glob, numpy as np, cv2
sys.path.insert(0, '/home/user/sockHero/experiments')
from select_v2 import select_socks_v2
from match_eval import canonical_crops, embed, similarity, best_matching, expected
tot_ok = tot = exact = 0
for npz in sorted(glob.glob('*.npz')):
    n = npz[:-4]; kind = n.split('__')[0]
    m = np.load(npz)['masks']; img = cv2.imread(n + '.jpg')
    sel = select_socks_v2(img, m, None)
    k = len(sel); res = ''
    if k == 10:
        exact += 1
        S = similarity(embed([canonical_crops(img, s) for s in sel])); pairs = best_matching(S)
        cy = [np.nonzero(s)[0].mean() for s in sel]; pos = {int(a): r + 1 for r, a in enumerate(np.argsort(cy))}
        got = {tuple(sorted((pos[a], pos[b]))) for a, b in pairs}
        ok = len(got & expected(kind)); tot_ok += ok; tot += 5; res = f'{ok}/5'
    print(n[-20:], 'socks', k, res, flush=True)
print(f'exactly 10 socks: {exact}/18   correct pairs {tot_ok}/{tot}')
