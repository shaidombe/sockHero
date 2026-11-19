import sys, time, json
import numpy as np, cv2
from ultralytics import SAM
model = SAM(sys.argv[1])
out = {}
for path in sys.argv[2:]:
    img = cv2.imread(path)
    t = time.time()
    r = model(path, imgsz=1024, verbose=False)[0]
    dt = time.time() - t
    masks = r.masks.data.cpu().numpy() if r.masks is not None else np.zeros((0,1,1))
    H, W = img.shape[:2]
    keep = []
    for m in masks:
        m = cv2.resize(m.astype(np.uint8), (W, H), interpolation=cv2.INTER_NEAREST)
        a = m.mean()
        if 0.004 < a < 0.25: keep.append(m)
    vis = img.copy()
    rng = np.random.default_rng(1)
    for m in keep:
        c = rng.integers(60, 255, 3)
        vis[m > 0] = (0.5 * vis[m > 0] + 0.5 * c).astype(np.uint8)
        cs, _ = cv2.findContours(m, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(vis, cs, -1, (255, 255, 255), 2)
    name = path.split('/')[-1].rsplit('.', 1)[0]
    cv2.imwrite(f'{name}_{sys.argv[1].split(".")[0]}.jpg', cv2.resize(vis, (W // 2, H // 2)) if W > 1500 else vis)
    print(name, f'{W}x{H}', 'all masks', len(masks), 'kept', len(keep), f'{dt:.1f}s')
