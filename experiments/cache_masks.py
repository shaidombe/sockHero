import sys, glob, os, time
import numpy as np, cv2
from ultralytics import SAM
model = SAM('mobile_sam.pt' if os.path.exists('mobile_sam.pt') else '../sam/mobile_sam.pt')
root = '/home/user/idofrizler/laundromat/testing/data/piles'
for kind in ['straight_line', 'outside_in']:
    for path in sorted(glob.glob(f'{root}/{kind}/*.jpg')):
        name = f'{kind}__{os.path.basename(path)[:-4]}'
        if os.path.exists(f'{name}.npz'): continue
        img = cv2.imread(path); H, W = img.shape[:2]
        s = 1280 / max(H, W); img = cv2.resize(img, (int(W * s), int(H * s)), interpolation=cv2.INTER_AREA)
        cv2.imwrite(f'{name}.jpg', img)
        t = time.time()
        r = model(f'{name}.jpg', imgsz=1024, verbose=False)[0]
        m = r.masks.data.cpu().numpy().astype(np.uint8) if r.masks is not None else np.zeros((0, 1, 1), np.uint8)
        h, w = img.shape[:2]
        m = np.stack([cv2.resize(x, (w, h), interpolation=cv2.INTER_NEAREST) for x in m]) if len(m) else m
        np.savez_compressed(f'{name}.npz', masks=m.astype(bool))
        print(name, img.shape, len(m), f'{time.time()-t:.1f}s', flush=True)
