# Experiments: SAM segmentation + embedding pairing

Python experiments (not part of the Next.js app yet).

- `seg.py` – run MobileSAM "segment everything" on images (`ultralytics`).
- `cache_masks.py` – cache SAM masks for the Laundromat test piles
  (https://github.com/idofrizler/laundromat, `testing/data/piles`).
- `match_eval.py` – select sock masks, build rotation-normalised crops, embed with
  a timm ResNet50 (weights loaded from a local `resnet50_a1.pth`), match pairs with an
  exact bitmask DP, and score against the known layouts.

Result so far (CPU, MobileSAM + ResNet50 a1 features, no fine-tuning):
in the 10 of 18 pile photos where exactly 10 socks were selected, 43 of 50 pairs were correct.
Sock selection from the SAM masks is the weak spot (merged or missed socks in dense piles).
SAM takes ~90 s per photo on 4 CPU cores.
