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

## Sock selection v2 (`select_v2.py`, `eval_v2.py`)
Added shape/contrast/relative-size filters and watershed splitting of merged masks.
Result on the 18 Laundromat pile photos: exactly 10 socks found in 9/18 (v1: 10/18), 38/45 pairs
correct in those (v1: 43/50). No real improvement: stacked/touching socks still merge in SAM's
"everything" masks, and bed folds at the border still slip in.

Also tried: YOLOE zero-shot with the text prompt "sock" (`yoloe-11l-seg.pt`). Too weak
(0-11 low-confidence, partial detections per photo), not usable.
SAM3 (what Laundromat uses for text-prompted sock masks) is gated and its weights are not downloadable here.
