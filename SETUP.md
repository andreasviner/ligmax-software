# Training setup — YOLO26 size comparison (s / m / l)

Target box: **Windows 11, Python 3.14, RTX 4090 (24 GB), CUDA installed.**
The whole `ligmax-software/` folder is self-contained — copy it over as-is. The
dataset split uses **relative paths**, so drive letter / username no longer matter.

---

## 1. Install PyTorch (manual, CUDA build) — do this FIRST

The pip wheels bundle their own CUDA runtime; only your GPU **driver** must be new
enough, so the wheel's `cuXXX` need not exactly match your installed toolkit.

```powershell
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu128
```

Pick the `cuXXX` tag at/below your CUDA (e.g. cu121 / cu124 / cu126 / cu128).

> **Python 3.14 caveat:** confirm a `cp314` wheel exists on that index. If not, use
> the nightly channel (`.../whl/nightly/cu128`) or run on Python 3.12/3.13.

Verify before continuing:

```powershell
python -c "import torch; print(torch.__version__, torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

You want `True` and `NVIDIA GeForce RTX 4090`.

## 2. Install the rest

```powershell
pip install -r requirements.txt
```

(Installing torch first prevents this step from pulling the CPU-only build.)

## 3. Train

```powershell
python training/train_compare.py
```

That bare command already does exactly the intended run (all baked into defaults):

| Setting | Value |
|---|---|
| Sizes | `s`, `m`, `l` (sequential) |
| Epochs | 150, **no early stopping** (`patience=0`) |
| Image size | 640, with `multi_scale=0.5` → trains at 0.5×–1.5× (≈320–960 px) |
| Batch | 16 (fixed — safe with multi_scale on 24 GB, keeps the comparison fair) |
| Augmentation | sun/glare (`hsv_s/v`), fisheye (`perspective`, `shear`, `degrees`), `scale`, `translate` |
| Mosaic | on, closed for the last 10 epochs |
| Seed | 0, deterministic |

Pretrained weights `yolo26{n,s,m,l}.pt` are already in the folder, and the AMP check
uses `yolo26n.pt` (also present) — so **no internet is needed to train.**

---

## Outputs

- Per-model run dirs: `runs/compare/yolo26_{s,m,l}/` (weights, curves, plots)
- Validation dirs: `runs/compare/yolo26_{s,m,l}_val/`
- **Comparison table:** `runs/compare/_summary/compare_results.csv` (+ `.json`)
  — mAP50-95 / mAP50, precision, recall, per-class AP, params, GFLOPs, speed, train time.
  Written incrementally after each model, so a crash still leaves partial results.

## Crash / restart recovery

Re-running the same command auto-resumes any model that has a `last.pt` (uses the
checkpoint's saved args), then continues to the next size. Safe to just re-launch.

## Troubleshooting

- **CUDA out of memory** (would appear in **epoch 1**, when multi_scale first hits 1.5×):
  lower the batch → `python training/train_compare.py --batch 8`.
  To use *more* of the card (faster), try `--batch 24` or `--batch 32` and watch epoch 1.
- **`torch.cuda.is_available()` is False:** you installed the CPU wheel. Reinstall torch
  from the `cuXXX` index (Step 1).
- **Dataset not found / no images:** keep `navier_merged/` intact — `data_split.yaml`,
  `train/val/test.txt`, `images/`, and `labels/` must stay together in that folder.
- **Regenerating the split** (`python training/make_split.py`, needs `manifest.csv`):
  now emits the same relative, portable paths — safe to re-run.
- **Only want a subset / add the biggest model:** `--sizes s` or `--sizes s m l x`.

## Expected runtime

A full 150-epoch run for all three sizes on a 4090 is a **multi-hour to ~1–2 day**
job (l is the slowest; multi_scale adds overhead). Plan for an uninterrupted run —
the resume logic above covers you if it drops.
