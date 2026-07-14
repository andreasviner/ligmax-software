# YOLO26 model-size comparison (s / m / l / x)

Trains multiple YOLO26 sizes on the **same data and same fixed validation split**,
then plots the accuracy/latency/size trade-offs used to pick the deployment model
for the Jetson Orin Nano.

## Contents
- `make_split.py` — builds the fixed split (`navier_merged/{train,val,test}.txt` + `data_split.yaml`) from the dataset's original split. Already run.
- `train_compare.py` — trains each size with identical config/seed and logs a metrics row per model.
- `plot_compare.py` — reads the summary and writes comparison PNGs.

## Data
- Labeled dataset: `navier_merged/` (30,157 imgs @ ≤1280px, 6 classes: green,red,north,east,south,west)
- Split (fixed, identical for every model): train 29,592 / val 430 / test 135
- Config: `navier_merged/data_split.yaml`

## Run the full comparison (GPU required)
This machine is CPU-only; run on a CUDA GPU (cloud or local). Then:

```bash
pip install "ultralytics>=8.4.14"          # YOLO26 support
python training/make_split.py              # only if paths changed / dataset moved

# full run: s, m, l, x @ 640px, 150 epochs, same val set, seed 0
python training/train_compare.py --sizes s m l x --epochs 150 --imgsz 640 --batch -1

python training/plot_compare.py            # -> runs/compare/_summary/*.png
```

Outputs:
- `runs/compare/_summary/compare_results.csv` / `.json` — one row per model
  (mAP50-95, mAP50, precision, recall, per-class AP, inference ms, params, GFLOPs, train time)
- `runs/compare/_summary/*.png` — accuracy-vs-latency, accuracy-vs-params, mAP bars, per-class bars
- `runs/compare/yolo26_<size>/` — each model's weights (`weights/best.pt`) + Ultralytics curves/confusion matrix

## Notes
- `--batch -1` auto-picks batch for the GPU's VRAM. On a small GPU (e.g. 8–12 GB) the `x` model may need an explicit smaller `--batch` or `--imgsz 512`.
- Rough GPU time: single mid-range GPU, 150 epochs on 30k imgs ≈ hours for `s`, longer scaling up to `x`. A 4-model sweep is a multi-hour to overnight job.
- `--fraction <f>` trains on a subset (used for the CPU smoke-test); leave at 1.0 for real runs.
- Validated end-to-end via a CPU smoke-test (`--sizes n --epochs 2 --fraction 0.01`).
