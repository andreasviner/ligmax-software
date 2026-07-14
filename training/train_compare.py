#!/usr/bin/env python3
"""
Train several YOLO26 sizes on the SAME data + SAME fixed validation split, with an
identical seed/config, and log a comparison row per model (mAP, per-class AP, speed,
params, FLOPs). Designed to run full on a GPU, or as a tiny CPU smoke-test.

Examples:
  # Full comparison (GPU):
  python training/train_compare.py --sizes s m l x --epochs 150 --imgsz 640 --batch -1

  # CPU smoke-test (validate the pipeline; not meaningful metrics):
  python training/train_compare.py --sizes n --epochs 2 --imgsz 320 --batch 4 \
      --fraction 0.01 --device cpu --name smoke
"""
import os, csv, json, argparse, time

def get_device(arg):
    if arg:
        return arg
    try:
        import torch
        return "0" if torch.cuda.is_available() else "cpu"
    except Exception:
        return "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "navier_merged", "data_split.yaml")))
    ap.add_argument("--sizes", nargs="+", default=["s", "m", "l", "x"])
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=-1)      # -1 = auto (GPU)
    ap.add_argument("--device", default="")                # "" = auto
    ap.add_argument("--fraction", type=float, default=1.0) # subset train set (smoke)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=50)
    ap.add_argument("--project", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "runs", "compare")))
    ap.add_argument("--name", default="yolo26")
    args = ap.parse_args()

    from ultralytics import YOLO
    from ultralytics.utils.torch_utils import get_flops
    device = get_device(args.device)
    print(f"[train_compare] device={device} sizes={args.sizes} epochs={args.epochs} "
          f"imgsz={args.imgsz} fraction={args.fraction}", flush=True)

    results_dir = os.path.join(args.project, "_summary")
    os.makedirs(results_dir, exist_ok=True)
    csv_path = os.path.join(results_dir, "compare_results.csv")
    rows = []
    names = None

    for sz in args.sizes:
        run_name = f"{args.name}_{sz}"
        print(f"\n{'='*60}\n>>> TRAIN yolo26{sz}\n{'='*60}", flush=True)
        t0 = time.time()
        try:
            model = YOLO(f"yolo26{sz}.pt")  # COCO-pretrained -> transfer learning
            model.train(
                data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                batch=args.batch, device=device, seed=args.seed, deterministic=True,
                fraction=args.fraction, patience=args.patience,
                project=args.project, name=run_name, exist_ok=True, plots=True,
                verbose=True,
            )
            best = os.path.join(args.project, run_name, "weights", "best.pt")
            best = best if os.path.exists(best) else os.path.join(
                args.project, run_name, "weights", "last.pt")

            ev0 = YOLO(best)
            m = ev0.val(data=args.data, imgsz=args.imgsz, device=device,
                        split="val", project=args.project, name=f"{run_name}_val",
                        exist_ok=True, verbose=False)
            names = getattr(m, "names", None) or names
            params = sum(p.numel() for p in ev0.model.parameters())
            try:
                flops = get_flops(ev0.model, args.imgsz)
            except Exception:
                flops = float("nan")
            sp = m.speed  # dict: preprocess/inference/postprocess (ms/img)
            row = {
                "model": f"yolo26{sz}", "params_M": round(params / 1e6, 3),
                "GFLOPs": round(flops, 2) if flops == flops else None,
                "mAP50-95": round(float(m.box.map), 5),
                "mAP50": round(float(m.box.map50), 5),
                "precision": round(float(m.box.mp), 5),
                "recall": round(float(m.box.mr), 5),
                "infer_ms": round(float(sp.get("inference", float("nan"))), 3),
                "total_ms": round(sum(sp.values()), 3),
                "train_min": round((time.time() - t0) / 60, 2),
                "epochs": args.epochs, "imgsz": args.imgsz, "device": device,
                "per_class_mAP50-95": [round(float(x), 4) for x in list(m.box.maps)],
            }
            rows.append(row)
            print(f"[done] {row}", flush=True)
        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({"model": f"yolo26{sz}", "error": f"{type(e).__name__}: {e}"})

        # write incrementally so a mid-run crash still leaves partial results
        _flush(csv_path, rows, names)

    with open(os.path.join(results_dir, "compare_results.json"), "w") as f:
        json.dump({"class_names": names, "rows": rows}, f, indent=2)
    print(f"\n[train_compare] summary -> {csv_path}", flush=True)

def _flush(csv_path, rows, names):
    base = ["model", "params_M", "GFLOPs", "mAP50-95", "mAP50", "precision",
            "recall", "infer_ms", "total_ms", "train_min", "epochs", "imgsz",
            "device", "per_class_mAP50-95", "error"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=base)
        w.writeheader()
        for r in rows:
            w.writerow({k: r.get(k, "") for k in base})

if __name__ == "__main__":
    main()
