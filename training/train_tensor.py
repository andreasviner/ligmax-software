#!/usr/bin/env python3
"""
Train several YOLO26 sizes on the SAME data + SAME fixed validation split, with an
identical config, and log a comparison row per model (mAP, per-class AP, speed,
params, FLOPs). Fully integrated with Weights & Biases (W&B) for cloud tracking.

Examples:
  python training/train_compare.py
"""
import os
import csv
import json
import argparse
import time
import torch
import wandb  # Fully migrated from TensorBoard to W&B

def get_device(arg):
    if arg:
        return arg
    return "0" if torch.cuda.is_available() else "cpu"

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "navier_merged", "data_split.yaml")))
    ap.add_argument("--sizes", nargs="+", default=["m", "l", "s"])  
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")                
    ap.add_argument("--fraction", type=float, default=1.0) 
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--hsv_s", type=float, default=0.75)   
    ap.add_argument("--hsv_v", type=float, default=0.55)   
    ap.add_argument("--degrees", type=float, default=8.0)  
    ap.add_argument("--translate", type=float, default=0.15)  
    ap.add_argument("--scale", type=float, default=0.55)   
    ap.add_argument("--shear", type=float, default=2.0)    
    ap.add_argument("--perspective", type=float, default=0.0008)  
    ap.add_argument("--project", default=os.path.abspath(
        os.path.join(os.path.dirname(__file__), "..", "runs", "compare")))
    ap.add_argument("--name", default="yolo26")
    ap.add_argument("--wandb_project", default="yolo26-architecture-comparison")
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
        
        # Initialize an isolated W&B run for this model size
        wandb_run = wandb.init(
            project=args.wandb_project,
            name=run_name,
            config=vars(args),
            reinit=True
        )

        try:
            run_dir = os.path.join(args.project, run_name)
            last = os.path.join(run_dir, "weights", "last.pt")
            
            if os.path.exists(last):  
                print(f">>> RESUME yolo26{sz} from {last}", flush=True)
                model = YOLO(last)
                try:
                    model.train(resume=True)  
                except Exception as re:
                    print(f"[resume] {type(re).__name__}: {re} -- run already complete, skipping to eval", flush=True)
            else:
                model = YOLO(f"yolo26{sz}.pt")  
                model.train(
                    data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=device, seed=args.seed, 
                    deterministic=False,  # Set to False to bypass unstable 4090 deterministic CUDA kernels
                    fraction=args.fraction, patience=args.patience,
                    project=args.project, name=run_name, exist_ok=True, plots=True,
                    verbose=True,
                    multi_scale=False,
                    scale=(0.25, 1),
                    close_mosaic=10,   
                    hsv_s=args.hsv_s, hsv_v=args.hsv_v, degrees=args.degrees,
                    translate=args.translate, shear=args.shear,
                    perspective=args.perspective,
                    workers=0
                )
                
            best = os.path.join(args.project, run_name, "weights", "best.pt")
            best = best if os.path.exists(best) else os.path.join(args.project, run_name, "weights", "last.pt")

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
            sp = m.speed  
            
            mAP50_95 = round(float(m.box.map), 5)
            mAP50 = round(float(m.box.map50), 5)
            params_M = round(params / 1e6, 3)
            infer_ms = round(float(sp.get("inference", float("nan"))), 3)

            row = {
                "model": f"yolo26{sz}", "params_M": params_M,
                "GFLOPs": round(flops, 2) if flops == flops else None,
                "mAP50-95": mAP50_95,
                "mAP50": mAP50,
                "precision": round(float(m.box.mp), 5),
                "recall": round(float(m.box.mr), 5),
                "infer_ms": infer_ms,
                "total_ms": round(sum(sp.values()), 3),
                "train_min": round((time.time() - t0) / 60, 2),
                "epochs": args.epochs, "imgsz": args.imgsz, "device": device,
                "per_class_mAP50-95": [round(float(x), 4) for x in list(m.box.maps)],
            }
            rows.append(row)
            print(f"[done] {row}", flush=True)

            # Log final benchmark metrics safely to the cloud run summary map
            wandb.log({
                "summary/mAP50-95": mAP50_95,
                "summary/mAP50": mAP50,
                "summary/params_M": params_M,
                "summary/GFLOPs": round(flops, 2) if flops == flops else None,
                "summary/infer_ms": infer_ms,
                "summary/train_minutes": row["train_min"]
            })

        except Exception as e:
            import traceback; traceback.print_exc()
            rows.append({"model": f"yolo26{sz}", "error": f"{type(e).__name__}: {e}"})
        
        finally:
            # Finalize W&B stream and flush out the VRAM caches between iterations
            wandb_run.finish()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

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
