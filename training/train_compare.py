#!/usr/bin/env python3
"""
Train several YOLO26 sizes on the SAME data + SAME fixed validation split, with an
identical seed/config, and log a comparison row per model (mAP, per-class AP, speed,
params, FLOPs). Designed to run full on a GPU, or as a tiny CPU smoke-test.

Examples:
  # Full comparison (GPU) -- defaults already match this: sizes s/m/l, 150 epochs,
  # imgsz 640, batch 16, NO early stopping, multi_scale + sun/fisheye augmentation:
  python training/train_compare.py

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
    ap.add_argument("--sizes", nargs="+", default=["s", "m", "l"])  # add "x" for the biggest
    ap.add_argument("--epochs", type=int, default=150)
    ap.add_argument("--imgsz", type=int, default=640)
    # Fixed batch (NOT autobatch): multi_scale trains up to 1.5x imgsz, which OOMs
    # batch=-1 (autobatch probes at base imgsz then overflows at peak scale). 16 is
    # safe for s/m/l on a 24GB 4090 and keeps the size comparison fair. Raise if you
    # have headroom (watch epoch 1 -- a multi_scale OOM shows up there).
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--device", default="")                # "" = auto
    ap.add_argument("--fraction", type=float, default=1.0) # subset train set (smoke)
    ap.add_argument("--seed", type=int, default=0)
    # 0 => no early stopping (Ultralytics maps patience=0 -> inf); trains full --epochs.
    ap.add_argument("--patience", type=int, default=0)
    # Augmentation tuned for the real use case: strong sun/glare + 220-deg fisheye
    # lenses. "Train hard, fight easy" -> between Ultralytics defaults and heavy.
    # (Only applied to fresh runs; resumed runs keep their checkpoint's saved args.)
    ap.add_argument("--hsv_s", type=float, default=0.75)   # sun washout (def 0.7)
    ap.add_argument("--hsv_v", type=float, default=0.55)   # sun/overexposure brightness (def 0.4)
    ap.add_argument("--degrees", type=float, default=8.0)  # boat roll / edge tilt (def 0.0)
    ap.add_argument("--translate", type=float, default=0.15)  # exercise radial positions (def 0.1)
    ap.add_argument("--scale", type=float, default=0.55)   # wide-angle scale variation (def 0.5)
    ap.add_argument("--shear", type=float, default=2.0)    # off-axis geometry (def 0.0)
    ap.add_argument("--perspective", type=float, default=0.0008)  # closest lever to fisheye warp (def 0.0)
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
            run_dir = os.path.join(args.project, run_name)
            last = os.path.join(run_dir, "weights", "last.pt")
            if os.path.exists(last):  # crash/restart recovery: continue where we left off
                print(f">>> RESUME yolo26{sz} from {last}", flush=True)
                model = YOLO(last)
                try:
                    model.train(resume=True)  # uses the checkpoint's saved args
                except Exception as re:
                    # resume raises once a run has already hit its epoch/patience limit
                    print(f"[resume] {type(re).__name__}: {re} -- run already complete, "
                          f"skipping to eval", flush=True)
            else:
                model = YOLO(f"yolo26{sz}.pt")  # COCO-pretrained -> transfer learning
                model.train(
                    data=args.data, epochs=args.epochs, imgsz=args.imgsz,
                    batch=args.batch, device=device, seed=args.seed, deterministic=True,
                    fraction=args.fraction, patience=args.patience,
                    project=args.project, name=run_name, exist_ok=True, plots=True,
                    verbose=True,
                    # sun + 220-deg fisheye augmentation (see argparse block above)
                    multi_scale=0.5,   # FLOAT fraction (NOT bool) in this Ultralytics: +/-50% -> trains at 0.5x-1.5x imgsz (~320-960 @ 640). (multi_scale=True == 1.0 lets the random size floor to 0 -> interpolate crash.) OK on 24GB with a FIXED batch; pairs badly with batch=-1.
                    close_mosaic=10,   # mosaic off for the last 10 epochs (reached because early stop is disabled by default; see --patience)
                    hsv_s=args.hsv_s, hsv_v=args.hsv_v, degrees=args.degrees,
                    translate=args.translate, scale=args.scale, shear=args.shear,
                    perspective=args.perspective,
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
