#!/usr/bin/env python3
"""Confirm the trained models are actually retrievable from W&B -- run from ANY machine
(your laptop, the training box, anywhere you've done `wandb login`). This is your proof that
the training was NOT for nothing.

    python training/verify_models.py                      # list what's on the server
    python training/verify_models.py --download ./models  # also download the final models
    python training/verify_models.py --project njord-yolo26-3class --entity YOUR_ENTITY

For each size, the FINAL model is the artifact `yolo26{sz}-final` (and, identically,
`yolo26{sz}-finetune`). `-base` is the phase-1 model; `-*-ckpt` are periodic safety copies.
Exit code 0 iff every requested size has a final model on the server.
"""
import argparse
import os
import sys

import wandb

# checked in priority order; first match counts as "final present"
FINAL_SUFFIXES = ["final", "finetune"]
OTHER_SUFFIXES = ["base", "base-ckpt", "finetune-ckpt"]


def _describe(art):
    try:
        mb = sum(f.size for f in art.files()) / 1e6
    except Exception:
        mb = float("nan")
    created = getattr(art, "created_at", "?")
    return f"v{art.version}  {mb:6.1f} MB  {created}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--project", default="njord-yolo26-3class")
    ap.add_argument("--entity", default="", help="W&B entity/username (default: your login)")
    ap.add_argument("--sizes", nargs="+", default=["m", "n", "s", "l"])
    ap.add_argument("--download", default="", help="directory to download the final models into")
    args = ap.parse_args()

    try:
        api = wandb.Api(timeout=30)
        entity = args.entity or api.default_entity
    except Exception as e:
        print(f"[verify] cannot reach W&B ({type(e).__name__}: {e}). Run `wandb login` first.")
        sys.exit(2)
    if not entity:
        print("[verify] no W&B entity found. Run `wandb login` first.")
        sys.exit(2)

    base = f"{entity}/{args.project}"
    print(f"[verify] project: {base}\n")

    all_ok = True
    for sz in args.sizes:
        print(f"yolo26{sz}:")
        final_ok = False
        for suf in FINAL_SUFFIXES + OTHER_SUFFIXES:
            name = f"{base}/yolo26{sz}-{suf}:latest"
            try:
                art = api.artifact(name, type="model")
            except Exception:
                print(f"   --  {sz}-{suf}: not found")
                continue
            print(f"   OK  {sz}-{suf:15s} {_describe(art)}")
            if suf in FINAL_SUFFIXES and not final_ok:
                final_ok = True
                if args.download:
                    dst = os.path.join(args.download, f"yolo26{sz}-{suf}")
                    try:
                        art.download(root=dst)
                        print(f"        downloaded -> {dst}")
                    except Exception as e:
                        print(f"        DOWNLOAD FAILED: {type(e).__name__}: {e}")
        if not final_ok:
            print(f"   !!! yolo26{sz}: NO FINAL MODEL on the server yet "
                  f"(training may still be running).")
            all_ok = False
        print()

    if all_ok:
        print("[verify] OK -- every requested size has a final model on W&B. Safe.")
        sys.exit(0)
    print("[verify] INCOMPLETE -- some sizes have no final model yet. Re-run this later.")
    sys.exit(1)


if __name__ == "__main__":
    main()
