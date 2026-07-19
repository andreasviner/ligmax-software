#!/usr/bin/env python3
"""
Downscale an already-extracted chunk of dataset B into navier_b/, preserving the
class-by-folder structure (e.g. "East 6 m/"). No labels exist for B.

Usage: python downscale_b.py <src_dir> <dst_root>
  <src_dir>  : a directory tree containing "Datasets buoys-markers/<folder>/*.jpg"
  <dst_root> : output root (navier_b). Folder structure under
               "Datasets buoys-markers/" is recreated directly under dst_root.
"""
import os, sys
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

MAX_SIDE = 1280
JPEG_Q = 90
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp")
PREFIX = "Datasets buoys-markers"

def process_one(task):
    src, dst = task
    r = {"dst": dst, "ok": False, "resized": False, "w": 0, "h": 0, "err": ""}
    try:
        os.makedirs(os.path.dirname(dst), exist_ok=True)
        with Image.open(src) as im:
            im = im.convert("RGB")
            w, h = im.size
            r["w"], r["h"] = w, h
            m = max(w, h)
            if m > MAX_SIDE:
                s = MAX_SIDE / m
                im = im.resize((max(1, round(w*s)), max(1, round(h*s))), Image.LANCZOS)
                r["resized"] = True
            im.save(dst, "JPEG", quality=JPEG_Q, optimize=True)
        r["ok"] = True
    except Exception as e:
        r["err"] = f"{type(e).__name__}: {e}"
    return r

def rel_under_prefix(path, src_root):
    """Return path relative to the '<...>/Datasets buoys-markers/' component."""
    rel = os.path.relpath(path, src_root).replace("\\", "/")
    parts = rel.split("/")
    if PREFIX in parts:
        i = parts.index(PREFIX)
        return "/".join(parts[i+1:])
    return rel  # fallback: keep as-is

def main():
    src_root, dst_root = sys.argv[1], sys.argv[2]
    tasks = []
    for dirpath, _, files in os.walk(src_root):
        for fn in files:
            if fn.lower().endswith(IMG_EXTS):
                src = os.path.join(dirpath, fn)
                rel = rel_under_prefix(src, src_root)
                stem = os.path.splitext(rel)[0]
                dst = os.path.join(dst_root, stem + ".jpg")
                tasks.append((src, dst))
    print(f"[downscale_b] {len(tasks)} images from {src_root}", flush=True)
    ok = resized = failed = 0
    fails = []
    with ProcessPoolExecutor(max_workers=max(2, os.cpu_count() - 1)) as ex:
        for r in as_completed([ex.submit(process_one, t) for t in tasks]):
            res = r.result()
            if res["ok"]:
                ok += 1
                resized += int(res["resized"])
            else:
                failed += 1
                fails.append(res)
    print(f"[downscale_b] ok={ok} resized={resized} failed={failed}", flush=True)
    for f in fails[:10]:
        print(f"   FAIL {f['dst']}: {f['err']}", flush=True)
    # Non-zero exit on ANY failure OR when zero images were found, so the
    # orchestrator never deletes a source zip that didn't yield real output.
    if failed or ok == 0:
        sys.exit(1)
    sys.exit(0)

if __name__ == "__main__":
    main()
