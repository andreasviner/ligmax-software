#!/usr/bin/env python3
"""
Merge the navier_dataset30k YOLO splits (train/valid/test) into one consolidated
dataset, downscaling images to a max long-side and re-encoding to JPEG to save space.

- YOLO bbox coords are normalized [0,1], and we scale uniformly (preserve aspect,
  no letterboxing), so labels stay valid without modification.
- PNG -> JPG (quality 90). Images already <= MAX_SIDE are NOT upscaled, just re-encoded.
- Labels are paired by filename stem. Images with no label file are kept as
  background/negative samples (empty .txt written).
"""
import os, sys, csv, glob
from concurrent.futures import ProcessPoolExecutor, as_completed
from PIL import Image, ImageFile

ImageFile.LOAD_TRUNCATED_IMAGES = True  # be tolerant of slightly corrupt jpegs

SRC = r"_work/ds_a_raw/navier_dataset30k"
DST = r"navier_merged"
SPLITS = ["train", "valid", "test"]
MAX_SIDE = 1280
JPEG_Q = 90
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp")

def process_one(task):
    """task = (img_path, label_path_or_None, out_stem, split). Returns dict result."""
    img_path, label_path, out_stem, split = task
    out_img = os.path.join(DST, "images", out_stem + ".jpg")
    out_lbl = os.path.join(DST, "labels", out_stem + ".txt")
    res = {"stem": out_stem, "split": split, "ok": False,
           "had_label": label_path is not None, "resized": False,
           "orig_w": 0, "orig_h": 0, "err": ""}
    try:
        with Image.open(img_path) as im:
            im = im.convert("RGB")
            w, h = im.size
            res["orig_w"], res["orig_h"] = w, h
            m = max(w, h)
            if m > MAX_SIDE:
                scale = MAX_SIDE / m
                im = im.resize((max(1, round(w * scale)), max(1, round(h * scale))),
                               Image.LANCZOS)
                res["resized"] = True
            im.save(out_img, "JPEG", quality=JPEG_Q, optimize=True)
        # label: copy through, or write empty for background
        if label_path and os.path.exists(label_path):
            with open(label_path, "r", encoding="utf-8", errors="replace") as f:
                data = f.read()
            with open(out_lbl, "w", encoding="utf-8") as f:
                f.write(data)
        else:
            open(out_lbl, "w").close()
        res["ok"] = True
    except Exception as e:
        res["err"] = f"{type(e).__name__}: {e}"
    return res

def build_tasks():
    tasks = []
    seen = {}          # out_stem -> source (collision detection)
    orphan_labels = [] # labels with no image
    for split in SPLITS:
        img_dir = os.path.join(SRC, split, "images")
        lbl_dir = os.path.join(SRC, split, "labels")
        if not os.path.isdir(img_dir):
            continue
        imgs = [f for f in os.listdir(img_dir) if f.lower().endswith(IMG_EXTS)]
        img_stems = set()
        for fn in imgs:
            stem = os.path.splitext(fn)[0]
            img_stems.add(stem)
            lbl = os.path.join(lbl_dir, stem + ".txt")
            lbl = lbl if os.path.exists(lbl) else None
            out_stem = stem
            if out_stem in seen:  # cross-split filename collision
                out_stem = f"{split}__{stem}"
            seen[out_stem] = split
            tasks.append((os.path.join(img_dir, fn), lbl, out_stem, split))
        # detect label files with no matching image
        if os.path.isdir(lbl_dir):
            for lf in os.listdir(lbl_dir):
                if lf.lower().endswith(".txt") and os.path.splitext(lf)[0] not in img_stems:
                    orphan_labels.append(os.path.join(split, lf))
    return tasks, orphan_labels

def main():
    os.makedirs(os.path.join(DST, "images"), exist_ok=True)
    os.makedirs(os.path.join(DST, "labels"), exist_ok=True)
    tasks, orphan_labels = build_tasks()
    print(f"Total images to process: {len(tasks)}", flush=True)
    print(f"Orphan label files (no image): {len(orphan_labels)}", flush=True)

    results = []
    done = 0
    with ProcessPoolExecutor(max_workers=max(2, os.cpu_count() - 1)) as ex:
        futs = [ex.submit(process_one, t) for t in tasks]
        for fut in as_completed(futs):
            r = fut.result()
            results.append(r)
            done += 1
            if done % 2000 == 0:
                print(f"  ...{done}/{len(tasks)} done", flush=True)

    ok = [r for r in results if r["ok"]]
    failed = [r for r in results if not r["ok"]]
    no_label = [r for r in ok if not r["had_label"]]
    resized = [r for r in ok if r["resized"]]

    # manifest for provenance
    with open(os.path.join(DST, "manifest.csv"), "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(["out_stem", "orig_split", "had_label", "orig_w", "orig_h", "resized"])
        for r in sorted(ok, key=lambda x: x["stem"]):
            wr.writerow([r["stem"], r["split"], int(r["had_label"]),
                         r["orig_w"], r["orig_h"], int(r["resized"])])

    with open(os.path.join(DST, "data.yaml"), "w", encoding="utf-8") as f:
        f.write("# Consolidated Navier Njord buoy dataset (train/valid/test merged)\n")
        f.write("# Images downscaled to max 1280px long side, re-encoded JPEG q90.\n")
        f.write("path: .\n")
        f.write("train: images\n")
        f.write("val: images\n")
        f.write("\nnc: 6\n")
        f.write("names: ['green','red','north','east','south','west']\n")

    print("\n===== SUMMARY =====", flush=True)
    print(f"Processed OK      : {len(ok)}", flush=True)
    print(f"Failed            : {len(failed)}", flush=True)
    print(f"Background (no lbl): {len(no_label)}", flush=True)
    print(f"Downscaled        : {len(resized)}  (rest were already <= {MAX_SIDE}px)", flush=True)
    print(f"Orphan labels     : {len(orphan_labels)}", flush=True)
    if failed:
        print("\n-- failures (first 20) --", flush=True)
        for r in failed[:20]:
            print(f"  {r['split']}/{r['stem']}: {r['err']}", flush=True)
    if orphan_labels:
        with open(os.path.join(DST, "_orphan_labels.txt"), "w", encoding="utf-8") as f:
            f.write("\n".join(orphan_labels))
        print(f"(orphan label list -> {DST}/_orphan_labels.txt)", flush=True)

if __name__ == "__main__":
    main()
