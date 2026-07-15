#!/usr/bin/env python3
"""
Remap the navier_merged labels from 6 classes to 3.

Original classes (labels_6class_backup):
    0 green  1 red  2 north  3 east  4 south  5 west

New classes (labels):
    0 green  1 red  2 cardinal   # north/east/south/west merged into "cardinal"

The four cardinal-mark directions are collapsed into a single detector class;
direction is resolved downstream, not by the object detector.

Behaviour (safe to re-run):
  * On first run the current `labels/` dir is copied to `labels_6class_backup/`.
  * Every run then rebuilds `labels/` FROM the backup, so the source of truth is
    always the original 6-class annotations and the remap is deterministic.
  * `data.yaml` is rewritten to nc: 3 with the merged names.

Usage:
    python training/remap_6to3.py
"""
import os
import shutil

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "navier_merged"))
LABELS = os.path.join(ROOT, "labels")
BACKUP = os.path.join(ROOT, "labels_6class_backup")
DATA_YAML = os.path.join(ROOT, "data.yaml")

# old class id -> new class id
REMAP = {0: 0, 1: 1, 2: 2, 3: 2, 4: 2, 5: 2}
NEW_NAMES = ["green", "red", "cardinal"]


def remap_line(line):
    parts = line.split()
    if not parts:
        return None
    old = int(parts[0])
    if old not in REMAP:
        raise ValueError(f"unexpected class id {old!r} (expected 0-5)")
    parts[0] = str(REMAP[old])
    return " ".join(parts)


def main():
    if not os.path.isdir(LABELS):
        raise SystemExit(f"labels dir not found: {LABELS}")

    # Preserve the original 6-class labels once.
    if not os.path.isdir(BACKUP):
        shutil.copytree(LABELS, BACKUP)
        print(f"backed up original labels -> {BACKUP}")
    else:
        print(f"backup already exists, using it as source -> {BACKUP}")

    files = [f for f in os.listdir(BACKUP) if f.endswith(".txt")]
    changed = 0
    for name in files:
        src = os.path.join(BACKUP, name)
        dst = os.path.join(LABELS, name)
        out = []
        with open(src, encoding="utf-8") as f:
            for line in f:
                mapped = remap_line(line.strip())
                if mapped is not None:
                    out.append(mapped)
        with open(dst, "w", encoding="utf-8") as f:
            f.write("\n".join(out))
            if out:
                f.write("\n")
        changed += 1

    # Ultralytics caches label parsing; drop it so the new classes take effect.
    cache = os.path.join(ROOT, "labels.cache")
    if os.path.exists(cache):
        os.remove(cache)
        print("removed stale labels.cache")

    with open(DATA_YAML, "w", encoding="utf-8") as f:
        f.write("# Consolidated Navier Njord buoy dataset (train/valid/test merged)\n")
        f.write("# Images downscaled to max 1280px long side, re-encoded JPEG q90.\n")
        f.write("path: .\n")
        f.write("train: images\n")
        f.write("val: images\n\n")
        f.write(f"nc: {len(NEW_NAMES)}\n")
        f.write("# cardinal = north/east/south/west merged; differentiated downstream, not by the detector\n")
        f.write(f"names: {NEW_NAMES}\n")

    print(f"remapped {changed} label files -> 3 classes")
    print(f"rewrote {DATA_YAML}")


if __name__ == "__main__":
    main()
