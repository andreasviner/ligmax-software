#!/usr/bin/env python3
"""
Build a FIXED train/val/test split for navier_merged as Ultralytics list files.
Reuses the dataset's ORIGINAL split (from manifest.csv) so the validation set is
identical for every model we compare. No image copying -- just .txt path lists.
"""
import os, csv

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "navier_merged"))
IMAGES = os.path.join(ROOT, "images")
MANIFEST = os.path.join(ROOT, "manifest.csv")
NAMES = ["green", "red", "north", "east", "south", "west"]

def main():
    buckets = {"train": [], "valid": [], "test": []}
    missing = 0
    for r in csv.DictReader(open(MANIFEST, encoding="utf-8")):
        p = os.path.join(IMAGES, r["out_stem"] + ".jpg").replace("\\", "/")
        if not os.path.exists(p):
            missing += 1
            continue
        buckets[r["orig_split"]].append(p)

    for split, key in [("train", "train"), ("valid", "val"), ("test", "test")]:
        out = os.path.join(ROOT, f"{key}.txt")
        with open(out, "w", encoding="utf-8") as f:
            f.write("\n".join(buckets[split]))
        print(f"{key}.txt: {len(buckets[split])} images")
    if missing:
        print(f"WARNING: {missing} manifest rows had no image on disk")

    yaml_path = os.path.join(ROOT, "data_split.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write(f"# Fixed split for model-size comparison (identical val across all models)\n")
        f.write(f"path: {ROOT}\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("test: test.txt\n\n")
        f.write(f"nc: {len(NAMES)}\n")
        f.write(f"names: {NAMES}\n")
    print(f"wrote {yaml_path}")

if __name__ == "__main__":
    main()
