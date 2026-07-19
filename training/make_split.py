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
        stem = r["out_stem"]
        if not os.path.exists(os.path.join(IMAGES, stem + ".jpg")):
            missing += 1
            continue
        # Relative + './'-prefixed so Ultralytics anchors each entry to this list
        # file's own directory (navier_merged/) -> portable across machines.
        buckets[r["orig_split"]].append(f"./images/{stem}.jpg")

    for split, key in [("train", "train"), ("valid", "val"), ("test", "test")]:
        out = os.path.join(ROOT, f"{key}.txt")
        with open(out, "w", encoding="utf-8", newline="\n") as f:
            f.write("\n".join(buckets[split]) + "\n")
        print(f"{key}.txt: {len(buckets[split])} images")
    if missing:
        print(f"WARNING: {missing} manifest rows had no image on disk")

    yaml_path = os.path.join(ROOT, "data_split.yaml")
    with open(yaml_path, "w", encoding="utf-8") as f:
        f.write("# Fixed split for model-size comparison (identical val across all models)\n")
        f.write("# 'path:' intentionally omitted -> Ultralytics resolves the dataset root to\n")
        f.write("# THIS file's own directory (navier_merged/), so the split is portable.\n")
        f.write("# List entries use './images/...' which anchor to this same directory.\n")
        f.write("train: train.txt\n")
        f.write("val: val.txt\n")
        f.write("test: test.txt\n\n")
        f.write(f"nc: {len(NAMES)}\n")
        f.write(f"names: {NAMES}\n")
    print(f"wrote {yaml_path}")

if __name__ == "__main__":
    main()
