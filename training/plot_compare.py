#!/usr/bin/env python3
"""
Plot YOLO26 size-comparison charts from train_compare.py output.
Reads runs/compare/_summary/compare_results.json and writes PNGs alongside it.

Charts (each single-axis, CVD-safe fixed colors, legend + direct labels):
  1. accuracy_vs_latency.png  -- mAP50-95 vs inference ms  (deployment decision)
  2. accuracy_vs_params.png   -- mAP50-95 vs params (M)
  3. map_bars.png             -- mAP50 & mAP50-95 per model
  4. per_class_map.png        -- per-class mAP50-95, models as series
"""
import os, json, argparse
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# Okabe-Ito: a published colorblind-safe categorical palette. Fixed order by size.
SIZE_COLOR = {"n": "#009E73", "s": "#0072B2", "m": "#E69F00",
              "l": "#D55E00", "x": "#CC79A7"}
SIZE_ORDER = ["n", "s", "m", "l", "x"]
INK = "#222222"; MUTED = "#666666"; GRID = "#DDDDDD"

def size_of(model):  # "yolo26s" -> "s"
    return model.replace("yolo26", "")

def style(ax):
    ax.grid(True, color=GRID, linewidth=0.8, zorder=0)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color(MUTED)
    ax.tick_params(colors=MUTED)
    for lab in ax.get_xticklabels() + ax.get_yticklabels():
        lab.set_color(INK)

def norm_names(names):
    """Ultralytics returns names as a dict {0:'green',...} (keys may be str after
    JSON round-trip). Coerce to an index-ordered list; pass lists through."""
    if isinstance(names, dict):
        return [names[k] for k in sorted(names, key=lambda x: int(x))]
    return names

def load(path):
    d = json.load(open(path, encoding="utf-8"))
    rows = [r for r in d["rows"] if not r.get("error") and r.get("mAP50-95") not in (None, "")]
    rows.sort(key=lambda r: SIZE_ORDER.index(size_of(r["model"])))
    return norm_names(d.get("class_names")), rows

def scatter(rows, xkey, xlabel, fname, out, title):
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    style(ax)
    for r in rows:
        s = size_of(r["model"])
        ax.scatter(r[xkey], r["mAP50-95"], s=120, color=SIZE_COLOR.get(s, INK),
                   zorder=3, edgecolor="white", linewidth=1.5)
        ax.annotate(r["model"], (r[xkey], r["mAP50-95"]),
                    textcoords="offset points", xytext=(8, 6),
                    color=INK, fontsize=10, fontweight="bold")
    # connect in size order to show the trade-off frontier
    ax.plot([r[xkey] for r in rows], [r["mAP50-95"] for r in rows],
            color=MUTED, linewidth=1.5, linestyle="--", zorder=2, alpha=0.7)
    ax.set_xlabel(xlabel, color=INK); ax.set_ylabel("mAP@50-95", color=INK)
    ax.set_title(title, color=INK, fontweight="bold", loc="left")
    fig.tight_layout(); fig.savefig(os.path.join(out, fname)); plt.close(fig)
    print("wrote", fname)

def map_bars(rows, out):
    import numpy as np
    fig, ax = plt.subplots(figsize=(7, 5), dpi=140)
    style(ax)
    x = np.arange(len(rows)); w = 0.38
    b1 = ax.bar(x - w/2, [r["mAP50"] for r in rows], w, label="mAP@50",
                color="#0072B2", zorder=3)
    b2 = ax.bar(x + w/2, [r["mAP50-95"] for r in rows], w, label="mAP@50-95",
                color="#E69F00", zorder=3)
    for bars in (b1, b2):
        for bar in bars:
            ax.annotate(f"{bar.get_height():.3f}", (bar.get_x()+bar.get_width()/2, bar.get_height()),
                        textcoords="offset points", xytext=(0, 3), ha="center",
                        fontsize=8, color=INK)
    ax.set_xticks(x); ax.set_xticklabels([r["model"] for r in rows])
    ax.set_ylabel("mAP", color=INK)
    ax.set_title("Accuracy by model size", color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, labelcolor=INK)
    fig.tight_layout(); fig.savefig(os.path.join(out, "map_bars.png")); plt.close(fig)
    print("wrote map_bars.png")

def per_class(names, rows, out):
    import numpy as np
    pc = [r for r in rows if r.get("per_class_mAP50-95")]
    if not names or not pc:
        print("skip per_class (no data)"); return
    x = np.arange(len(names)); n = len(pc); w = 0.8 / n
    fig, ax = plt.subplots(figsize=(9, 5), dpi=140)
    style(ax)
    for i, r in enumerate(pc):
        s = size_of(r["model"])
        ax.bar(x + (i - (n-1)/2)*w, r["per_class_mAP50-95"], w,
               label=r["model"], color=SIZE_COLOR.get(s, INK), zorder=3)
    ax.set_xticks(x); ax.set_xticklabels(names)
    ax.set_ylabel("mAP@50-95", color=INK)
    ax.set_title("Per-class accuracy by model size", color=INK, fontweight="bold", loc="left")
    ax.legend(frameon=False, labelcolor=INK, ncol=len(pc))
    fig.tight_layout(); fig.savefig(os.path.join(out, "per_class_map.png")); plt.close(fig)
    print("wrote per_class_map.png")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--summary", default=os.path.abspath(os.path.join(
        os.path.dirname(__file__), "..", "runs", "compare", "_summary")))
    args = ap.parse_args()
    jpath = os.path.join(args.summary, "compare_results.json")
    names, rows = load(jpath)
    if not rows:
        print("No successful model rows to plot yet."); return
    out = args.summary
    scatter(rows, "infer_ms", "Inference latency (ms/img)  ← faster",
            "accuracy_vs_latency.png", out, "Accuracy vs latency (deployment trade-off)")
    scatter(rows, "params_M", "Parameters (millions)",
            "accuracy_vs_params.png", out, "Accuracy vs model size")
    map_bars(rows, out)
    per_class(names, rows, out)
    print("\nAll plots ->", out)

if __name__ == "__main__":
    main()
