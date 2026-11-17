"""Builds the report figures. Runs no new measurement; reads files in results/.

  report/figures/recall_by_arm.png : Recall@1 of every arm with its 95% bootstrap interval
  report/figures/hubness.png       : distribution of how often each image is ranked first

Run scripts\\make_table.py first (results/tables/main.json).
Run: .venv\\Scripts\\python.exe scripts\\make_figures.py
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "report" / "figures"

# Colour = domain of the reference bank.
DOMAIN_COLOR = {"No bank": "#898781", "Flickr30k bank (out-of-domain)": "#eb6834",
                "COCO bank (in-domain)": "#2a78d6"}
ARMS = {  # arm: (label, domain)
    "A": ("A  CLIP, no NNN", "No bank"),
    "B": ("B  Flickr human captions (5/img)", "Flickr30k bank (out-of-domain)"),
    "F": ("F  Flickr BLIP captions (1/img)", "Flickr30k bank (out-of-domain)"),
    "E": ("E  COCO BLIP captions (1/img)", "COCO bank (in-domain)"),
    "D": ("D  COCO human captions (1/img)", "COCO bank (in-domain)"),
    "C": ("C  COCO human captions (5/img)", "COCO bank (in-domain)"),
}
INK, MUTED, GRID = "#0b0b0b", "#52514e", "#e1e0d9"

plt.rcParams.update({"font.family": "sans-serif", "font.size": 10, "text.color": INK,
                     "axes.edgecolor": "#c3c2b7", "axes.labelcolor": MUTED,
                     "xtick.color": MUTED, "ytick.color": MUTED,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "figure.facecolor": "white", "axes.facecolor": "white"})


def recall_figure(table):
    arms = [a for a in ARMS if a in table["arms"]]
    fig, ax = plt.subplots(figsize=(7.2, 3.4))
    for y, arm in enumerate(arms):
        row, (label, domain) = table["arms"][arm], ARMS[arm]
        low, high = row["r1_ci95"]
        color = DOMAIN_COLOR[domain]
        ax.plot([low, high], [y, y], color=color, linewidth=2, solid_capstyle="round")
        ax.plot(row["r1"], y, "o", color=color, markersize=8, markeredgecolor="white",
                markeredgewidth=1.5)
        ax.text(high + 0.2, y, f"{row['r1']:.2f}", va="center", color=INK)
    ax.set_yticks(range(len(arms)), [ARMS[a][0] for a in arms], color=INK)
    # Dot: mean over seeds. Line: 95% bootstrap interval on the first seed (state in caption).
    ax.set_xlabel("Image Recall@1 on COCO Karpathy test (%)")
    ax.set_xlim(29, 40)
    ax.grid(axis="x", color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)
    ax.spines["left"].set_visible(False)
    ax.tick_params(axis="y", length=0)
    handles = [plt.Line2D([], [], marker="o", color=c, linewidth=2, markersize=7, label=d)
               for d, c in DOMAIN_COLOR.items()]
    fig.legend(handles=handles, loc="upper center", frameon=False, fontsize=9, ncol=3,
               handlelength=1.5, columnspacing=1.5)
    fig.tight_layout(rect=(0, 0, 1, 0.92))
    fig.savefig(OUT / "recall_by_arm.png", dpi=200)
    plt.close(fig)


def hubness_figure():
    panels = [("A", "A  CLIP, no NNN"), ("B", "B  Flickr human"),
              ("E", "E  COCO BLIP"), ("D", "D  COCO human (1/img)")]
    fig, axes = plt.subplots(1, 4, figsize=(9.6, 2.6), sharex=True, sharey=True)
    bins = np.arange(0, 225, 5)
    for ax, (arm, title) in zip(axes, panels):
        path = ROOT / "results" / "test" / f"{arm}_seed0.npz"
        if not path.exists():
            continue
        counts = np.bincount(np.load(path)["top1"], minlength=5000)
        ax.hist(counts, bins=bins, color="#2a78d6", edgecolor="white", linewidth=0.4)
        ax.set_yscale("log")
        ax.set_title(title, fontsize=10, color=INK, loc="left")
        ax.text(0.97, 0.9, f"max = {counts.max()}", transform=ax.transAxes, ha="right", color=INK)
        ax.grid(axis="y", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)
    axes[0].set_ylabel("Number of images (log)")
    fig.supxlabel("Queries for which the image is ranked first (5 are correct per image)",
                  fontsize=10, color=MUTED)
    fig.tight_layout()
    fig.savefig(OUT / "hubness.png", dpi=200)
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    with open(ROOT / "results" / "tables" / "main.json", encoding="utf-8") as f:
        recall_figure(json.load(f))
    hubness_figure()
    print("written:", OUT)


if __name__ == "__main__":
    main()
