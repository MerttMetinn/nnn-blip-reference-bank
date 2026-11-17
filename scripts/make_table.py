"""Builds the main results table from the files in results/test/ (pipeline step 7).

Runs no new measurement; it only reads what evaluate.py saved.

For every arm:
  - Recall@1/5/10: mean over seeds (and min/max over seeds for R@1)
  - 95% bootstrap confidence interval for R@1: queries resampled with replacement
    1,000 times (on the first seed)
  - Hubness: distribution of how many queries rank each image first
      max      : count of the most frequently top-ranked image
      skewness : skewness
      kurtosis : excess (Fisher) kurtosis, normal distribution = 0
      MAE      : mean absolute deviation of the counts from their mean
    Note: the paper (Section 3.2, Table 3) does not state the formulas, the rank
    cut-off (top 1 or top k) or the split for these measures. The definitions
    here are ours and are not directly comparable to the paper's numbers.
  - Recovery ratio: (R@1_E - R@1_B) / (R@1_C - R@1_B), also with D as denominator

Output: results/tables/main.md and results/tables/main.json
Run: .venv\\Scripts\\python.exe scripts\\make_table.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
ARM_NAMES = {
    "A": "Plain CLIP (no bank)",
    "B": "NNN, Flickr human captions",
    "C": "NNN, COCO human captions (5/img)",
    "D": "NNN, COCO human captions (1/img)",
    "E": "NNN, COCO BLIP captions (1/img)",
    "F": "NNN, Flickr BLIP captions (1/img)",
}
# Values PUBLISHED in the paper (Table 1); these are not our measurements.
PAPER_R1 = {"A": 30.43, "B": 33.45, "C": 37.53}


def bootstrap_ci(hit, n_boot=1000, seed=0):
    """hit: 0/1 per query. 95% confidence interval for recall (percent)."""
    rng = np.random.default_rng(seed)
    n = len(hit)
    means = np.array([hit[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5)


def paired_diff_ci(diff, n_boot=1000, seed=0):
    """diff: per-query (hit of arm a - hit of arm b). 95% CI of the difference (points)."""
    rng = np.random.default_rng(seed)
    n = len(diff)
    means = np.array([diff[rng.integers(0, n, n)].mean() for _ in range(n_boot)])
    return 100 * np.percentile(means, 2.5), 100 * np.percentile(means, 97.5)


def hubness(top1, gallery_size):
    counts = np.bincount(top1, minlength=gallery_size).astype(np.float64)
    centered = counts - counts.mean()
    std = counts.std()
    return {
        "max": int(counts.max()),
        "skewness": float((centered ** 3).mean() / std ** 3),
        "kurtosis": float((centered ** 4).mean() / std ** 4 - 3),
        "mae": float(np.abs(centered).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--test", default=str(ROOT / "results" / "test"))
    parser.add_argument("--out", default=str(ROOT / "results" / "tables"))
    args = parser.parse_args()
    test_dir, out_dir = Path(args.test), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = {}
    for arm in ARM_NAMES:
        files = sorted(test_dir.glob(f"{arm}_seed*.json"))
        if not files:
            continue
        runs = [json.load(open(f, encoding="utf-8")) for f in files]
        first = np.load(files[0].with_suffix(".npz"))
        low, high = bootstrap_ci((first["rank"] < 1).astype(np.float64))
        r1 = [r["r1"] for r in runs]
        rows[arm] = {
            "seeds": [r["seed"] for r in runs],
            "r1": float(np.mean(r1)), "r1_min": min(r1), "r1_max": max(r1),
            "r1_by_seed": {r["seed"]: r["r1"] for r in runs},
            "r5": float(np.mean([r["r5"] for r in runs])),
            "r10": float(np.mean([r["r10"] for r in runs])),
            "r1_ci95": [low, high],
            "alpha_k": [(r.get("alpha"), r.get("k")) for r in runs],
            "reference_count": runs[0].get("reference_count"),
            "hubness": hubness(first["top1"], runs[0]["gallery"]),
        }

    # Recovery ratio: computed per seed (same seed = same bank images), then averaged.
    recovery = {}
    if "E" in rows and "B" in rows:
        for upper in ("C", "D"):
            if upper not in rows:
                continue
            per_seed = {}
            for seed in rows["E"]["seeds"]:
                r = {a: rows[a]["r1_by_seed"].get(seed) for a in ("E", "B", upper)}
                if None not in r.values() and r[upper] != r["B"]:
                    per_seed[seed] = (r["E"] - r["B"]) / (r[upper] - r["B"])
            if per_seed:
                recovery[upper] = {"mean": float(np.mean(list(per_seed.values()))), "by_seed": per_seed}

    # Paired bootstrap: both arms are measured on the same queries, so the difference
    # is computed per query and resampled. If the interval excludes 0, the gap is not chance.
    paired = []
    for a, b in (("E", "B"), ("E", "D"), ("E", "C"), ("D", "C"), ("C", "B"), ("E", "F"), ("F", "B")):
        if a in rows and b in rows:
            seed = rows[a]["seeds"][0]
            pa, pb = test_dir / f"{a}_seed{seed}.npz", test_dir / f"{b}_seed{seed}.npz"
            if pb.exists():
                diff = (np.load(pa)["rank"] < 1).astype(np.float64) - (np.load(pb)["rank"] < 1)
                low, high = paired_diff_ci(diff)
                paired.append({"a": a, "b": b, "seed": seed, "diff": 100 * diff.mean(),
                               "ci95": [low, high]})

    lines = ["# Main results (OUR MEASUREMENTS)", "",
             "COCO Karpathy test, text-to-image retrieval, 5,000 images. "
             "Alpha and k selected on the tuning set (Karpathy val).", "",
             "| Arm | References | R@1 | R@1 95% CI | R@5 | R@10 | seeds | (alpha, k) |",
             "|---|---|---|---|---|---|---|---|"]
    for arm, r in rows.items():
        spread = "" if len(r["seeds"]) == 1 else f" [{r['r1_min']:.2f}-{r['r1_max']:.2f}]"
        params = "-" if arm == "A" else ", ".join(f"({a}, {k})" for a, k in r["alpha_k"])
        lines.append(f"| {arm}: {ARM_NAMES[arm]} | {r['reference_count'] or '-'} | {r['r1']:.2f}{spread} | "
                     f"{r['r1_ci95'][0]:.2f}-{r['r1_ci95'][1]:.2f} | {r['r5']:.2f} | {r['r10']:.2f} | "
                     f"{len(r['seeds'])} | {params} |")
    lines += ["", "R@1 is the mean over seeds; brackets give the min-max over seeds. "
              "The CI is a bootstrap over queries on the first seed (1,000 resamples).", ""]
    for upper, value in recovery.items():
        seeds = ", ".join(f"seed {s}: {v:.2f}" for s, v in value["by_seed"].items())
        lines.append(f"Recovery ratio (E-B)/({upper}-B): **{value['mean']:.2f}** ({seeds})")

    if paired:
        lines += ["", "## R@1 difference between arms (our measurements, paired bootstrap)", "",
                  "| Comparison | seed | Difference (points) | 95% CI |", "|---|---|---|---|"]
        for p in paired:
            lines.append(f"| {p['a']} - {p['b']} | {p['seed']} | {p['diff']:+.2f} | "
                         f"{p['ci95'][0]:+.2f} to {p['ci95'][1]:+.2f} |")

    lines += ["", "## Hubness (our measurements, first seed)", "",
              "| Arm | max | skewness | kurtosis | MAE |", "|---|---|---|---|---|"]
    for arm, r in rows.items():
        h = r["hubness"]
        lines.append(f"| {arm} | {h['max']} | {h['skewness']:.2f} | {h['kurtosis']:.2f} | {h['mae']:.2f} |")

    lines += ["", "## Values published in the paper (NOT our measurements)", "",
              "Source: NNN paper, Table 1 (CLIP, COCO, image Recall@1). "
              "The paper does not state the CLIP checkpoint or the sampling unit of the bank.", "",
              "| Arm | Paper R@1 | Our R@1 | Difference |", "|---|---|---|---|"]
    for arm, paper in PAPER_R1.items():
        ours = f"{rows[arm]['r1']:.2f}" if arm in rows else "-"
        diff = f"{rows[arm]['r1'] - paper:+.2f}" if arm in rows else "-"
        lines.append(f"| {arm} | {paper:.2f} | {ours} | {diff} |")
    lines += ["", "Paper hubness (CLIP, COCO; baseline -> NNN): kurtosis 59.8 -> 9.5; "
              "MAE 4.8 -> 2.6; max 162 -> 48."]

    sys.stdout.reconfigure(encoding="utf-8")     # avoid console encoding errors on Windows
    (out_dir / "main.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    with open(out_dir / "main.json", "w", encoding="utf-8") as f:
        json.dump({"arms": rows, "recovery": recovery, "paired": paired}, f, indent=1)
    print("\n".join(lines))


if __name__ == "__main__":
    main()
