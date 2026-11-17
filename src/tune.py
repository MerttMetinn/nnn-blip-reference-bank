"""Selects alpha and k for every arm and seed on the tuning set (Karpathy val) (pipeline step 5).

Never touches the test set. Gallery: the 5,000 val images; queries: their captions.
Same sweep as the paper: alpha from 0.25 to 1.5 in steps of 0.125, k in powers of
two from 1 to 512; the setting with the highest Recall@1 is kept.

Output: results/tune/<arm>_seed<seed>.json (selected values + full grid)

Run: .venv\\Scripts\\python.exe src\\tune.py --arms B C D E F
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from arms import NNN_ARMS, eval_set, load_features, load_splits, reference_pool
from karpathy import ROOT
from nnn_core import nnn_bias, recall_at, retrieve, top_reference_sims

ALPHAS = [round(a, 3) for a in np.arange(0.25, 1.5 + 1e-9, 0.125)]
KS = [2 ** i for i in range(10)]            # 1, 2, 4, ..., 512


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=NNN_ARMS)
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--features", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--out", default=str(ROOT / "results" / "tune"))
    args = parser.parse_args()

    feats, splits = load_features(args.features), load_splits(args.splits)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    queries, gallery, target = eval_set(feats, splits["coco_tune"])
    rank, _ = retrieve(queries, gallery, target)
    base = recall_at(rank)
    print(f"tuning set: {len(queries)} queries, {len(gallery)} images")
    print(f"plain CLIP (tuning set) R@1 = {base[1]:.2f}")

    for arm in args.arms:
        for seed in args.seeds:
            t = time.time()
            refs = reference_pool(arm, seed, feats, splits)
            top_sims = top_reference_sims(gallery, refs, k_max=max(KS))
            grid, best = [], None
            for k in KS:
                for alpha in ALPHAS:
                    rank, _ = retrieve(queries, gallery, target, bias=nnn_bias(top_sims, k, alpha))
                    r1 = recall_at(rank)[1]
                    grid.append({"alpha": alpha, "k": k, "r1": r1})
                    if best is None or r1 > best["r1"]:
                        best = {"alpha": alpha, "k": k, "r1": r1}
            result = {
                "arm": arm, "seed": seed, "set": "coco_tune (Karpathy val)",
                "reference_count": int(refs.shape[0]),
                "baseline_r1": base[1],
                "best": best, "grid": grid,
                "seconds": round(time.time() - t, 1),
            }
            with open(out_dir / f"{arm}_seed{seed}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=1)
            print(f"arm {arm} seed {seed}: {refs.shape[0]} references, "
                  f"alpha={best['alpha']} k={best['k']} tuning R@1={best['r1']:.2f} "
                  f"({result['seconds']} s)")


if __name__ == "__main__":
    main()
