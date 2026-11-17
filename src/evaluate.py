"""Measures the arms on the test set (Karpathy test, 5,000 images) (pipeline step 6).

Alpha and k are NOT chosen here; the locked values in results/tune/ are read.
The script stops if a tuning file is missing, so the test set never influences tuning.

Output (results/test/):
  <arm>_seed<seed>.json : Recall@1/5/10, alpha and k used, timings
  <arm>_seed<seed>.npz  : per-query rank and top-1 image; used for bootstrap
                          confidence intervals and the hubness analysis

Run: .venv\\Scripts\\python.exe src\\evaluate.py --arms A B C D E F
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np

from arms import eval_set, load_features, load_splits, reference_pool
from karpathy import ROOT
from nnn_core import nnn_bias, recall_at, retrieve, top_reference_sims


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arms", nargs="+", default=["A", "B", "C"])
    parser.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--features", default=None)
    parser.add_argument("--splits", default=None)
    parser.add_argument("--tune", default=str(ROOT / "results" / "tune"))
    parser.add_argument("--out", default=str(ROOT / "results" / "test"))
    args = parser.parse_args()

    feats, splits = load_features(args.features), load_splits(args.splits)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    queries, gallery, target = eval_set(feats, splits["coco_test"])
    print(f"test set: {len(queries)} queries, {len(gallery)} images")

    for arm in args.arms:
        # Arm A has no reference bank, so the seed has no effect; measured once.
        for seed in ([0] if arm == "A" else args.seeds):
            result = {"arm": arm, "seed": seed, "set": "coco_test (Karpathy test)",
                      "queries": len(queries), "gallery": len(gallery)}
            bias = None
            if arm != "A":
                tune_path = Path(args.tune) / f"{arm}_seed{seed}.json"
                if not tune_path.exists():
                    raise SystemExit(f"{tune_path} missing: run tune.py first")
                with open(tune_path, encoding="utf-8") as f:
                    best = json.load(f)["best"]
                refs = reference_pool(arm, seed, feats, splits)
                t = time.time()
                bias = nnn_bias(top_reference_sims(gallery, refs, k_max=best["k"]),
                                best["k"], best["alpha"])
                result.update(alpha=best["alpha"], k=best["k"],
                              reference_count=int(refs.shape[0]),
                              bias_seconds=round(time.time() - t, 3))
            t = time.time()
            rank, top = retrieve(queries, gallery, target, bias=bias)
            result["retrieve_seconds"] = round(time.time() - t, 3)
            recalls = recall_at(rank)
            result.update({f"r{k}": v for k, v in recalls.items()})

            name = f"{arm}_seed{seed}"
            with open(out_dir / f"{name}.json", "w", encoding="utf-8") as f:
                json.dump(result, f, indent=1)
            np.savez_compressed(out_dir / f"{name}.npz",
                                rank=rank.cpu().numpy().astype(np.int32),
                                top1=top[:, 0].cpu().numpy().astype(np.int32))
            extra = "" if arm == "A" else f" (alpha={result['alpha']}, k={result['k']})"
            print(f"arm {arm} seed {seed}: R@1={recalls[1]:.2f} R@5={recalls[5]:.2f} "
                  f"R@10={recalls[10]:.2f}{extra}")


if __name__ == "__main__":
    main()
