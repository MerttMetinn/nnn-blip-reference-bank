"""Builds the fixed id lists used by every experiment (pipeline step 1).

Output: results/splits/splits.json
  coco_test          : 5,000 images  -> gallery and test queries (Karpathy test)
  coco_tune          : 5,000 images  -> alpha/k tuning (Karpathy val)
  coco_pool[seed]    : 22,657 images -> reference bank, drawn from restval only
  coco_pool_one_caption[seed] : one human caption per bank image (arm D)
  flickr_pool[seed]  : 5,800 images  -> 20% of Flickr30k train images (arms B, F)

Leakage rule: bank, test set and tuning set are disjoint; the script asserts this.

Run: .venv\\Scripts\\python.exe src\\make_splits.py
"""
import argparse
import json
from pathlib import Path

import numpy as np

from karpathy import ROOT, load_karpathy

SEEDS = [0, 1, 2]
COCO_POOL_SIZE = 22657     # the paper's 20%: 0.20 * 113,287 (train + restval)
FLICKR_POOL_SIZE = 5800    # 0.20 * 29,000 Flickr train images


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(ROOT / "results" / "splits" / "splits.json"))
    args = parser.parse_args()

    coco = load_karpathy("coco")
    flickr = load_karpathy("flickr30k")

    # Ids are sorted so that sampling depends only on the seed, not on JSON order.
    coco_test = sorted(r["cocoid"] for r in coco if r["split"] == "test")
    coco_tune = sorted(r["cocoid"] for r in coco if r["split"] == "val")
    restval = sorted(r["cocoid"] for r in coco if r["split"] == "restval")
    flickr_train = sorted(r["imgid"] for r in flickr if r["split"] == "train")

    # Caption ids of every COCO image (arm D picks one of them).
    sentids = {r["cocoid"]: sorted(s["sentid"] for s in r["sentences"]) for r in coco}

    coco_pool, one_caption, flickr_pool = {}, {}, {}
    for seed in SEEDS:
        rng = np.random.default_rng(seed)
        pool = sorted(rng.choice(restval, size=COCO_POOL_SIZE, replace=False).tolist())
        coco_pool[str(seed)] = pool
        # Arm D: same images, one random human caption per image.
        one_caption[str(seed)] = {str(i): int(rng.choice(sentids[i])) for i in pool}
        flickr_pool[str(seed)] = sorted(
            rng.choice(flickr_train, size=FLICKR_POOL_SIZE, replace=False).tolist())

    # Leakage checks.
    for seed, pool in coco_pool.items():
        assert set(pool) <= set(restval), "bank contains images outside restval"
        assert not set(pool) & set(coco_test), "bank overlaps the test set"
        assert not set(pool) & set(coco_tune), "bank overlaps the tuning set"
    assert not set(coco_test) & set(coco_tune), "test and tuning sets overlap"

    out = {
        "meta": {
            "seeds": SEEDS,
            "coco_pool_source": "Karpathy restval (val2014)",
            "coco_tune_source": "Karpathy val",
            "coco_test_source": "Karpathy test",
            "flickr_pool_source": "Karpathy Flickr30k train",
            "sampler": "numpy.random.default_rng(seed).choice, replace=False",
        },
        "coco_test": coco_test,
        "coco_tune": coco_tune,
        "coco_pool": coco_pool,
        "coco_pool_one_caption": one_caption,
        "flickr_pool": flickr_pool,
    }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(out, f)

    print("test images      :", len(coco_test))
    print("tuning images    :", len(coco_tune))
    print("restval total    :", len(restval))
    for seed in SEEDS:
        s = str(seed)
        print(f"seed {seed}: coco bank {len(coco_pool[s])} images, "
              f"one caption {len(one_caption[s])}, flickr bank {len(flickr_pool[s])} images")
    print("leakage check    : passed")
    print("written          :", out_path)


if __name__ == "__main__":
    main()
