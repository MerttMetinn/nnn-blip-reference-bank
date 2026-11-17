"""Failure analysis: where does E (BLIP bank) differ from D (human-caption bank)?

Runs no new measurement; reads per-query ranks in results/test/, the BLIP captions
and the Karpathy JSON. Uses seed 0 (D and E share the same bank images).

Writes three sections to results/tables/failure.md:
  1. Query level: how many queries D gets right and E wrong, and vice versa, with examples
  2. Hub images: images most often ranked first under E, with their counts under A and D
  3. Caption text: length, vocabulary, repetition of BLIP vs human captions,
     most frequent BLIP captions

Run: .venv\\Scripts\\python.exe scripts\\failure_analysis.py
"""
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from karpathy import load_karpathy  # noqa: E402

SEED = 0
N_EXAMPLES = 10


def main():
    splits = json.load(open(ROOT / "results" / "splits" / "splits.json", encoding="utf-8"))
    coco = load_karpathy("coco")
    text_of = {s["sentid"]: s["raw"].strip() for r in coco for s in r["sentences"]}
    first_caption = {r["cocoid"]: r["sentences"][0]["raw"].strip() for r in coco}

    # Test queries are ordered exactly as in evaluate.py (arms.eval_set).
    meta = np.load(ROOT / "features" / "coco_txt_meta.npz")
    test_ids = np.asarray(splits["coco_test"])
    mask = np.isin(meta["image_id"], test_ids)
    query_sentid, query_image = meta["sentid"][mask], meta["image_id"][mask]

    runs = {a: np.load(ROOT / "results" / "test" / f"{a}_seed{SEED}.npz") for a in "ADE"}
    hit = {a: runs[a]["rank"] < 1 for a in runs}
    lines = ["# Failure analysis (OUR MEASUREMENTS, seed 0, COCO Karpathy test)", ""]

    # 1. Query level
    d_only, e_only = hit["D"] & ~hit["E"], hit["E"] & ~hit["D"]
    lines += ["## 1. D and E at query level", "",
              "| Case | Queries | Share |", "|---|---|---|"]
    for name, m in (("Both correct", hit["D"] & hit["E"]), ("Only D correct", d_only),
                    ("Only E correct", e_only), ("Both wrong", ~hit["D"] & ~hit["E"])):
        lines.append(f"| {name} | {int(m.sum())} | {100 * m.mean():.2f}% |")
    rng = np.random.default_rng(0)
    for title, m in (("Only D correct (lost by E)", d_only),
                     ("Only E correct (gained by E)", e_only)):
        lines += ["", f"### {N_EXAMPLES} random examples: {title}", "",
                  "| Query | Correct image | E's top-1 image (its first caption) | Rank of correct image under E |",
                  "|---|---|---|---|"]
        for q in rng.choice(np.flatnonzero(m), size=min(N_EXAMPLES, int(m.sum())), replace=False):
            top = int(test_ids[runs["E"]["top1"][q]])
            lines.append(f"| {text_of[int(query_sentid[q])]} | {int(query_image[q])} | "
                         f"{top}: {first_caption[top]} | {int(runs['E']['rank'][q]) + 1} |")

    # 2. Hub images
    count = {a: np.bincount(runs[a]["top1"], minlength=len(test_ids)) for a in runs}
    lines += ["", "## 2. The 10 images most often ranked first under E", "",
              "Each image has about 5 correct captions; a count far above 5 marks a hub.", "",
              "| Image (cocoid) | First human caption | A | D | E |", "|---|---|---|---|---|"]
    for g in np.argsort(-count["E"])[:10]:
        cid = int(test_ids[g])
        lines.append(f"| {cid} | {first_caption[cid]} | {count['A'][g]} | {count['D'][g]} | {count['E'][g]} |")
    for a in "ADE":
        lines.append("")
        lines.append(f"Arm {a}: images never ranked first {int((count[a] == 0).sum())}, "
                     f"images ranked first for more than 20 queries {int((count[a] > 20).sum())}.")

    # 3. Caption text (seed 0 bank: the same 22,657 images)
    pool = set(splits["coco_pool"][str(SEED)])
    with open(ROOT / "results" / "blip" / "coco_captions.jsonl", encoding="utf-8") as f:
        blip = [r["caption"] for r in map(json.loads, f) if r["image_id"] in pool]
    real = [text_of[s] for s in splits["coco_pool_one_caption"][str(SEED)].values()]
    lines += ["", "## 3. Caption text in the bank (same 22,657 images, 1 caption per image)", "",
              "| Measure | Human (arm D) | BLIP (arm E) |", "|---|---|---|"]
    stats = {}
    for name, caps in (("real", real), ("blip", blip)):
        words = [c.lower().rstrip(".").split() for c in caps]
        freq = Counter(caps)
        stats[name] = {
            "n": len(caps), "distinct": len(freq),
            "repeated": sum(v for v in freq.values() if v > 1),
            "mean_words": float(np.mean([len(w) for w in words])),
            "vocab": len({t for w in words for t in w}),
        }
    for label, key, fmt in (("Captions", "n", "{}"), ("Distinct captions", "distinct", "{}"),
                            ("Captions identical to another image's", "repeated", "{}"),
                            ("Mean words per caption", "mean_words", "{:.2f}"),
                            ("Vocabulary size", "vocab", "{}")):
        lines.append(f"| {label} | {fmt.format(stats['real'][key])} | {fmt.format(stats['blip'][key])} |")
    lines += ["", "### 10 most frequent BLIP captions", "", "| Caption | Images |", "|---|---|"]
    for caption, n in Counter(blip).most_common(10):
        lines.append(f"| {caption} | {n} |")

    out = ROOT / "results" / "tables" / "failure.md"
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    sys.stdout.reconfigure(encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
