"""Encodes BLIP captions with the CLIP text encoder (reference banks of arms E and F).

Input : results/blip/<dataset>_captions.jsonl   (--dataset coco: arm E, flickr: arm F)
Output: features/blip_<dataset>_txt.npy and features/blip_<dataset>_txt_meta.npz (image_id)

Uses the same CLIP model and the same code as for human captions
(extract_clip.encode_texts), so the only difference between arms is the text source.

Run: .venv\\Scripts\\python.exe src\\encode_blip.py --dataset coco
"""
import argparse
import json

import numpy as np
from transformers import CLIPModel, CLIPProcessor

from extract_clip import MODEL_ID, encode_texts
from karpathy import ROOT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["coco", "flickr"], default="coco")
    parser.add_argument("--out", default="features")
    parser.add_argument("--batch-size", type=int, default=256)
    args = parser.parse_args()

    with open(ROOT / "results" / "blip" / f"{args.dataset}_captions.jsonl", encoding="utf-8") as f:
        rows = [json.loads(line) for line in f]
    texts = [r["caption"] for r in rows]
    image_ids = np.array([r["image_id"] for r in rows])

    model = CLIPModel.from_pretrained(MODEL_ID).to("cuda").eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)
    feats = encode_texts(model, processor, texts, "cuda", args.batch_size)

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / f"blip_{args.dataset}_txt.npy", feats)
    np.savez(out_dir / f"blip_{args.dataset}_txt_meta.npz", image_id=image_ids)
    print(f"{len(texts)} captions, distinct: {len(set(texts))}, shape: {feats.shape}")


if __name__ == "__main__":
    main()
