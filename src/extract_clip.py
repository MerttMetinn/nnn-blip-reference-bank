"""Computes CLIP image and text features once and caches them on disk (pipeline step 2).

Why a cache: running CLIP is the expensive part. Once features are saved, every
arm (A-F) and the alpha/k sweep are just matrix products.

Files written (features/, not in the repo):
  coco_img.npy        (40504, 512)   all Karpathy images in val2014
  coco_img_ids.npy    (40504,)       cocoid per row
  coco_txt.npy        (~202654, 512) all captions of these images
  coco_txt_meta.npz   sentid, image_id (per row)
  flickr_txt.npy      (145000, 512)  Flickr30k train captions (arm B)
  flickr_txt_meta.npz sentid, image_id

All vectors are L2-normalised, so a dot product is a cosine similarity. Which rows
belong to test, tuning or bank is decided by results/splits/splits.json, not here.

Quick trial:
  .venv\\Scripts\\python.exe src\\extract_clip.py --limit 200 --out features_test
Full run:
  .venv\\Scripts\\python.exe src\\extract_clip.py
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from huggingface_hub import snapshot_download
from PIL import Image
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import CLIPModel, CLIPProcessor

from karpathy import COCO_IMAGE_DIR, ROOT, image_id, load_karpathy

MODEL_ID = "openai/clip-vit-base-patch32"


class ImageDataset(Dataset):
    """Opens an image and applies the model's resize/normalisation."""

    def __init__(self, paths, processor):
        self.paths, self.processor = paths, processor

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        image = Image.open(self.paths[i]).convert("RGB")
        return self.processor(images=image, return_tensors="pt")["pixel_values"][0]


@torch.no_grad()
def encode_images(model, processor, paths, device, batch_size, workers):
    loader = DataLoader(ImageDataset(paths, processor), batch_size=batch_size, num_workers=workers)
    out = []
    for pixels in tqdm(loader, desc="images"):
        pooled = model.vision_model(pixel_values=pixels.to(device)).pooler_output
        feats = model.visual_projection(pooled)             # project to the shared 512-d space
        feats = feats / feats.norm(dim=-1, keepdim=True)    # unit length
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out)


@torch.no_grad()
def encode_texts(model, processor, texts, device, batch_size):
    out = []
    for start in tqdm(range(0, len(texts), batch_size), desc="texts"):
        tokens = processor.tokenizer(texts[start:start + batch_size], padding=True,
                                     truncation=True, max_length=77, return_tensors="pt").to(device)
        pooled = model.text_model(input_ids=tokens["input_ids"],
                                  attention_mask=tokens["attention_mask"]).pooler_output
        feats = model.text_projection(pooled)
        feats = feats / feats.norm(dim=-1, keepdim=True)
        out.append(feats.float().cpu().numpy())
    return np.concatenate(out)


def captions_of(records):
    """Returns (texts, sentids, image ids) for a list of records."""
    texts, sentids, image_ids = [], [], []
    for r in records:
        for s in r["sentences"]:
            texts.append(s["raw"].strip())
            sentids.append(s["sentid"])
            image_ids.append(image_id(r))
    return texts, np.array(sentids), np.array(image_ids)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default="features", help="output folder (relative to project root)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (trial run)")
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--workers", type=int, default=4, help="image loading workers")
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda"
    torch.cuda.reset_peak_memory_stats()
    started = time.time()

    model = CLIPModel.from_pretrained(MODEL_ID).to(device).eval()
    processor = CLIPProcessor.from_pretrained(MODEL_ID)

    # COCO: only images in val2014 (restval + val + test).
    coco = sorted((r for r in load_karpathy("coco") if r["filepath"] == "val2014"),
                  key=lambda r: r["cocoid"])
    flickr = sorted((r for r in load_karpathy("flickr30k") if r["split"] == "train"),
                    key=lambda r: r["imgid"])
    if args.limit:
        coco, flickr = coco[:args.limit], flickr[:args.limit]

    timings = {}

    t = time.time()
    paths = [COCO_IMAGE_DIR / r["filepath"] / r["filename"] for r in coco]
    img = encode_images(model, processor, paths, device, args.batch_size, args.workers)
    np.save(out_dir / "coco_img.npy", img)
    np.save(out_dir / "coco_img_ids.npy", np.array([r["cocoid"] for r in coco]))
    timings["coco_img_seconds"] = round(time.time() - t, 1)

    for name, records in (("coco", coco), ("flickr", flickr)):
        t = time.time()
        texts, sentids, image_ids = captions_of(records)
        txt = encode_texts(model, processor, texts, device, args.batch_size)
        np.save(out_dir / f"{name}_txt.npy", txt)
        np.savez(out_dir / f"{name}_txt_meta.npz", sentid=sentids, image_id=image_ids)
        timings[f"{name}_txt_seconds"] = round(time.time() - t, 1)
        timings[f"{name}_txt_count"] = len(texts)

    meta = {
        "model_id": MODEL_ID,
        # The folder name of the downloaded snapshot is the Hugging Face commit hash.
        "model_revision": Path(snapshot_download(MODEL_ID, local_files_only=True)).name,
        "torch": torch.__version__,
        "gpu": torch.cuda.get_device_name(0),
        "limit": args.limit,
        "batch_size": args.batch_size,
        "coco_img_count": int(img.shape[0]),
        "feature_dim": int(img.shape[1]),
        **timings,
        "total_seconds": round(time.time() - started, 1),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
    }
    with open(out_dir / "meta.json", "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    if args.limit is None:   # keep a record of the full run in the repo
        log_path = ROOT / "results" / "logs" / "extract_clip_meta.json"
        with open(log_path, "w", encoding="utf-8") as f:
            json.dump(meta, f, indent=2)
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
