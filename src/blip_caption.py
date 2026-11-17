"""Generates BLIP captions for reference-bank images (pipeline step 4, arms E and F).

Only bank images are processed; test and tuning images are never touched.
Decoding: beam search with 3 beams, at most 30 new tokens, no text prompt.
Beam search is deterministic, so no seed is needed.

Output:
  results/blip/<dataset>_captions.jsonl   one line per image: {"image_id": ..., "caption": ...}
  results/logs/blip_<dataset>_meta.jsonl  time, time per image, peak GPU memory
  (<dataset>: coco = arm E, flickr = arm F; chosen with --dataset)

If interrupted, the same command resumes (images already written are skipped).

Quick trial:
  .venv\\Scripts\\python.exe src\\blip_caption.py --limit 64 --out results\\blip_test
Full run (banks of all three seeds):
  .venv\\Scripts\\python.exe src\\blip_caption.py --seeds 0 1 2
  .venv\\Scripts\\python.exe src\\blip_caption.py --dataset flickr --seeds 0 1 2
"""
import argparse
import json
import time
from pathlib import Path

import torch
from huggingface_hub import snapshot_download
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BlipForConditionalGeneration, BlipProcessor

from arms import load_splits
from extract_clip import ImageDataset
from karpathy import COCO_IMAGE_DIR, FLICKR_IMAGE_DIR, ROOT, image_id, load_karpathy

MODEL_ID = "Salesforce/blip-image-captioning-base"
NUM_BEAMS = 3
MAX_NEW_TOKENS = 30


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["coco", "flickr"], default="coco",
                        help="coco: arm E bank, flickr: arm F bank")
    parser.add_argument("--seeds", nargs="+", type=int, default=[0])
    parser.add_argument("--out", default="results/blip", help="output folder (relative to project root)")
    parser.add_argument("--limit", type=int, default=None, help="only the first N images (trial run)")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--splits", default=None)
    args = parser.parse_args()

    out_dir = ROOT / args.out
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{args.dataset}_captions.jsonl"

    # Images to caption: union of the banks of the requested seeds.
    splits = load_splits(args.splits)
    wanted = sorted(set().union(*(splits[f"{args.dataset}_pool"][str(s)] for s in args.seeds)))
    if args.dataset == "coco":      # Flickr is a separate dataset and cannot overlap COCO test/tuning
        forbidden = set(splits["coco_test"]) | set(splits["coco_tune"])
        assert not set(wanted) & forbidden, "bank contains test/tuning images"

    done = set()
    if out_path.exists():
        with open(out_path, encoding="utf-8") as f:
            done = {json.loads(line)["image_id"] for line in f}
    todo = [i for i in wanted if i not in done]
    if args.limit:
        todo = todo[:args.limit]
    print(f"bank: {len(wanted)} images, done: {len(done)}, to caption: {len(todo)}")
    if not todo:
        return

    if args.dataset == "coco":
        record = {r["cocoid"]: r for r in load_karpathy("coco")}
        paths = [COCO_IMAGE_DIR / record[i]["filepath"] / record[i]["filename"] for i in todo]
    else:
        record = {image_id(r): r for r in load_karpathy("flickr30k")}
        paths = [FLICKR_IMAGE_DIR / record[i]["filename"] for i in todo]

    device = "cuda"
    processor = BlipProcessor.from_pretrained(MODEL_ID)
    model = BlipForConditionalGeneration.from_pretrained(MODEL_ID).to(device).eval()
    loader = DataLoader(ImageDataset(paths, processor), batch_size=args.batch_size,
                        num_workers=args.workers)

    torch.cuda.reset_peak_memory_stats()
    started = time.time()
    position = 0
    with open(out_path, "a", encoding="utf-8") as f, torch.no_grad():
        for pixels in tqdm(loader, desc="blip"):
            tokens = model.generate(pixel_values=pixels.to(device), num_beams=NUM_BEAMS,
                                    max_new_tokens=MAX_NEW_TOKENS)
            captions = processor.batch_decode(tokens, skip_special_tokens=True)
            for caption in captions:
                f.write(json.dumps({"image_id": todo[position], "caption": caption.strip()}) + "\n")
                position += 1
            f.flush()                      # keep written captions if interrupted
    seconds = time.time() - started

    meta = {
        "model_id": MODEL_ID,
        "model_revision": Path(snapshot_download(MODEL_ID, local_files_only=True)).name,
        "num_beams": NUM_BEAMS, "max_new_tokens": MAX_NEW_TOKENS, "prompt": None,
        "dataset": args.dataset, "seeds": args.seeds, "limit": args.limit, "batch_size": args.batch_size,
        "images_this_run": len(todo),
        "seconds_this_run": round(seconds, 1),
        "seconds_per_image": round(seconds / len(todo), 4),
        "peak_gpu_memory_mb": round(torch.cuda.max_memory_allocated() / 2**20, 1),
        "torch": torch.__version__, "gpu": torch.cuda.get_device_name(0),
    }
    # One line per run, so resumed runs stay visible.
    log_path = (ROOT / "results" / "logs" / f"blip_{args.dataset}_meta.jsonl") if args.limit is None \
        else out_dir / "meta.jsonl"
    with open(log_path, "a", encoding="utf-8") as f:
        f.write(json.dumps(meta) + "\n")
    print(json.dumps(meta, indent=2))


if __name__ == "__main__":
    main()
