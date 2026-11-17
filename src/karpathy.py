"""Shared helper for reading the Karpathy split files.

Each image record is a dict; the fields we use:
  split      : "train" / "restval" / "val" / "test"
  filepath   : "train2014" or "val2014" for COCO
  filename   : image file name
  cocoid     : COCO image id (not present for Flickr; "imgid" is used there)
  sentences  : list of captions; each has "raw" (text) and "sentid" (id)
"""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]          # project root
KARPATHY_DIR = ROOT / "data" / "karpathy" / "extracted"
COCO_IMAGE_DIR = ROOT / "data" / "coco"             # contains val2014/
FLICKR_IMAGE_DIR = ROOT / "data" / "flickr30k" / "flickr30k-images"


def load_karpathy(name):
    """name: "coco" or "flickr30k". Returns the list of image records."""
    path = KARPATHY_DIR / f"dataset_{name}.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)["images"]


def image_id(record):
    """cocoid for COCO, imgid for Flickr."""
    return record["cocoid"] if "cocoid" in record else record["imgid"]
