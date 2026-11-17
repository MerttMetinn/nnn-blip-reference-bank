"""Loads cached features; builds the evaluation set and each arm's reference bank.

Arms:
  A : no reference bank (plain CLIP)
  B : Flickr30k human captions (5 per bank image)
  C : COCO human captions (5 per bank image)
  D : same COCO images, 1 human caption per image
  E : BLIP captions of the same COCO images, 1 per image (proposed)
  F : BLIP captions of the Flickr30k bank images, 1 per image (control)
"""
import json
from pathlib import Path

import numpy as np
import torch

from karpathy import ROOT

NNN_ARMS = ["B", "C", "D"]          # arms that need only human captions


def load_splits(path=None):
    path = Path(path) if path else ROOT / "results" / "splits" / "splits.json"
    with open(path, encoding="utf-8") as f:
        return json.load(f)


def load_features(feature_dir=None):
    """Returns the files in features/ as a dict of NumPy arrays."""
    d = Path(feature_dir) if feature_dir else ROOT / "features"
    coco_meta = np.load(d / "coco_txt_meta.npz")
    flickr_meta = np.load(d / "flickr_txt_meta.npz")
    blip = {}
    for name in ("coco", "flickr"):              # created by the BLIP step (arms E, F)
        if (d / f"blip_{name}_txt.npy").exists():
            blip[f"blip_{name}_txt"] = np.load(d / f"blip_{name}_txt.npy")
            blip[f"blip_{name}_txt_image"] = np.load(d / f"blip_{name}_txt_meta.npz")["image_id"]
    return {
        **blip,
        "coco_img": np.load(d / "coco_img.npy"),
        "coco_img_ids": np.load(d / "coco_img_ids.npy"),
        "coco_txt": np.load(d / "coco_txt.npy"),
        "coco_txt_sentid": coco_meta["sentid"],
        "coco_txt_image": coco_meta["image_id"],
        "flickr_txt": np.load(d / "flickr_txt.npy"),
        "flickr_txt_image": flickr_meta["image_id"],
    }


def eval_set(feats, image_ids, device="cuda"):
    """Builds (queries, gallery, target) for the given image ids.

    Gallery: the image vectors. Queries: all captions of these images.
    Target: gallery row of the correct image for every query.
    """
    image_ids = np.asarray(image_ids)
    row_of = {int(i): n for n, i in enumerate(feats["coco_img_ids"])}
    gallery = feats["coco_img"][[row_of[int(i)] for i in image_ids]]
    position = {int(i): n for n, i in enumerate(image_ids)}       # row in the gallery
    mask = np.isin(feats["coco_txt_image"], image_ids)
    queries = feats["coco_txt"][mask]
    target = np.array([position[int(i)] for i in feats["coco_txt_image"][mask]])
    to = lambda x: torch.from_numpy(x).to(device)
    return to(queries), to(gallery), to(target)


def reference_pool(arm, seed, feats, splits, device="cuda"):
    """Reference query vectors of an arm (torch tensor)."""
    s = str(seed)
    if arm == "B":
        mask = np.isin(feats["flickr_txt_image"], splits["flickr_pool"][s])
        refs = feats["flickr_txt"][mask]
    elif arm == "C":
        mask = np.isin(feats["coco_txt_image"], splits["coco_pool"][s])
        refs = feats["coco_txt"][mask]
    elif arm == "D":
        chosen = list(splits["coco_pool_one_caption"][s].values())
        mask = np.isin(feats["coco_txt_sentid"], chosen)
        refs = feats["coco_txt"][mask]
    elif arm in ("E", "F"):
        name = "coco" if arm == "E" else "flickr"
        pool = splits[f"{name}_pool"][s]
        mask = np.isin(feats[f"blip_{name}_txt_image"], pool)
        # Matched comparison: every bank image must have exactly one BLIP caption.
        assert mask.sum() == len(pool), (
            f"seed {seed} bank has {len(pool)} images but {mask.sum()} BLIP captions; "
            "run blip_caption.py and encode_blip.py for this seed")
        refs = feats[f"blip_{name}_txt"][mask]
    else:
        raise ValueError(f"no reference bank defined for arm {arm}")
    return torch.from_numpy(refs).to(device)
