# Reused components

Every piece of code, model and data in this project that was made by others.

## Code

| Component | Source | Version | License | Use |
|---|---|---|---|---|
| `nnn-retrieval` | PyPI; https://multimodal-interpretability.csail.mit.edu/nnn/ | 1.0.1 | MIT | Reference NNN implementation; our bias code is checked against it |
| PyTorch, torchvision | https://pytorch.org | see `requirements.txt` (cu128) | BSD-3 | Running the models |
| Hugging Face `transformers`, `datasets` | https://github.com/huggingface | see `requirements.txt` | Apache-2.0 | Loading CLIP and BLIP |

## Models

| Model | Source | Revision | Use |
|---|---|---|---|
| CLIP ViT-B/32 | `openai/clip-vit-base-patch32` | `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268` | Image and text features |
| BLIP base captioning | `Salesforce/blip-image-captioning-base` | `82a37760796d32b1411fe092ab5d4e227313294b` | Reference captions (arms E, F) |

## Data

| Data | Source | Note |
|---|---|---|
| COCO val2014 images | http://images.cocodataset.org/zips/val2014.zip (6,645,013,297 bytes) | Not in the repo; terms: https://cocodataset.org/#termsofuse |
| Karpathy splits and captions | https://cs.stanford.edu/people/karpathy/deepimagesent/caption_datasets.zip (36,745,453 bytes) | `dataset_coco.json`, `dataset_flickr30k.json`; not in the repo |
| Flickr30k images | https://huggingface.co/datasets/nlphuji/flickr30k (`flickr30k-images.zip`, 4,390,240,817 bytes) | Arm F only; not in the repo. Original source: https://shannon.cs.illinois.edu/DenotationGraph/ |
| BLIP-generated captions | `results/blip/*.jsonl` (produced in this project) | Captions for the COCO and Flickr bank images |
