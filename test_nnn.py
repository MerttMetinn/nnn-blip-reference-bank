"""Smoke test: do the environment (torch + CUDA) and the nnn-retrieval package work?

Uses no real data; runs one NNN search on random vectors.
Expected output: cuda 12.8 True, the GPU name, "nnn ok: (3, 5)".
Run: .venv\\Scripts\\python.exe test_nnn.py
"""
import numpy as np
import torch
from nnn import NNNRetriever, NNNRanker

print("torch:", torch.__version__, "cuda:", torch.version.cuda, torch.cuda.is_available())
print("gpu:", torch.cuda.get_device_name(0))


def random_unit_vectors(n, dim=64):
    """n random unit-length vectors (fake CLIP features)."""
    x = np.random.normal(size=(n, dim))
    return x / np.linalg.norm(x, axis=1, keepdims=True)


images = random_unit_vectors(500)    # fake gallery: 500 images
refs = random_unit_vectors(2000)     # fake reference query bank

# alternate_ks = k (number of nearest reference queries), alternate_weight = alpha
ranker = NNNRanker(NNNRetriever(64), images, refs,
                   alternate_ks=16, alternate_weight=0.75, batch_size=128)
scores, indices = ranker.search(refs[:3], top_k=5)   # 3 queries, top 5 images each
print("nnn ok:", indices.shape)
