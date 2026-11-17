"""Does our NNN code (src/nnn_core.py) rank exactly like the nnn-retrieval package?

Needs no data; uses random vectors. Takes a few seconds.
Run: .venv\\Scripts\\python.exe tests\\test_nnn_core.py
"""
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from nnn import NNNRanker, NNNRetriever  # noqa: E402
from nnn_core import nnn_bias, recall_at, retrieve, top_reference_sims  # noqa: E402

rng = np.random.default_rng(0)


def unit(n, d=64):
    x = rng.normal(size=(n, d)).astype(np.float32)
    return x / np.linalg.norm(x, axis=1, keepdims=True)


gallery, refs, queries = unit(500), unit(3000), unit(200)
K, ALPHA = 16, 0.75

# Package
ranker = NNNRanker(NNNRetriever(64), gallery, refs, alternate_ks=K, alternate_weight=ALPHA,
                   batch_size=128)
_, package_top = ranker.search(queries, top_k=5)
package_top = np.asarray(package_top)

# Our code
g, r, q = (torch.from_numpy(x).cuda() for x in (gallery, refs, queries))
bias = nnn_bias(top_reference_sims(g, r, k_max=64), k=K, alpha=ALPHA)
target = torch.zeros(len(queries), dtype=torch.long, device="cuda")   # unused in this test
_, our_top = retrieve(q, g, target, bias=bias, top_k=5)
our_top = our_top.cpu().numpy()

same = (package_top == our_top).mean()
print(f"share of identical top-5 rankings vs package: {same:.4f}")
assert same > 0.999, "our NNN code does not match the package"

# Recall: if every query is a copy of a gallery image, Recall@1 must be 100.
rank, _ = retrieve(g[:50], g, torch.arange(50, device="cuda"))
assert recall_at(rank)[1] == 100.0
# With zero bias the result must equal plain retrieval.
rank0, top0 = retrieve(q, g, target, bias=torch.zeros(500, device="cuda"), top_k=5)
rank1, top1 = retrieve(q, g, target, bias=None, top_k=5)
assert torch.equal(top0, top1) and torch.equal(rank0, rank1)
print("all checks passed")
