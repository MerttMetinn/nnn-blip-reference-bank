"""A short, self-written implementation of NNN and Recall@K.

Formula (paper, Section 2):
    b(r)      = alpha * (mean similarity of the k reference queries closest to image r)
    s_D(q, r) = s(q, r) - b(r)

All vectors are assumed unit length; similarity = dot product.
tests/test_nnn_core.py checks that it ranks exactly like the nnn-retrieval package.
"""
import torch


@torch.no_grad()
def top_reference_sims(gallery, refs, k_max, chunk=1000):
    """For every gallery image, the similarities of its k_max closest reference queries.

    Returns a (gallery size, k_max) tensor sorted in descending order. It is computed
    once; the bias for any k is the mean of the first k columns, so the alpha/k sweep
    never repeats this search. The gallery is processed in chunks to bound memory.
    """
    out = []
    for start in range(0, gallery.shape[0], chunk):
        sims = gallery[start:start + chunk] @ refs.T          # (chunk, number of references)
        out.append(sims.topk(k_max, dim=1).values)
    return torch.cat(out)


def nnn_bias(top_sims, k, alpha):
    """Per-image bias b(r). top_sims: output of top_reference_sims."""
    return alpha * top_sims[:, :k].mean(dim=1)


@torch.no_grad()
def retrieve(queries, gallery, target, bias=None, top_k=10, chunk=2000):
    """Text-to-image retrieval.

    queries : (num queries, d)   gallery : (num images, d)
    target  : gallery row of the correct image for every query
    bias    : (num images,) NNN bias; None means plain CLIP (arm A)

    Returns:
      rank : rank of the correct image for every query (0 = first)
      top  : gallery rows of the top_k highest-scoring images for every query
    """
    ranks, tops = [], []
    for start in range(0, queries.shape[0], chunk):
        scores = queries[start:start + chunk] @ gallery.T
        if bias is not None:
            scores = scores - bias[None, :]                   # subtract b(r) from each image column
        t = target[start:start + chunk]
        true_score = scores.gather(1, t[:, None])
        ranks.append((scores > true_score).sum(dim=1))        # images scoring above the correct one
        tops.append(scores.topk(top_k, dim=1).indices)
    return torch.cat(ranks), torch.cat(tops)


def recall_at(rank, ks=(1, 5, 10)):
    """Recall@K (percent): share of queries whose correct image is in the top K."""
    return {k: 100.0 * (rank < k).float().mean().item() for k in ks}
