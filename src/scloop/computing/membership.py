# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.sparse import csr_matrix
from sklearn.neighbors import NearestNeighbors

from ..data.constants import DEFAULT_ALPHA_MEMBERSHIP, DEFAULT_N_NEIGHBORS_MEMBERSHIP


def compute_loop_membership(
    embedding: np.ndarray,
    representatives: Sequence[Sequence[Sequence[int]] | None],
    scales: Sequence[float],
    n_neighbors: int = DEFAULT_N_NEIGHBORS_MEMBERSHIP,
    alpha: float = DEFAULT_ALPHA_MEMBERSHIP,
    tol: float = 1e-7,
    max_iter: int = 1000,
) -> np.ndarray:
    n_cells = embedding.shape[0]
    membership = np.zeros((n_cells, len(representatives)), dtype=np.float32)
    if n_cells < 2:
        return membership

    k = min(n_neighbors, n_cells - 1)
    nn = NearestNeighbors(n_neighbors=k, n_jobs=-1).fit(embedding)
    knn_dist, knn_idx = nn.kneighbors()
    rows = np.repeat(np.arange(n_cells), k)
    cols = knn_idx.ravel()
    dists = knn_dist.ravel()

    for j, (reps, scale) in enumerate(zip(representatives, scales)):
        if not reps:
            continue
        keep = dists <= scale
        adj = csr_matrix(
            (np.ones(int(keep.sum())), (rows[keep], cols[keep])),
            shape=(n_cells, n_cells),
        )
        adj = adj.maximum(adj.T).tocsr()
        deg = np.asarray(adj.sum(axis=1)).ravel()
        inv_deg = np.divide(1.0, deg, out=np.zeros_like(deg), where=deg > 0)

        seeds = np.unique(np.concatenate([np.asarray(r) for r in reps]))
        restart = np.zeros(n_cells)
        restart[seeds] = deg[seeds]
        if restart.sum() == 0:
            continue
        restart /= restart.sum()

        ppr = restart.copy()
        for _ in range(max_iter):
            ppr_next = alpha * restart + (1 - alpha) * (adj @ (inv_deg * ppr))
            converged = np.abs(ppr_next - ppr).sum() < tol
            ppr = ppr_next
            if converged:
                break

        score = ppr * inv_deg
        ref = np.median(score[seeds])
        if ref > 0:
            membership[:, j] = np.minimum(score / ref, 1.0)
    return membership
