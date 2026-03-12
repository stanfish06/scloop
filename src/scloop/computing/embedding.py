# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np
import scanpy as sc
from anndata import AnnData
from numba import jit
from pydantic.dataclasses import dataclass
from pynndescent import NNDescent

from ..data.constants import NUMERIC_EPSILON
from ..data.types import Count_t, Percent_t


def compute_diffmap(
    adata: AnnData,
    n_comps: int = 15,
    n_neighbors: int = 15,
    use_rep: str | None = None,
    key_added_neighbors: str = "neighbors_diffmap",
    random_state: int = 0,
) -> np.ndarray:
    sc.pp.neighbors(
        adata,
        n_neighbors=n_neighbors,
        use_rep=use_rep,
        method="gauss",
        random_state=random_state,
        key_added=key_added_neighbors,
    )
    sc.tl.diffmap(
        adata,
        n_comps=n_comps,
        neighbors_key=key_added_neighbors,
    )
    return np.array(adata.obsm["X_diffmap"])


@jit(nopython=True)
def compute_pairwise_adaptive_kernel_similarity(
    idx_nei: np.ndarray, dist_nei: np.ndarray
) -> np.ndarray:
    """Adaptive bandwidth Guassian kernel with density normalization
    Returns
    -------
    ndarray
    1D kernel distance array for upper diag entries of matrix nxn: k = n * i + j - ((i + 2)(i + 1)) // 2
    """
    vars_local = np.square(np.median(dist_nei, axis=1))
    n = idx_nei.shape[0]
    nn = idx_nei.shape[1]
    kernel_sim_raw = np.zeros(n * (n - 1) // 2)
    kernel_density = np.ones(n) * NUMERIC_EPSILON

    idx_a = np.empty(nn, dtype=idx_nei.dtype)
    idx_b = np.empty(nn, dtype=idx_nei.dtype)
    for i in range(n):
        idx_i_nei = idx_nei[i]
        dist_i_nei = dist_nei[i]
        vars_local_i_nei = vars_local[idx_i_nei]
        for ni in range(len(idx_i_nei)):
            j = idx_i_nei[ni]
            if i < j:
                idx_a[ni] = i
                idx_b[ni] = j
            else:
                idx_a[ni] = j
                idx_b[ni] = i
        idx_i_nei_global = n * idx_a + idx_b - ((idx_a + 2) * (idx_a + 1)) // 2
        # average of vars between two neighbors
        sum_vars_local = vars_local[i] + vars_local_i_nei
        kernel_sim_raw[idx_i_nei_global] = np.maximum(
            1
            / np.sqrt(sum_vars_local * 0.5)
            * np.exp(-np.square(dist_i_nei) / sum_vars_local),
            kernel_sim_raw[idx_i_nei_global],
        )

    for i in range(n):
        idx_i_nei = idx_nei[i]
        vars_local_i_nei = vars_local[idx_i_nei]
        for ni in range(len(idx_i_nei)):
            j = idx_i_nei[ni]
            if i < j:
                idx_a[ni] = i
                idx_b[ni] = j
            else:
                idx_a[ni] = j
                idx_b[ni] = i
        idx_i_nei_global = n * idx_a + idx_b - ((idx_a + 2) * (idx_a + 1)) // 2
        kernel_density[i] += np.sum(
            kernel_sim_raw[idx_i_nei_global] * np.sqrt(vars_local_i_nei)
        )

    emitted = np.zeros(len(kernel_sim_raw), dtype=np.bool_)
    rows = np.empty(n * nn, dtype=np.int64)
    cols = np.empty(n * nn, dtype=np.int64)
    vals = np.empty(n * nn, dtype=np.float64)
    count = 0
    for i in range(n):
        for ni in range(nn):
            j = idx_nei[i, ni]
            lo, hi = (i, j) if i < j else (j, i)
            k = n * lo + hi - ((lo + 2) * (lo + 1)) // 2
            if emitted[k]:
                continue
            emitted[k] = True
            rows[count] = lo
            cols[count] = hi
            vals[count] = kernel_sim_raw[k] / (kernel_density[lo] * kernel_density[hi])
            count += 1

    return rows[:count], cols[:count], vals[:count]


@dataclass
class DiffusionMap:
    eigenvalues: np.ndarray
    eigenvectors: np.ndarray
    n_neighbors: Count_t

    diffusion_t: Count_t
    damp_t: Percent_t
    _knn_index_cache: NNDescent | None = None

    def _compute_multi_step_transition():
        pass

    def _compute_multi_scale_eigenspace():
        pass

    def _compute_knn_index(self, emb: np.ndarray, cache: bool = False, **nn_kwargs):
        index = NNDescent(emb, n_neighbors=self.n_neighbors, **nn_kwargs)
        if cache:
            self._knn_index_cache = index

    def compute_diffmap():
        pass

    def project_query_data():
        pass
