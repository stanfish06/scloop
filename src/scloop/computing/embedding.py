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
    n = idx_nei.shape[0]
    nn = idx_nei.shape[1]

    vars_local = np.empty(n, dtype=np.float64)
    for i in range(n):
        vars_local[i] = np.square(np.median(dist_nei[i]))

    kernel_sim_raw = np.zeros(n * (n - 1) // 2)
    kernel_density = np.ones(n) * NUMERIC_EPSILON

    for i in range(n):
        for ni in range(nn):
            j = idx_nei[i, ni]
            dist = dist_nei[i, ni]
            lo, hi = (i, j) if i < j else (j, i)
            k = n * lo + hi - ((lo + 2) * (lo + 1)) // 2
            sum_var = vars_local[i] + vars_local[j]
            new_val = (1.0 / np.sqrt(sum_var * 0.5)) * np.exp(-(dist**2) / sum_var)
            if new_val > kernel_sim_raw[k]:
                kernel_sim_raw[k] = new_val

    for i in range(n):
        for ni in range(nn):
            j = idx_nei[i, ni]
            lo, hi = (i, j) if i < j else (j, i)
            k = n * lo + hi - ((lo + 2) * (lo + 1)) // 2
            kernel_density[i] += kernel_sim_raw[k] * np.sqrt(vars_local[j])

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
