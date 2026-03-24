# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Literal

import numpy as np
import scanpy as sc
from anndata import AnnData
from numba import jit
from pydantic.dataclasses import dataclass
from pynndescent import NNDescent
from scipy.sparse import csr_matrix, diags

from ..data.constants import NUMERIC_EPSILON
from ..data.types import Count_t, Percent_t
from .utils import compute_sparse_eigendecomposition


def compute_diffmap(
    adata: AnnData,
    n_comps: int = 15,
    n_neighbors: int = 15,
    use_rep: str | None = None,
    key_added_neighbors: str = "neighbors_diffmap",
    flavor: Literal["scanpy", "custom"] = "custom",
    random_state: int = 0,
    *,
    damp_multistep_diffusion: Percent_t = 1.0,
    use_multistep_eigenvalues: bool = True,
) -> np.ndarray:
    match flavor:
        case "scanpy":
            sc.pp.neighbors(
                adata,
                n_neighbors=n_neighbors,
                use_rep=f"X_{use_rep}" if use_rep is not None else None,
                method="gauss",
                random_state=random_state,
                key_added=key_added_neighbors,
            )
            sc.tl.diffmap(
                adata,
                n_comps=n_comps,
                neighbors_key=key_added_neighbors,
            )
        case "custom":
            diffmap = DiffusionMap(
                n_neighbors=n_neighbors, damp_multistep=damp_multistep_diffusion
            )
            # TODO: better input handling
            emb = adata.obsm[f"X_{use_rep}"] if use_rep is not None else adata.X
            assert emb is not None and type(emb) is np.ndarray
            diffmap.compute_multi_step_eigenspace(emb=emb, ndim_eigenspace=n_comps)
            eigvals = (
                diffmap.eigenvalues_multistep
                if use_multistep_eigenvalues
                else diffmap.eigenvalues
            )
            assert eigvals is not None
            eigvals[eigvals < 0] = NUMERIC_EPSILON
            eigvals /= eigvals.max()
            eigvecs = diffmap.eigenvectors
            assert eigvecs is not None
            diffmap.diffmap_coords = eigvecs * eigvals
            adata.obsm["X_diffmap"] = diffmap.diffmap_coords
            adata.unx["diffmap_evals"] = eigvals

    return np.array(adata.obsm["X_diffmap"])


@jit(nopython=True, cache=True)
def compute_knn_diffusion_projection(
    idx_nei: np.ndarray,
    dist_nei: np.ndarray,
    diffmap_coords_reference: np.ndarray,
    vars_local_reference: np.ndarray,
) -> np.ndarray:
    n = idx_nei.shape[0]
    nn = idx_nei.shape[1]
    p = diffmap_coords_reference.shape[1]
    emb_query = np.zeros((n, p))
    for i in range(n):
        weight_total = 0.0
        emb_i = np.zeros(p)
        for ni in range(nn):
            j = idx_nei[i, ni]
            d2 = dist_nei[i, ni] ** 2
            weight = np.exp(-d2 / (2.0 * vars_local_reference[j] + NUMERIC_EPSILON))
            weight_total += weight
            emb_i += diffmap_coords_reference[j] * weight
        emb_query[i] = emb_i / (weight_total + NUMERIC_EPSILON)
    return emb_query


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


@dataclass(config=dict(arbitrary_types_allowed=True))
class DiffusionMap:
    n_neighbors: Count_t
    damp_multistep: Percent_t = 1.0
    eigenvalues: np.ndarray | None = None
    eigenvalues_multistep: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    diffmap_coords: np.ndarray | None = None
    _knn_index_cache: NNDescent | None = None
    _d_inv_sqrt: np.ndarray | None = None
    _vars_local: np.ndarray | None = None

    def _compute_knn_index(
        self, emb: np.ndarray, cache: bool = False, query: bool = False, **nn_kwargs
    ):
        # need to use n_neighbors + 1 because neighbor graph contains self edges
        index = NNDescent(emb, n_neighbors=self.n_neighbors + 1, **nn_kwargs)
        if query:
            # this makes new data query faster, but not needed for getting nn for training data
            index.prepare()
        if cache:
            self._knn_index_cache = index
        return index

    def _compute_one_step_transition(self, emb: np.ndarray, **nn_kwargs) -> csr_matrix:
        n = emb.shape[0]
        knn_index = self._compute_knn_index(
            emb=emb, cache=False, query=False, **nn_kwargs
        )
        dist_nei = knn_index.neighbor_graph[1][:, 1:]
        self._vars_local = np.array([np.median(dist_nei[i]) ** 2 for i in range(n)])
        _rows, _cols, _vals = compute_pairwise_adaptive_kernel_similarity(
            idx_nei=knn_index.neighbor_graph[0][:, 1:],
            dist_nei=dist_nei,
        )
        K = csr_matrix((_vals, (_rows, _cols)), shape=(n, n))
        K = K + K.T
        D = np.asarray(K.sum(axis=1)).flatten()
        D[D == 0] = 1.0
        D_inv_sqrt = 1.0 / np.sqrt(D)
        self._d_inv_sqrt = D_inv_sqrt
        D_inv_sqrt_diag = diags(D_inv_sqrt)
        return D_inv_sqrt_diag @ K @ D_inv_sqrt_diag

    def compute_multi_step_eigenspace(
        self, emb: np.ndarray, ndim_eigenspace: Count_t, **nn_kwargs
    ):
        _A = self._compute_one_step_transition(emb=emb, **nn_kwargs)
        res = compute_sparse_eigendecomposition(
            matrix=_A, which="LM", n_components=ndim_eigenspace
        )
        assert res is not None
        eigvals, eigvecs = res
        self.eigenvectors = self._d_inv_sqrt[:, np.newaxis] * eigvecs
        self.eigenvalues = eigvals
        if self.damp_multistep < 1.0:
            self.eigenvalues_multistep = eigvals / (1 - self.damp_multistep * eigvals)
        else:
            self.eigenvalues_multistep = eigvals

    def project_query_data(
        self,
        emb_reference: np.ndarray,
        emb_query: np.ndarray,
        cache_knn_index: bool = True,
    ) -> np.ndarray:
        assert emb_reference.shape[0] == self.eigenvectors.shape[0], (
            "mismatched dimensions between reference embedding and eigenspace"
        )
        assert emb_query.shape[1] == emb_reference.shape[1], (
            "mismatched dimensions between reference and query embeddings"
        )
        if self._knn_index_cache is not None:
            knn_index = self._knn_index_cache
        else:
            knn_index = self._compute_knn_index(
                emb_reference, cache=cache_knn_index, query=True
            )
        nn_indices, nn_distances = knn_index.query(
            query_data=emb_query, k=self.n_neighbors
        )
        return compute_knn_diffusion_projection(
            nn_indices, nn_distances, self.diffmap_coords, self._vars_local
        )
