# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Literal

import numpy as np
import scanpy as sc
from anndata import AnnData
from loguru import logger
from numba import jit
from pydantic.dataclasses import dataclass
from pynndescent import NNDescent
from scipy.sparse import csr_matrix, diags
from sklearn.utils.extmath import randomized_svd

from ..data.constants import NUMERIC_EPSILON
from ..data.types import Count_t, Percent_t, PositiveFloat
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
    alpha_kernel_diffusion: PositiveFloat = 10.0,
    use_multistep_eigenvalues: bool = True,
    use_potential_embedding: bool = False,
    potential_t: PositiveFloat | list[PositiveFloat] = 3.0,
    potential_kind: Literal["log", "sqrt"] = "sqrt",
    auto_t: bool = True,
    pct_lower_bound_potential_t: Percent_t = 0.75,
) -> DiffusionMap:
    diffmap = DiffusionMap(
        n_neighbors=n_neighbors,
        damp_multistep=damp_multistep_diffusion,
        alpha_kernel=alpha_kernel_diffusion,
    )
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
            assert (
                "X_diffmap" in adata.obsm
                and type(adata.obsm["X_diffmap"]) is np.ndarray
            )
            diffmap.diffmap_coords = adata.obsm["X_diffmap"][:, 1:]
            adata.obsm["X_diffmap_original"] = adata.obsm["X_diffmap"].copy()
            adata.obsm["X_diffmap"] = adata.obsm["X_diffmap"][:, 1:]
        case "custom":
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
            diffusion_coords_full = eigvecs * eigvals
            adata.obsm["X_diffmap_original"] = diffusion_coords_full.copy()
            adata.obsm["X_diffmap"] = diffusion_coords_full[:, 1:]
            diffmap.diffmap_coords = diffusion_coords_full[:, 1:]
    if use_potential_embedding:
        if flavor != "custom":
            raise ValueError("use_potential_embedding requires flavor='custom'")
        coords = diffmap.compute_multi_step_potential_space(
            n_comps=n_comps,
            t=potential_t,
            kind=potential_kind,
            auto_t=auto_t,
            pct_lower_bound_potential_t=pct_lower_bound_potential_t,
        )
        adata.obsm["X_diffmap"] = coords
        diffmap.diffmap_coords = coords
    return diffmap


def von_neumann_entropy(eigvals: np.ndarray, t: PositiveFloat) -> float:
    p = np.clip(np.asarray(eigvals, dtype=np.float64), 0.0, None) ** float(t)
    total = p.sum()
    if total <= NUMERIC_EPSILON:
        return 0.0
    p = p / total
    p = p[p > 0.0]
    return float(-np.sum(p * np.log(p)))


def select_potential_ts(
    eigvals: np.ndarray,
    tvals: np.ndarray,
    lb_pct: Percent_t = 0.95,
) -> np.ndarray:
    eigvals = np.asarray(eigvals, dtype=np.float64)
    tvals = np.asarray(tvals, dtype=np.float64).ravel()
    tvals = np.sort(tvals)
    if tvals.size == 1:
        return tvals.copy()

    H = np.array([von_neumann_entropy(eigvals, t) for t in tvals])

    h_range = float(np.ptp(H))

    x = (tvals - tvals[0]) / (np.ptp(tvals) + NUMERIC_EPSILON)
    y = (H - H.min()) / h_range
    dx, dy = x[-1] - x[0], y[-1] - y[0]
    denom = np.hypot(dx, dy) + NUMERIC_EPSILON
    dist = np.abs(dy * (x - x[0]) - dx * (y - y[0])) / denom
    knee_idx = int(np.argmax(dist))

    threshold = lb_pct * float(H.max())
    below = np.flatnonzero(H <= threshold)
    lb_idx = int(below[0]) if below.size > 0 else 0
    lb_idx = min(lb_idx, knee_idx)

    return tvals[lb_idx : knee_idx + 1].copy()


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
    idx_nei: np.ndarray, dist_nei: np.ndarray, alpha: float = 10
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
        v = np.square(np.median(dist_nei[i]))
        vars_local[i] = NUMERIC_EPSILON if v == 0.0 else v

    kernel_sim_raw = np.zeros(n * (n - 1) // 2)
    kernel_density = np.ones(n) * NUMERIC_EPSILON

    for i in range(n):
        for ni in range(nn):
            j = idx_nei[i, ni]
            dist = dist_nei[i, ni]
            lo, hi = (i, j) if i < j else (j, i)
            k = n * lo + hi - ((lo + 2) * (lo + 1)) // 2
            sum_var = vars_local[i] + vars_local[j] + NUMERIC_EPSILON
            new_val = (1.0 / np.sqrt(sum_var * 0.5)) * np.exp(
                -(((dist**2) / sum_var) ** alpha)
            )
            if new_val > kernel_sim_raw[k]:
                kernel_sim_raw[k] = new_val

    for i in range(n):
        for ni in range(nn):
            j = idx_nei[i, ni]
            lo, hi = (i, j) if i < j else (j, i)
            k = n * lo + hi - ((lo + 2) * (lo + 1)) // 2
            kernel_density[i] += kernel_sim_raw[k] * np.sqrt(
                vars_local[j] + NUMERIC_EPSILON
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
            denom = kernel_density[lo] * kernel_density[hi] + NUMERIC_EPSILON
            vals[count] = kernel_sim_raw[k] / denom
            count += 1
    return rows[:count], cols[:count], vals[:count]


@dataclass(config=dict(arbitrary_types_allowed=True))
class DiffusionMap:
    n_neighbors: Count_t
    damp_multistep: Percent_t = 1.0
    alpha_kernel: PositiveFloat = 10.0
    eigenvalues: np.ndarray | None = None
    eigenvalues_multistep: np.ndarray | None = None
    eigenvectors: np.ndarray | None = None
    diffmap_coords: np.ndarray | None = (
        None  # no first component: first component of diffusion map represent local density
    )
    _knn_index_cache: NNDescent | None = None
    _d_inv_sqrt: np.ndarray | None = None
    _vars_local: np.ndarray | None = None

    _PERSISTED_ARRAYS = (
        "eigenvalues",
        "eigenvalues_multistep",
        "eigenvectors",
        "diffmap_coords",
        "_d_inv_sqrt",
        "_vars_local",
    )

    def to_hdf5_group(self, group, compress: bool = True) -> None:
        group.attrs["_type"] = "DiffusionMap"
        group.attrs["n_neighbors"] = int(self.n_neighbors)
        group.attrs["damp_multistep"] = float(self.damp_multistep)
        group.attrs["alpha_kernel"] = float(self.alpha_kernel)
        kw = {"compression": "gzip"} if compress else {}
        # NNDescent cache is intentionally not persisted; it is rebuilt lazily
        # by project_query_data.
        for name in self._PERSISTED_ARRAYS:
            value = getattr(self, name)
            if value is not None:
                group.create_dataset(name, data=np.asarray(value), **kw)

    @classmethod
    def from_hdf5_group(cls, group) -> "DiffusionMap":
        def _opt(name):
            return np.asarray(group[name]) if name in group else None

        obj = cls(
            n_neighbors=int(group.attrs["n_neighbors"]),
            damp_multistep=float(group.attrs.get("damp_multistep", 1.0)),
            alpha_kernel=float(group.attrs.get("alpha_kernel", 10.0)),
            eigenvalues=_opt("eigenvalues"),
            eigenvalues_multistep=_opt("eigenvalues_multistep"),
            eigenvectors=_opt("eigenvectors"),
            diffmap_coords=_opt("diffmap_coords"),
        )
        obj._d_inv_sqrt = _opt("_d_inv_sqrt")
        obj._vars_local = _opt("_vars_local")
        return obj

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
        logger.debug(f"Diffusion transition: {n} cells, n_neighbors={self.n_neighbors}")
        knn_index = self._compute_knn_index(
            emb=emb, cache=False, query=False, **nn_kwargs
        )
        dist_nei = knn_index.neighbor_graph[1][:, 1:]
        self._vars_local = np.array(
            [max(np.median(dist_nei[i]) ** 2, NUMERIC_EPSILON) for i in range(n)]
        )
        _rows, _cols, _vals = compute_pairwise_adaptive_kernel_similarity(
            idx_nei=knn_index.neighbor_graph[0][:, 1:],
            dist_nei=dist_nei,
            alpha=self.alpha_kernel,
        )
        K = csr_matrix((_vals, (_rows, _cols)), shape=(n, n))
        K = K + K.T
        D = np.asarray(K.sum(axis=1)).flatten()
        D = np.maximum(D, NUMERIC_EPSILON)
        D_inv_sqrt = 1.0 / np.sqrt(D)
        self._d_inv_sqrt = D_inv_sqrt
        D_inv_sqrt_diag = diags(D_inv_sqrt)
        return D_inv_sqrt_diag @ K @ D_inv_sqrt_diag

    def compute_multi_step_eigenspace(
        self, emb: np.ndarray, ndim_eigenspace: Count_t, **nn_kwargs
    ):
        _A = self._compute_one_step_transition(emb=emb, **nn_kwargs)
        logger.debug(
            f"Multi-step eigenspace: transition matrix {_A.shape}, "
            f"ndim={ndim_eigenspace}"
        )
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

    def compute_multi_step_potential_space(
        self,
        n_comps: Count_t,
        t: PositiveFloat | list[PositiveFloat],
        kind: Literal["log", "sqrt"] = "sqrt",
        auto_t: bool = True,
        pct_lower_bound_potential_t: Percent_t = 0.75,
        random_state: int = 0,
    ) -> np.ndarray:
        assert (
            self.eigenvectors is not None
            and self.eigenvalues is not None
            and self._d_inv_sqrt is not None
        )
        eigvecs = self.eigenvectors.astype(np.float32)
        eigvals = self.eigenvalues.astype(np.float32)
        d_inv_sqrt = self._d_inv_sqrt.astype(np.float32)
        d_inv_sqrt_safe = np.clip(d_inv_sqrt, NUMERIC_EPSILON, None)
        d_sqrt = (1.0 / d_inv_sqrt_safe).astype(np.float32)
        V_sym = d_sqrt[:, np.newaxis] * eigvecs
        eigvals_clipped = np.clip(eigvals, 0.0, None)
        ts = np.atleast_1d(np.asarray(t, dtype=np.float32))
        if auto_t:
            ts = select_potential_ts(
                eigvals=eigvals_clipped,
                tvals=ts,
                lb_pct=pct_lower_bound_potential_t,
            )
        n = V_sym.shape[0]
        logger.debug(
            f"Potential space: {len(ts)} diffusion time(s), "
            f"building {n}x{n * len(ts)} float32 matrix"
        )
        U_all = np.empty((n, n * len(ts)), dtype=np.float32)
        for i, t_i in enumerate(ts):
            logger.debug(f"Potential space: diffusion time {i + 1}/{len(ts)} (t={t_i:.4g})")
            eigvals_t = eigvals_clipped**t_i
            P_t = (V_sym * eigvals_t) @ V_sym.T
            P_t *= d_inv_sqrt_safe[:, np.newaxis]
            P_t *= d_sqrt[np.newaxis, :]
            block = U_all[:, i * n : (i + 1) * n]
            match kind:
                case "log":
                    np.clip(P_t, NUMERIC_EPSILON, None, out=P_t)
                    np.log(P_t, out=block)
                    block *= -1.0
                case "sqrt":
                    np.clip(P_t, 0.0, None, out=P_t)
                    np.sqrt(P_t, out=block)
        U_all -= U_all.mean(axis=0, keepdims=True)
        k = min(n_comps, n - 1)
        logger.debug(f"Potential space: randomized SVD on {U_all.shape} matrix, k={k}")
        U_left, S, _ = randomized_svd(U_all, n_components=k, random_state=random_state)
        return (U_left * S).astype(np.float32)

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
