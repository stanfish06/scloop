# Copyright 2026 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
import numpy as np
from anndata import AnnData
from numba import jit
from scipy.sparse import issparse

from ...data.constants import NUMERIC_EPSILON
from .Sanity import run_sanity


def compute_posterior_gene_noise_model(
    adata: AnnData,
    use_layer: str | None = None,
    nbins: int = 160,
    vmin: float = 0.001,
    vmax: float = 50,
):
    # todo, send warnings for non-integers and large input
    X = adata.X if use_layer is None else adata.layers[use_layer]
    # Sanity runs gene-wisely, so we do transpose first
    X = X.T
    # make sure numpy is row-major (each row (gene) is contiguous in memory)
    X = X.toarray(order="C") if issparse(X) else np.ascontiguousarray(X)
    library_size = X.sum(axis=0)
    gene_total = X.sum(axis=1)
    log_mean, log_var = run_sanity(X, library_size, gene_total, nbins, vmin, vmax)
    adata.layers["sanity_log_mean"] = log_mean.T
    adata.layers["sanity_log_var"] = log_var.T


@jit(nopython=True, cache=True)
def _sample_posterior_predictive_counts(
    log_mean: np.ndarray,
    log_var: np.ndarray,
    library_size: np.ndarray,
    cell_idx: np.ndarray,
) -> np.ndarray:
    n_cells = cell_idx.shape[0]
    n_genes = log_mean.shape[1]
    counts = np.empty((n_cells, n_genes), dtype=np.float64)
    for i in range(n_cells):
        c = cell_idx[i]
        for g in range(n_genes):
            ltq = np.random.normal(log_mean[c, g], np.sqrt(log_var[c, g]))
            rate = library_size[c] * np.exp(ltq)
            counts[i, g] = np.random.poisson(rate)
    return counts


def sample_posterior_predictive_counts(
    adata: AnnData,
    cell_idx: np.ndarray,
    scale_before_pca: bool = False,
    n_pca_comps: int | None = None,
) -> np.ndarray:
    log_mean = np.ascontiguousarray(adata.layers["sanity_log_mean"])
    log_var = np.ascontiguousarray(adata.layers["sanity_log_var"])
    library_size = np.asarray(
        adata.layers["counts"].sum(axis=1), dtype=np.float64
    ).ravel()
    X = _sample_posterior_predictive_counts(log_mean, log_var, library_size, cell_idx)
    # map back to Sanity's log-fraction scale
    N_c = library_size[cell_idx]
    X = np.log(X / N_c[:, np.newaxis] + NUMERIC_EPSILON)
    if scale_before_pca:
        X = (X - adata.var["mean"].values) / adata.var["std"].values
    if n_pca_comps is not None:
        X = X @ adata.varm["PCs"]
    return X
