from scipy.sparse import csr_matrix, issparse
from anndata import AnnData
import numpy as np
from .Sanity import run_sanity

def compute_posterior_gene_noise_model(adata: AnnData, use_layer: str | None = None, nbins: int = 160, vmin: float = 0.001, vmax: float = 50):
    # todo, send warnings for non-integers and large input
    X = adata.X if use_layer is None else adata.layers[use_layer]
    # Sanity runs gene-wisely, so we do transpose first
    X = X.T
    # make sure numpy is row-major (each row (gene) is contiguous in memory)
    X = X.toarray(order='C') if issparse(X) else np.ascontiguousarray(X)
    library_size = X.sum(axis=0)
    gene_total = X.sum(axis=1)
    log_mean, log_var = run_sanity(X, library_size, gene_total, nbins, vmin, vmax)
    adata.layers["sanity_log_mean"] = log_mean.T
    adata.layers["sanity_log_var"] = log_var.T
