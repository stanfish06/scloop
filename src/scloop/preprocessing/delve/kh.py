import logging
from typing import Union

import anndata
import numpy as np
import scipy
from joblib import Parallel, delayed
from numba import njit
from pandas.api.types import is_numeric_dtype
from tqdm import tqdm
from tqdm_joblib import tqdm_joblib


def random_feats(
    X: np.ndarray,
    gamma: Union[int, float] = 1,
    frequency_seed: int = None,
    n_features: int = 1000,
):
    """Computes random Fourier frequency features: https://papers.nips.cc/paper/2007/hash/013a006f03dbc5392effeb8f18fda755-Abstract.html

    Parameters
    X: np.ndarray
        array of input data (dimensions = cells x features)
    gamma: Union([int, float]) (default = 1)
        scale for standard deviation of the normal distribution
    frequency_seed: int (default = None):
        random state parameter
    ----------

    Returns
    phi: np.ndarray
        random Fourier frequency features (dimensions = cells x 2000)
    ----------
    """
    if n_features <= 0:
        raise ValueError("n_features must be > 0.")
    scale = 1 / gamma

    if frequency_seed is not None:
        rng = np.random.default_rng(int(frequency_seed))
        W = rng.normal(scale=scale, size=(X.shape[1], n_features))
    else:
        W = np.random.normal(scale=scale, size=(X.shape[1], n_features))

    XW = np.dot(X, W)
    sin_XW = np.sin(XW)
    cos_XW = np.cos(XW)
    phi = np.concatenate((cos_XW, sin_XW), axis=1)

    return phi


def _density_to_weights(density: np.ndarray) -> np.ndarray:
    density = np.asarray(density, dtype=np.float64)
    density = np.where(np.isfinite(density), density, 0.0)
    scale = np.mean(density)
    eps = scale * 1e-6
    density = np.maximum(density, eps)
    weights = 1.0 / density
    weights = np.where(np.isfinite(weights), weights, 0.0)
    weights = weights / weights.sum()
    return weights


@njit
def kernel_herding(phi: np.ndarray, num_subsamples: int, weights: np.ndarray):
    """Performs kernel herding subsampling: https://arxiv.org/abs/1203.3472 using numba

    Parameters
    phi: np.ndarray
        random Fourier frequency features (dimensions = cells x 2000)
    num_subsamples: int (default = None)
        number of cells to subsample
    ----------

    Returns
    kh_indices: np.ndarray
        indices of subsampled cells
    ----------
    """
    num_cells, num_features = phi.shape
    kh_indices = np.empty(num_subsamples, dtype=np.int64)
    selected_mask = np.zeros(num_cells, dtype=np.int8)

    w_t = np.zeros(num_features)
    for i in range(num_features):
        total = 0.0
        for j in range(num_cells):
            total += phi[j, i] * weights[j]
        w_t[i] = total

    w_0 = np.copy(w_t)
    for subsample_idx in range(num_subsamples):  # find argmax
        max_score = -1e20
        new_ind = -1
        for cell_idx in range(num_cells):
            if selected_mask[cell_idx] == 0:
                score = 0.0
                for feature_idx in range(num_features):
                    score += phi[cell_idx, feature_idx] * w_t[feature_idx]
                if score > max_score:
                    max_score = score
                    new_ind = cell_idx
        if new_ind == -1:
            raise ValueError("Not enough unique indices to sample.")
        kh_indices[subsample_idx] = new_ind
        selected_mask[new_ind] = 1
        # update w_t
        for feature_idx in range(num_features):
            w_t[feature_idx] += w_0[feature_idx] - phi[new_ind, feature_idx]

    return kh_indices


def _parse_input(adata: anndata.AnnData):
    """accesses and parses data from adata object

    Parameters
    adata: anndata.AnnData
        annotated data object where adata.X is the attribute for preprocessed data

    ----------

    Returns
    X: np.ndarray
        array of data (dimensions = cells x features)
    ----------
    """
    try:
        if isinstance(adata, anndata.AnnData):
            X = adata.X.copy()
        if isinstance(X, scipy.sparse.csr_matrix):
            X = np.asarray(X.todense())
        if is_numeric_dtype(adata.obs_names):
            logging.warning("Converting cell IDs to strings.")
            adata.obs_names = adata.obs_names.astype("str")
    except NameError:
        pass

    return X


def kernel_herding_main(
    sample_set_ind,
    X: np.ndarray = None,
    gamma: Union[int, float] = 1,
    frequency_seed: int = None,
    num_subsamples: int = 500,
    density=None,
    n_features: int = 1000,
):
    """Performs kernel herding subsampling on a single sample-set using random features

    Parameters
    X: np.ndarray
        array of input data (dimensions = cells x features)
    gamma: Union([int, float]) (default = 1)
        scale for standard deviation of the normal distribution
    frequency_seed: int (default = None):
        random state parameter
    num_samples: int (default = None)
        number of cells to subsample
    sample_set_ind: np.ndarray
        array containing the indices of the sample-set to subsample. if you'd like to use all cells within X, please pass in np.arange(0, X.shape[0])
    ----------

    Returns
    kh_indices: np.ndarray
        indices of subsampled cells within the sample-set
    ----------
    """
    X = X[sample_set_ind, :]
    phi = random_feats(
        X, gamma=gamma, frequency_seed=frequency_seed, n_features=n_features
    )
    if density is None:
        weights = np.full(phi.shape[0], 1.0 / phi.shape[0], dtype=np.float64)
    else:
        density_arr = np.asarray(density, dtype=np.float64)
        if density_arr.shape[0] != phi.shape[0]:
            weights = np.full(phi.shape[0], 1.0 / phi.shape[0], dtype=np.float64)
        else:
            weights = _density_to_weights(density_arr)
    kh_indices = kernel_herding(phi, num_subsamples, weights)

    return kh_indices


def sketch(
    adata,
    sample_set_key: str = None,
    sample_set_inds=None,
    gamma: Union[int, float] = 1,
    frequency_seed: int = None,
    num_subsamples: int = 500,
    density=None,
    n_jobs: int = -1,
):
    """constructs a sketch using kernel herding and random Fourier frequency features

    Parameters
    adata: anndata.Anndata
        annotated data object (dimensions = cells x features)
    sample_set_key: str (default = None)
        string referring to the key within adata.obs that contains the sample-sets to subsample
            ~ if sample_set_key is None, will parse according to sample_set_inds
            ~ if sample_set_key is None and sample_set_inds is None, will use all cells as a single sample-set
    sample_set_inds: list (default = None)
        list of arrays containig the indices of the sample-sets to subsample. (dimensions = len(sample_sets)) e.g. [np.array([]), np.array([]), ... , np.array([])]
            ~ if sample_set_key is None and sample_set_inds is None, will use all cells as a single sample-set
    gamma: Union([int, float]) (default = 1)
        scale for standard deviation of the normal distribution within random Fourier frequency feature computation
    frequency_seed: int (default = None):
        random state parameter
    num_samples: int (default = None)
        number of cells to subsample per sample-set
    n_jobs: int (default = -1)
        number of tasks
    ----------

    Returns
    kh_indices: np.ndarray
        list of indices of subsampled cells per sample-set e.g. [np.array(ind0_S0..indx_S0), np.array(ind0_S1..indx_S1), ... , np.array(ind0_SX..indx_SX)]
    adata_subsample: anndata.AnnData
        annotated data object containing subsampled data
    ----------
    """
    if n_jobs == -1:
        n_jobs = Parallel().n_jobs
    elif n_jobs < -1:
        n_jobs = Parallel().n_jobs + 1 + n_jobs

    if isinstance(adata, anndata.AnnData) and (sample_set_key is not None):
        sample_set_id, idx = np.unique(adata.obs[sample_set_key], return_index=True)
        sample_set_id = sample_set_id[np.argsort(idx)]
        sample_set_inds = [
            np.where(adata.obs[sample_set_key] == i)[0] for i in sample_set_id
        ]
    elif sample_set_inds is None:
        sample_set_inds = [np.arange(0, adata.X.shape[0])]

    min_cell_size = min([len(i) for i in sample_set_inds])
    if num_subsamples > min_cell_size:
        logging.warning(
            f"Number of subsamples per sample-set {num_subsamples} is greater than the maximum number of cells in the smallest sample-set {min_cell_size}. \n Performing subsampling using {min_cell_size} cells per sample-set"
        )
        num_subsamples = min_cell_size

    n_sample_sets = len(sample_set_inds)
    X = _parse_input(adata)
    density_arr = None
    if density is not None:
        density_arr = np.asarray(density, dtype=np.float64)
        if density_arr.shape[0] != X.shape[0]:
            density_arr = None

    def process_set(i, inds):
        density_subset = None
        if density_arr is not None:
            density_subset = density_arr[inds]
        return kernel_herding_main(
            sample_set_ind=inds,
            X=X,
            gamma=gamma,
            frequency_seed=frequency_seed,
            num_subsamples=num_subsamples,
            density=density_subset,
        )

    with tqdm_joblib(tqdm(desc="Performing subsampling", total=n_sample_sets)):
        kh_indices = Parallel(n_jobs=n_jobs)(
            delayed(process_set)(i, inds) for i, inds in enumerate(sample_set_inds)
        )

    subsampled_cell_indices = [
        sample_set_inds[i][kh_indices[i]] for i in range(n_sample_sets)
    ]
    subsampled_cell_indices = np.concatenate(subsampled_cell_indices)
    adata_subsample = adata[subsampled_cell_indices, :].copy()

    return kh_indices, adata_subsample
