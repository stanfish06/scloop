# Copyright 2026 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from typing import Literal

import numpy as np
from pynndescent import NNDescent


# compute density for energy landscape calculation
def compute_density(
    adata,
    basis: str = "X_diffmap",
    n_neighbors: int = 10,
    flavor: Literal["custom", "mellon"] = "mellon",
    density_key="scloop_log_density",
    score_key="scloop_score",
) -> None:
    density_embedding = adata.obsm[basis]
    match flavor:
        case "custom":
            index = NNDescent(density_embedding, n_neighbors=n_neighbors)
            _, distances = index.query(density_embedding, k=n_neighbors)
            distances = distances / np.median(distances)
            d_sq = distances**2
            bandwidth_inv = d_sq.mean(axis=1, keepdims=True)
            similarities = np.exp(-d_sq * bandwidth_inv / 2)
            log_density = np.log(similarities.sum(axis=1))
        case "mellon":
            import mellon

            model = mellon.DensityEstimator()
            model.fit(density_embedding)
            predictor = model.predict
            log_density = np.asarray(predictor(density_embedding))
            adata.obsm[score_key] = np.asarray(predictor.gradient(density_embedding))
    adata.obs[density_key] = log_density


# compute drift and diffusion tensors by using knn graph and pseudotime
def compute_drift_and_diffusion_tensors(
    adata,
    basis: str = "X_diffmap",
    n_neighbors: int = 30,
    pseudotime_key: str = "dpt_pseudotime",
    threshold_scheme: Literal["hard", "soft"] = "hard",
    frac_to_keep: float = 0.3,
    b: float = 10.0,
    nu: float = 0.5,
    drift_key="scloop_drift",
    diffusion_key="scloop_diffusion",
) -> tuple[np.ndarray, np.ndarray]:
    X = adata.obsm[basis]
    pseudotime = np.asarray(adata.obs[pseudotime_key])

    index = NNDescent(X, n_neighbors=n_neighbors + 1)
    neighbors, distances = index.query(X, k=n_neighbors + 1)
    neighbors, distances = neighbors[:, 1:], distances[:, 1:]

    bandwidth = distances[:, -1][:, None]
    affinity = np.exp(-(distances**2) / (bandwidth**2))

    cell_pt = pseudotime[:, None]
    neigh_pt = pseudotime[neighbors]

    if threshold_scheme == "hard":
        k_thresh = min(30, int(np.floor(n_neighbors * frac_to_keep)))
        is_close = np.arange(n_neighbors)[None, :] < k_thresh
        keep = is_close | (neigh_pt >= cell_pt)
        biased = affinity * keep
    else:
        dt = cell_pt - neigh_pt
        weights = np.where(dt > 0, 2.0 / (1.0 + np.exp(b * dt)) ** (1.0 / nu), 1.0)
        biased = affinity * weights

    T = biased / biased.sum(axis=1, keepdims=True)

    displacements = X[neighbors] - X[:, None, :]
    drift = np.einsum("nk,nkd->nd", T, displacements)

    residuals = displacements - drift[:, None, :]
    D = 0.5 * np.einsum("nk,nkd,nke->nde", T, residuals, residuals).mean(axis=0)
    D = 0.5 * (D + D.T)

    adata.obsm[drift_key] = drift
    adata.uns[diffusion_key] = D
    return drift, D


# decompose drift into gradient and rotational part
def compute_drift_decomposition(
    adata,
    drift_key="scloop_drift",
    diffusion_key="scloop_diffusion",
    score_key="scloop_score",
    density_key="scloop_log_density",
    gradient_drift_key="scloop_gradient_drift",
    flux_key="scloop_flux_velocity",
    current_key="scloop_probability_current",
) -> None:
    drift = adata.obsm[drift_key]
    D = adata.uns[diffusion_key]
    score = adata.obsm[score_key]
    log_density = np.asarray(adata.obs[density_key])

    gradient_drift = score @ D
    flux_velocity = drift - gradient_drift
    current = np.exp(log_density)[:, None] * flux_velocity

    adata.obsm[gradient_drift_key] = gradient_drift
    adata.obsm[flux_key] = flux_velocity
    adata.obsm[current_key] = current
