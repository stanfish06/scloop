# Copyright 2026 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from typing import Literal

import numpy as np
from pynndescent import NNDescent


# compute density for energy landscape calculation
def compute_density(
    adata,
    basis: str = "X_diffmap",
    n_neighbors: int = 10,
    flavor: Literal["custom", "mellon"] = "custom",
    density_key="scloop_log_density",
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
            log_density = model.fit_predict(density_embedding)
    adata.obs[density_key] = log_density


# compute drift and diffusion tensors by using knn graph and pseudotime
def compute_drift_and_diffusion_tensors(
    adata,
    basis: str = "X_diffmap",
    n_neighbors: int = 10,
    pseudotime_key: str = "dpt_pseudotime",
) -> tuple[np.ndarray, np.ndarray]:
    pass


# decompose drift into gradient and rotational part
def compute_drift_decomposition(adata) -> None:
    pass
