# Copyright 2026 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np
from loguru import logger
from pydantic.dataclasses import dataclass
from pynndescent import NNDescent


@dataclass
class DistortionEstimate:
    L: float
    median_ratio: float
    quantiles: dict[float, float]


def estimate_bilipschitz_distortion(
    source_embedding: np.ndarray,
    mapped_embedding: np.ndarray,
    k: int = 15,
    quantile_levels: tuple[float, ...] = (0.5, 0.9, 0.99, 1.0),
) -> DistortionEstimate:
    assert source_embedding.shape[0] == mapped_embedding.shape[0]
    search_index = NNDescent(source_embedding)
    nn_indices, nn_distances_source = search_index.query(
        query_data=source_embedding, k=k
    )
    nn_distances_mapped = np.linalg.norm(
        mapped_embedding[nn_indices] - mapped_embedding[:, None, :], axis=2
    )

    d_source = nn_distances_source.ravel()
    d_shared = nn_distances_mapped.ravel()
    valid = d_source > 0
    ratio = d_shared[valid] / d_source[valid]
    distortion = np.maximum(ratio, 1.0 / ratio)

    L = float(distortion.max())
    median_ratio = float(np.median(ratio))
    quantiles = {float(q): float(np.quantile(distortion, q)) for q in quantile_levels}

    logger.info(
        f"Bi-Lipschitz distortion over {distortion.size} kNN edges (k={k}): "
        f"L={L:.4f}, median ratio={median_ratio:.4f}"
    )
    return DistortionEstimate(L=L, median_ratio=median_ratio, quantiles=quantiles)
