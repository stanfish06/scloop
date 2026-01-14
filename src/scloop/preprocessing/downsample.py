# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
import numpy as np
import pandas as pd
from anndata import AnnData
from numba import jit
from pydantic import validate_call
from pynndescent import NNDescent

from ..data.types import EmbeddingMethod, IndexListDownSample, SizeDownSample

__all__ = ["sample", "sample_farthest_points", "sample_farthest_points_randomized"]


@jit(nopython=True)
def _sample_impl(
    data: np.ndarray,
    class_labels: np.ndarray,
    classes: np.ndarray,
    seed_indices: np.ndarray,
    n: int,
) -> np.ndarray:
    # selected observation indices
    indices = np.zeros(n, dtype=np.int64)
    min_dists = np.full(len(data), np.inf)

    num_seeds = len(seed_indices)
    num_classes = len(classes)
    indices[0] = seed_indices[0]

    # for each newly added points, recompute and select out-of-bag hausdorff point
    for i in range(1, n):
        last_point = data[indices[i - 1]]
        dists = np.sum((data - last_point) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[indices[:i]] = -1
        if i >= num_seeds:
            class_indicies = np.where(class_labels == classes[i % num_classes])[0]
            next_idx = class_indicies[np.argmax(min_dists[class_indicies])]
            # if this class is exhausted, fall back to reguler sampling
            if min_dists[next_idx] == -1:
                next_idx = np.argmax(min_dists)
            indices[i] = next_idx
        else:
            indices[i] = seed_indices[i]

    return indices


@jit(nopython=True)
def _select_index_from_candidates(
    min_dists: np.ndarray,
    candidate_indices: np.ndarray,
    top_k: int,
    alpha: float,
) -> int:
    top_idx = np.full(top_k, -1, dtype=np.int64)
    top_val = np.full(top_k, -1.0)

    for i in range(candidate_indices.shape[0]):
        idx = candidate_indices[i]
        val = min_dists[idx]
        if val < 0:
            continue

        min_pos = 0
        min_val = top_val[0]
        for t in range(1, top_k):
            if top_val[t] < min_val:
                min_val = top_val[t]
                min_pos = t
        if val > min_val:
            top_val[min_pos] = val
            top_idx[min_pos] = idx

    count = 0
    for t in range(top_k):
        if top_idx[t] >= 0:
            top_idx[count] = top_idx[t]
            top_val[count] = top_val[t]
            count += 1

    if count == 0:
        return -1

    total = 0.0
    for t in range(count):
        val = top_val[t]
        if alpha == 0.0:
            # Uniform selection among top-k candidates.
            weight = 1.0
        elif val <= 0.0:
            weight = 0.0
        else:
            weight = val if alpha == 1.0 else val**alpha
        top_val[t] = weight
        total += weight

    if total <= 0.0:
        pick = int(np.random.random() * count)
        return top_idx[pick]

    r = np.random.random() * total
    cum = 0.0
    for t in range(count):
        cum += top_val[t]
        if r <= cum:
            return top_idx[t]

    return top_idx[count - 1]


@jit(nopython=True)
def _sample_impl_randomized(
    data: np.ndarray,
    class_labels: np.ndarray,
    classes: np.ndarray,
    seed_indices: np.ndarray,
    n: int,
    top_k: int,
    alpha: float,
    seed: int,
) -> np.ndarray:
    if seed >= 0:
        np.random.seed(seed)

    indices = np.zeros(n, dtype=np.int64)
    min_dists = np.full(len(data), np.inf)

    num_seeds = len(seed_indices)
    num_classes = len(classes)
    indices[0] = seed_indices[0]
    all_indices = np.arange(len(data))

    for i in range(1, n):
        last_point = data[indices[i - 1]]
        dists = np.sum((data - last_point) ** 2, axis=1)
        min_dists = np.minimum(min_dists, dists)
        min_dists[indices[:i]] = -1

        if i < num_seeds:
            indices[i] = seed_indices[i]
            continue

        if num_classes > 0:
            class_indices = np.where(class_labels == classes[i % num_classes])[0]
            next_idx = _select_index_from_candidates(
                min_dists, class_indices, top_k, alpha
            )
        else:
            next_idx = -1

        if next_idx < 0:
            next_idx = _select_index_from_candidates(
                min_dists, all_indices, top_k, alpha
            )

        indices[i] = next_idx

    return indices


def sample_farthest_points(
    embedding: np.ndarray, n: int, *, random_state: int | None = None
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be > 0")
    if n > embedding.shape[0]:
        raise ValueError(
            f"n must be <= number of points (got n={n}, n_points={embedding.shape[0]})"
        )

    embedding_arr = np.ascontiguousarray(embedding)
    n_points = embedding_arr.shape[0]

    rng = np.random if random_state is None else np.random.RandomState(random_state)
    seed_indices = np.array([rng.randint(n_points)], dtype=np.int64)

    class_labels = np.zeros(n_points, dtype=np.int64)
    classes = np.array([0], dtype=np.int64)

    return _sample_impl(embedding_arr, class_labels, classes, seed_indices, n)


def sample_farthest_points_randomized(
    embedding: np.ndarray,
    n: int,
    *,
    random_state: int | None = None,
    top_k: int = 5,
    alpha: float = 1.0,
) -> np.ndarray:
    if n <= 0:
        raise ValueError("n must be > 0")
    if top_k <= 0:
        raise ValueError("top_k must be > 0")
    if alpha < 0:
        raise ValueError("alpha must be >= 0")

    embedding_arr = np.ascontiguousarray(embedding)
    n_points = embedding_arr.shape[0]

    if n > n_points:
        raise ValueError(
            f"n must be <= number of points (got n={n}, n_points={n_points})"
        )

    top_k = min(top_k, n_points)

    if random_state is None:
        seed = np.random.randint(np.iinfo(np.int32).max)
    else:
        seed = int(random_state)

    rng = np.random.RandomState(seed)
    seed_indices = np.array([rng.randint(n_points)], dtype=np.int64)

    class_labels = np.zeros(n_points, dtype=np.int64)
    classes = np.array([0], dtype=np.int64)

    return _sample_impl_randomized(
        embedding_arr, class_labels, classes, seed_indices, n, top_k, alpha, seed
    )


@validate_call(config={"arbitrary_types_allowed": True})
def sample(
    adata: AnnData,
    groupby: str | None,
    embedding_method: EmbeddingMethod,
    n: SizeDownSample,
    random_state: int = 0,
    percent_removal_density: float = 0.025,
    n_neighbors_density: int = 50,
) -> IndexListDownSample:
    """
    Topology-preserving downsampling using greedy farthest-point sampling.

    Args:
        adata: AnnData object containing the data
        groupby: column in adata.obs for class-balanced sampling, or None
        embedding_method: which embedding to use from adata.obsm
        n: number of points to sample
        random_state: random seed for reproducibility

    Returns:
        list of indices into adata.obs for the downsampled points
    """
    assert n <= adata.shape[0]
    assert f"X_{embedding_method}" in adata.obsm
    downsample_embedding = adata.obsm[f"X_{embedding_method}"]
    assert type(downsample_embedding) is np.ndarray

    # TODO: try mellon for better density estimation
    if percent_removal_density > 0:
        index = NNDescent(downsample_embedding, n_neighbors=n_neighbors_density)
        _, distances = index.query(downsample_embedding, k=n_neighbors_density)
        distances = distances / np.median(distances)
        d_sq = distances**2
        bandwidth_inv = d_sq.mean(axis=1, keepdims=True)
        similarities = np.exp(-d_sq * bandwidth_inv / 2)
        density = similarities.sum(axis=1)

        n_total = len(density)
        min_removal = percent_removal_density
        current_percentile = percent_removal_density / 2
        threshold = np.percentile(density, current_percentile * 100)
        while current_percentile < 1.0:
            threshold = np.percentile(density, current_percentile * 100)
            n_kept = np.sum(density >= threshold)
            if n_kept <= n_total * (1 - min_removal):
                break
            current_percentile *= 2

        valid_mask = density >= threshold
        local_indices = np.where(valid_mask)[0]
        downsample_embedding_local = downsample_embedding[local_indices]
    else:
        local_indices = np.arange(adata.shape[0])
        downsample_embedding_local = downsample_embedding

    if groupby is None:
        class_labels = np.zeros(len(local_indices), dtype=np.int64)
        classes = np.array([0])
        np.random.seed(random_state)
        seed_indices = np.array([np.random.randint(len(local_indices))])
    else:
        assert type(adata.obs) is pd.DataFrame
        assert groupby in adata.obs.columns
        class_labels_global, classes = pd.factorize(adata.obs.loc[:, groupby])
        class_labels = class_labels_global[local_indices].astype(np.int64)
        classes_present = np.unique(class_labels)
        seed_indices = []
        np.random.seed(random_state)
        for c in classes_present:
            seed_indices.append(np.random.choice(np.where(class_labels == c)[0]))
        seed_indices = np.array(seed_indices)
        classes = classes_present

    local_result = _sample_impl(
        downsample_embedding_local, class_labels, classes, seed_indices, n
    )
    return local_indices[local_result].tolist()
