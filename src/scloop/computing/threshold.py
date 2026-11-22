# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
"""
Pick the downsample size and the PH / boundary thresholds before the full PH run.

The search grows the point set from a small n and stops at the last n whose
boundary matrix fits the row budget, so no large PH or boundary matrix is built
just to be shrunk afterwards. A threshold that is not given is calibrated on
the point set that is actually used: pilot resamples of that set, drawn like
the real bootstrap, give the deaths of the most persistent H1 classes.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from anndata import AnnData
from loguru import logger
from scipy.spatial import KDTree
from sklearn.neighbors import radius_neighbors_graph

from ..data.constants import (
    DEFAULT_PILOT_DEATH_QUANTILE,
    DEFAULT_PILOT_THRESHOLD_MARGIN,
    DEFAULT_SEARCH_N_PILOT,
    DEFAULT_SEARCH_N_START,
)
from ..data.metadata import ScloopMeta
from ..data.ripser_lib import ripser  # type: ignore[import-not-found]
from ..preprocessing.downsample import sample
from .homology import select_bootstrap_sample

# growth per step is clamped to this range of multiples of the current n
_MIN_GROWTH = 1.2
_MAX_GROWTH = 2.0
# the interpolated n after an overshoot aims this fraction of the way to the budget
_REFINE_SAFETY = 0.97


@dataclass
class SizeThresholdChoice:
    n: int
    threshold_homology: float
    threshold_boundary: float
    # pairs within threshold_boundary: an upper bound on the boundary matrix rows
    n_edges: int
    # (n, threshold_boundary, n_edges) of every evaluated point set
    history: list[tuple[int, float, int]] = field(default_factory=list)


def count_edges_within(points: np.ndarray, radius: float) -> int:
    tree = KDTree(points)
    # count_neighbors counts ordered pairs and includes each point with itself
    return (int(tree.count_neighbors(tree, radius)) - len(points)) // 2


def _top_h1_deaths(
    points: np.ndarray, n_top: int, radius: float, radius_max: float
) -> tuple[np.ndarray, float]:
    # double the PH radius until every H1 class dies, then rank by persistence
    while True:
        dist = radius_neighbors_graph(points, radius=radius, mode="distance")
        result = ripser(
            distance_matrix=dist.tocoo(copy=False),
            modulus=2,
            dim_max=1,
            threshold=radius,
            do_cocycles=False,
        )
        births, deaths = (
            np.asarray(a, dtype=np.float64) for a in result.births_and_deaths_by_dim[1]
        )
        if np.isfinite(deaths).all() or radius >= radius_max:
            break
        radius = min(2.0 * radius, radius_max)
    deaths = np.where(np.isfinite(deaths), deaths, radius)
    order = np.argsort(births - deaths)
    return deaths[order[:n_top]], radius


def calibrate_threshold(
    points: np.ndarray,
    n_top: int,
    n_pilot: int = DEFAULT_SEARCH_N_PILOT,
    death_quantile: float = DEFAULT_PILOT_DEATH_QUANTILE,
    margin: float = DEFAULT_PILOT_THRESHOLD_MARGIN,
    radius_init: float | None = None,
    sampling_kwargs: dict[str, Any] | None = None,
) -> tuple[float, float]:
    """
    Threshold = margin * quantile of the top-n_top H1 deaths over n_pilot resamples.

    Returns the threshold and the PH radius the pilots reached, which seeds the
    next calibration.
    """
    radius_max = float(np.linalg.norm(np.ptp(points, axis=0)))
    if radius_init is None:
        nn_dist, _ = KDTree(points).query(points, k=2)
        radius_init = 2.0 * float(nn_dist[:, 1].max())
    radius = min(radius_init, radius_max)
    source = list(range(len(points)))
    deaths: list[np.ndarray] = []
    for _ in range(n_pilot):
        idx = select_bootstrap_sample(
            source_indices=source, source_embedding=points, **(sampling_kwargs or {})
        )
        top, radius = _top_h1_deaths(points[idx], n_top, radius, radius_max)
        deaths.append(top)
    all_deaths = np.concatenate(deaths)
    if all_deaths.size == 0:
        logger.warning("Pilot resamples have no H1 class; using the pilot PH radius")
        return radius, radius
    return float(np.quantile(all_deaths, death_quantile) * margin), radius


def search_size_and_thresholds(
    subset_points: Callable[[int], np.ndarray],
    n_max: int,
    max_edges: int,
    n_top: int,
    threshold_homology: float | None = None,
    threshold_boundary: float | None = None,
    auto_size: bool = True,
    n_start: int = DEFAULT_SEARCH_N_START,
    verbose: bool = False,
    **calibrate_kwargs,
) -> SizeThresholdChoice:
    """
    Pick n and the thresholds; subset_points(n) returns the n-point set.

    threshold_boundary given: only n is searched, then a missing
    threshold_homology is calibrated on the final set and kept >= threshold_boundary.
    threshold_homology given alone: threshold_boundary = min(calibrated, threshold_homology) at each n.
    Neither given: both equal the calibrated threshold at each n.
    """
    if (
        threshold_homology is not None
        and threshold_boundary is not None
        and threshold_homology < threshold_boundary
    ):
        raise ValueError(
            f"threshold_homology ({threshold_homology}) must be >= threshold_boundary ({threshold_boundary})"
        )
    radius_seed: float | None = None

    def calibrate(points: np.ndarray) -> float:
        nonlocal radius_seed
        thresh, radius_seed = calibrate_threshold(
            points, n_top=n_top, radius_init=radius_seed, **calibrate_kwargs
        )
        return thresh

    def evaluate(n: int) -> tuple[np.ndarray, float, int]:
        points = subset_points(n)
        if threshold_boundary is not None:
            tb = threshold_boundary
        else:
            tb = calibrate(points)
            if threshold_homology is not None:
                tb = min(tb, threshold_homology)
        n_edges = count_edges_within(points, tb)
        if verbose:
            logger.info(
                f"Size search: n={n}, threshold_boundary={tb:.4f}, edges={n_edges}"
            )
        return points, tb, n_edges

    history: list[tuple[int, float, int]] = []
    fit: tuple[int, np.ndarray, float, int] | None = None
    overshoot: tuple[int, int] | None = None
    refined = False
    n = n_max if not auto_size else min(n_start, n_max)
    while True:
        points, tb, n_edges = evaluate(n)
        history.append((n, tb, n_edges))
        if n_edges <= max_edges or not auto_size:
            fit = (n, points, tb, n_edges)
        else:
            overshoot = (n, n_edges)
        if not auto_size or (fit is not None and fit[0] >= n_max):
            break
        if fit is not None and overshoot is not None:
            # one interpolation step between the last fit and the overshoot, using edges ~ n^a
            if refined:
                break
            refined = True
            n_fit, e_fit = fit[0], max(fit[3], 1)
            n_over, e_over = overshoot
            a = (
                np.log(e_over / e_fit) / np.log(n_over / n_fit)
                if e_over > e_fit
                else 2.0
            )
            n_next = int(
                n_fit * (max_edges / e_fit) ** (1.0 / max(a, 1e-3)) * _REFINE_SAFETY
            )
            n_next = min(n_next, n_over - 1)
            if n_next <= n_fit * 1.05:
                break
        elif fit is not None:
            growth = np.sqrt(max_edges / max(n_edges, 1))
            n_next = min(n_max, int(n * np.clip(growth, _MIN_GROWTH, _MAX_GROWTH)))
        else:
            n_next = n // 2
            if n_next < 3:
                raise ValueError(
                    f"Boundary matrix exceeds {max_edges} rows even at n={n}; raise max_rows_boundary_matrix or lower the thresholds"
                )
        n = n_next
    assert fit is not None

    n, points, tb, n_edges = fit
    if threshold_homology is not None:
        th = threshold_homology
    elif threshold_boundary is not None:
        th = max(calibrate(points), threshold_boundary)
    else:
        th = tb
    if n_edges > max_edges:
        logger.warning(
            f"Boundary matrix has up to {n_edges} rows, above the budget of {max_edges}. Downstream computation could be slow"
        )
    return SizeThresholdChoice(
        n=n,
        threshold_homology=th,
        threshold_boundary=tb,
        n_edges=n_edges,
        history=history,
    )


def choose_downsample_and_thresholds(
    adata: AnnData,
    meta: ScloopMeta,
    **search_kwargs,
) -> tuple[SizeThresholdChoice, list[int] | None]:
    """
    Run search_size_and_thresholds on prefixes of the current downsample.

    Returns the choice and the downsample indices for choice.n (None = all cells).
    """
    assert meta.preprocess is not None
    assert meta.preprocess.embedding_method is not None
    emb = np.asarray(adata.obsm[f"X_{meta.preprocess.embedding_method}"])
    indices_current = meta.preprocess.indices_downsample
    kwargs_downsample = dict(meta.preprocess.kwargs_downsample or {})
    kwargs_downsample.pop("n", None)
    kwargs_downsample.setdefault("embedding_method", meta.preprocess.embedding_method)
    kwargs_downsample.setdefault("groupby", None)
    n_max = len(indices_current) if indices_current is not None else emb.shape[0]
    subset_indices: dict[int, list[int] | None] = {}

    def subset_points(n: int) -> np.ndarray:
        # farthest-point sampling is greedy, so the first n points of the
        # existing downsample are the n-point downsample
        if n >= n_max:
            subset_indices[n] = indices_current
        elif indices_current is not None:
            subset_indices[n] = list(indices_current[:n])
        else:
            subset_indices[n] = sample(adata=adata, n=n, **kwargs_downsample)
        idx = subset_indices[n]
        return emb if idx is None else emb[idx]

    choice = search_size_and_thresholds(
        subset_points=subset_points, n_max=n_max, **search_kwargs
    )
    return choice, subset_indices[choice.n]
