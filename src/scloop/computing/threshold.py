# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
"""
Pick the downsample size and the PH / boundary thresholds before the full PH run.

Start with small sample size and grows until exceeds budget to speed things up (compare with the other way around).
Pick threshold such that it is large enough for loop identification and matching.
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
    DEFAULT_MAX_PCT_ECC_RADIUS,
    DEFAULT_MIN_PCT_ECC_RADIUS,
    DEFAULT_PILOT_DEATH_QUANTILE,
    DEFAULT_PILOT_SEED_QUANTILE,
    DEFAULT_PILOT_THRESHOLD_MARGIN,
    DEFAULT_SEARCH_N_PILOT,
    DEFAULT_SEARCH_N_START,
)
from ..data.metadata import ScloopMeta
from ..data.ripser_lib import ripser  # type: ignore[import-not-found]
from ..preprocessing.downsample import eligible_indices, sample
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


def _cone_radius(points: np.ndarray) -> float:
    # eccentricity of the point nearest the centroid: at any radius >= this, that point
    # connects to every other point, so the Rips complex is a cone over it (contractible)
    # and no H1 class is born or alive at or beyond it
    points = np.asarray(points, dtype=np.float64)
    centre = points[np.argmin(((points - points.mean(axis=0)) ** 2).sum(axis=1))]
    eccentricity = float(np.sqrt(((points - centre) ** 2).sum(axis=1)).max())
    # pad outward so neighbour queries at the cap keep the farthest edge despite
    # rounding differences between distance routines
    return eccentricity * (1.0 + 1e-9)


def _top_h1_deaths(
    points: np.ndarray,
    n_top: int,
    radius: float,
    radius_max: float,
    radius_min_check: float,
) -> tuple[np.ndarray, float | None, bool]:
    """
    Top-n_top H1 deaths by persistence, the radius where every class had died (None if
    that never happened below radius_max), and whether a top class was still alive at
    radius_max (its death is then reported as radius_max).

    Doubles the PH radius until H1 classes exist and every one dies.
    """
    births = deaths = np.empty(0, dtype=np.float64)
    settled_at: float | None = None
    n_settled = 0
    radius = min(radius, radius_max)
    while True:
        dist = radius_neighbors_graph(points, radius=radius, mode="distance")
        if dist.nnz:
            result = ripser(
                distance_matrix=dist.tocoo(copy=False),
                modulus=2,
                dim_max=1,
                threshold=radius,
                do_cocycles=False,
            )
            births, deaths = (
                np.asarray(a, dtype=np.float64)
                for a in result.births_and_deaths_by_dim[1]
            )
            if deaths.size and np.isfinite(deaths).all():
                born_since = settled_at is not None and births.size > n_settled
                if (
                    settled_at is not None
                    and not born_since
                    and radius >= radius_min_check
                ):
                    break
                if settled_at is None or born_since:
                    settled_at = radius
                    n_settled = births.size
            else:
                settled_at = None
        if radius >= radius_max:
            break
        radius = min(2.0 * radius, radius_max) if radius > 0 else radius_max
    alive = ~np.isfinite(deaths)
    deaths = np.where(alive, radius, deaths)
    order = np.argsort(births - deaths)[:n_top]
    return deaths[order], settled_at, bool(alive.any())


def calibrate_threshold(
    points: np.ndarray,
    n_top: int,
    n_pilot: int = DEFAULT_SEARCH_N_PILOT,
    death_quantile: float = DEFAULT_PILOT_DEATH_QUANTILE,
    margin: float = DEFAULT_PILOT_THRESHOLD_MARGIN,
    radius_init: float | None = None,
    sampling_kwargs: dict[str, Any] | None = None,
    min_pct_ecc_radius: float = DEFAULT_MIN_PCT_ECC_RADIUS,
    max_pct_ecc_radius: float = DEFAULT_MAX_PCT_ECC_RADIUS,
) -> tuple[float, float]:
    """
    Threshold = margin * quantile of the top-n_top H1 deaths over n_pilot resamples.
    """
    if not 0.0 < min_pct_ecc_radius <= max_pct_ecc_radius <= 1.0:
        raise ValueError(
            "need 0 < min_pct_ecc_radius <= max_pct_ecc_radius <= 1, got "
            f"{min_pct_ecc_radius} and {max_pct_ecc_radius}"
        )
    radius_max = max_pct_ecc_radius * _cone_radius(points)
    if radius_init is None:
        distinct = np.unique(points, axis=0)
        k = min(max(1, int(np.ceil(np.log(len(distinct))))), len(distinct) - 1)
        if k >= 1:
            nn_dist, _ = KDTree(distinct).query(distinct, k=k + 1)
            radius_init = 2.0 * float(
                np.quantile(nn_dist[:, k], DEFAULT_PILOT_SEED_QUANTILE)
            )
        else:
            radius_init = radius_max
    radius_start = min(radius_init, radius_max)
    radius = radius_start
    source = list(range(len(points)))
    deaths: list[np.ndarray] = []
    censored = False
    for _ in range(n_pilot):
        idx = select_bootstrap_sample(
            source_indices=source, source_embedding=points, **(sampling_kwargs or {})
        )
        pilot = points[idx]
        # a small sampling fraction can leave too few points for a loop
        if len(pilot) < 3:
            continue
        cone_radius = _cone_radius(pilot)
        top, settled_at, alive_at_cap = _top_h1_deaths(
            pilot,
            n_top,
            radius,
            max_pct_ecc_radius * cone_radius,
            min_pct_ecc_radius * cone_radius,
        )
        deaths.append(top)
        censored |= alive_at_cap
        if settled_at is not None:
            radius = settled_at
    all_deaths = np.concatenate(deaths) if deaths else np.empty(0)
    if censored:
        logger.warning(
            "An H1 class is still alive at max_pct_ecc_radius x the cone radius; the top "
            "classes and the threshold may be underestimated. Raise max_pct_ecc_radius"
        )
    if all_deaths.size == 0:
        if count_edges_within(points, radius_start) == 0:
            # the shortest nearest-neighbour distance is the first radius with an edge
            distinct = np.unique(points, axis=0)
            if len(distinct) < 2:
                raise ValueError("All points coincide; no edge at any radius")
            shortest = float(KDTree(distinct).query(distinct, k=2)[0][:, 1].min())
            # pad outward so radius queries keep that edge despite rounding; may exceed
            # a fractional cap, which only limits the loop search
            radius_start = shortest * (1.0 + 1e-9)
        logger.warning(
            "Pilot resamples have no H1 class up to the cap; using the starting radius"
        )
        return radius_start, radius_start
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
    # a caller-supplied radius_init seeds the first calibration only
    radius_seed: float | None = calibrate_kwargs.pop("radius_init", None)

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
            # at least one more point, or small n never grows (int(4 * 1.2) == 4)
            n_next = min(
                n_max, max(n + 1, int(n * np.clip(growth, _MIN_GROWTH, _MAX_GROWTH)))
            )
        else:
            # try the smallest supported size before giving up
            n_next = max(n // 2, 3)
            if n <= 3:
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
    full_indices = indices_current
    n_max = len(indices_current) if indices_current is not None else emb.shape[0]
    eligible: list[int] | None = None
    filtering = (
        kwargs_downsample.get("percent_removal_density", 0.025) > 0
        or kwargs_downsample.get("percent_removal_lof", 0) > 0
    )
    if indices_current is None and filtering:
        # subsets come from one filtered pool (density filtering is not deterministic
        # across calls), and the search cannot ask for more cells than it holds
        filter_kwargs = {
            key: value
            for key, value in kwargs_downsample.items()
            if key not in ("random_state",)
        }
        pool = eligible_indices(adata, **filter_kwargs).tolist()
        eligible = pool
        if len(pool) < 2:
            raise ValueError(
                f"Only {len(pool)} cells remain after density/LOF filtering; at least 2 are needed"
            )
        if len(pool) < n_max:
            full_indices = pool
            n_max = len(pool)
    subset_indices: dict[int, list[int] | None] = {}

    def subset_points(n: int) -> np.ndarray:
        if n >= n_max and full_indices is eligible and eligible is not None:
            subset_indices[n] = sample(
                adata=adata, n=n_max, eligible=eligible, **kwargs_downsample
            )
        elif n >= n_max:
            subset_indices[n] = full_indices
        elif indices_current is not None:
            subset_indices[n] = list(indices_current[:n])
        else:
            subset_indices[n] = sample(
                adata=adata, n=n, eligible=eligible, **kwargs_downsample
            )
        idx = subset_indices[n]
        return emb if idx is None else emb[idx]

    choice = search_size_and_thresholds(
        subset_points=subset_points, n_max=n_max, **search_kwargs
    )
    return choice, subset_indices[choice.n]
