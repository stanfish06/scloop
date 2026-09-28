# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from ..data.boundary import BoundaryMatrixD1
from ..data.constants import DEFAULT_COLUMN_TRIM_METHOD
from ..data.types import ColumnTrimMethod
from .homology import compute_loop_homological_equivalence


def compute_loop_fillings(
    loop_mask: np.ndarray,
    boundary_matrix_d1: BoundaryMatrixD1,
    death: float,
    column_trim_method: ColumnTrimMethod = DEFAULT_COLUMN_TRIM_METHOD,
    column_scores: np.ndarray | None = None,
) -> list[tuple[int, ...] | None]:
    n_loops = loop_mask.shape[0]
    result = compute_loop_homological_equivalence(
        boundary_matrix_d1=boundary_matrix_d1,
        loop_mask_a=loop_mask,
        loop_mask_b=np.zeros((1, loop_mask.shape[1]), dtype=bool),
        n_pairs_check=n_loops,
        with_relaxation=False,
        max_column_diameter=death,
        column_trim_method=column_trim_method,
        column_scores=column_scores,
    )
    fillings: list[tuple[int, ...] | None] = [None] * n_loops
    for (loop_index, _), deformation in zip(
        result.loop_pairs_matched, result.mapping_deformation_matched
    ):
        fillings[loop_index] = deformation["triangle_ids"]
    return fillings


def compute_coherence(
    filling: Sequence[int] | None,
    deformation: Sequence[int],
    triangle_areas: Mapping[int, float],
) -> float | None:
    if filling is None:
        return None
    source = set(filling)
    homotopy = set(deformation)
    area_union = sum(triangle_areas[t] for t in source | homotopy)
    if area_union <= 0:
        return None
    # F ∩ (F + H) = F \ H and F ∪ (F + H) = F ∪ H over GF2
    # Based on Jaccard index
    return sum(triangle_areas[t] for t in source - homotopy) / area_union
