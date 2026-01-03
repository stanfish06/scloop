# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np

from ..computing.homology import (
    compute_loop_geometric_distance,
    compute_loop_homological_equivalence,
)
from ..data.boundary import BoundaryMatrixD1
from ..data.constants import DEFAULT_N_PAIRS_CHECK
from ..data.types import Count_t, LoopDistMethod, PositiveFloat
from ..data.utils import loop_vertices_to_edge_ids_with_signs


def loops_to_edge_mask(
    loops: list[list[int]],
    boundary_matrix_d1: BoundaryMatrixD1,
    return_valid_indices: bool = False,
    use_order: bool = False,
) -> np.ndarray | tuple[np.ndarray, list[list[int]], list[np.ndarray]]:
    num_vertices = boundary_matrix_d1.num_vertices
    n_edges = boundary_matrix_d1.shape[0]

    edge_lookup = {
        int(edge_id): row_idx
        for row_idx, edge_id in enumerate(boundary_matrix_d1.row_simplex_ids)
    }

    dtype = np.int32 if use_order else bool
    mask = np.zeros((len(loops), n_edges), dtype=dtype)
    valid_indices_per_rep = []
    edge_signs_per_rep = []

    for idx, loop in enumerate(loops):
        edge_ids, edge_signs = loop_vertices_to_edge_ids_with_signs(
            np.asarray(loop, dtype=np.int64), num_vertices
        )
        valid_indices = []
        valid_signs = []
        seen_row_ids = set()
        order = 1
        for edge_idx, (eid, sign) in enumerate(zip(edge_ids, edge_signs)):
            row_id = edge_lookup.get(int(eid), -1)
            if row_id >= 0 and row_id not in seen_row_ids:
                mask[idx, row_id] = order if use_order else True
                order += 1
                valid_indices.append(edge_idx)
                valid_signs.append(sign)
                seen_row_ids.add(row_id)
        valid_indices_per_rep.append(valid_indices)
        edge_signs_per_rep.append(np.array(valid_signs, dtype=np.int8))

    if return_valid_indices:
        return mask, valid_indices_per_rep, edge_signs_per_rep
    return mask


def compute_geometric_distance(
    source_coords_list: list[list[list[float]]],
    target_coords_list: list[list[list[float]]],
    method: LoopDistMethod = "hausdorff",
    n_workers: Count_t = 1,
) -> float:
    distances_arr = compute_loop_geometric_distance(
        source_coords_list=source_coords_list,
        target_coords_list=target_coords_list,
        method=method,
        n_workers=n_workers,
    )
    return float(np.nanmean(distances_arr))


def check_homological_equivalence(
    source_loops: list[list[int]],
    target_loops: list[list[int]],
    boundary_matrix_d1: BoundaryMatrixD1,
    n_pairs_check: int = DEFAULT_N_PAIRS_CHECK,
    max_column_diameter: PositiveFloat | None = None,
) -> bool:
    if len(source_loops) == 0 or len(target_loops) == 0:
        return False

    mask_a = loops_to_edge_mask(source_loops, boundary_matrix_d1)
    mask_b = loops_to_edge_mask(target_loops, boundary_matrix_d1)

    assert isinstance(mask_a, np.ndarray)
    assert isinstance(mask_b, np.ndarray)

    results, _ = compute_loop_homological_equivalence(
        boundary_matrix_d1=boundary_matrix_d1,
        loop_mask_a=mask_a,
        loop_mask_b=mask_b,
        n_pairs_check=n_pairs_check,
        max_column_diameter=max_column_diameter,
    )
    return any(r == 0 for r in results)
