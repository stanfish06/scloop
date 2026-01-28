# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np
from numba import jit
from scipy.sparse import csr_matrix

__all__ = ["scatter_loop_edge_field_to_global", "compute_divergence_from_edge_field"]


@jit(nopython=True, cache=True)
def _scatter_single_rep(
    edge_vals: np.ndarray,
    mask: np.ndarray,
    signs: np.ndarray,
    global_edge: np.ndarray,
) -> None:
    # Note: this step assumes edge_vals are oriented in loop traversal direction
    edge_vals_simplicial_reorient = edge_vals.flatten() * signs.flatten()

    for edge_idx in range(len(edge_vals_simplicial_reorient)):
        val = edge_vals_simplicial_reorient[edge_idx]
        mask_row = mask[edge_idx]

        global_idx = -1
        for i in range(len(mask_row)):
            if mask_row[i]:
                global_idx = i
                break

        if global_idx < 0:
            continue

        if global_edge[global_idx] == 0:
            global_edge[global_idx] = val
        else:
            existing = global_edge[global_idx]
            if np.abs(val) > np.abs(existing):
                global_edge[global_idx] = val


def scatter_loop_edge_field_to_global(
    edge_values_per_rep: list[np.ndarray],
    edge_masks_per_rep: list[np.ndarray],
    edge_signs_per_rep: list[np.ndarray],
    n_global_edges: int,
) -> np.ndarray:
    global_edge = np.zeros(n_global_edges, dtype=np.float64)

    for rep_idx, edge_vals in enumerate(edge_values_per_rep):
        if edge_vals is None or len(edge_vals) == 0:
            continue

        mask = edge_masks_per_rep[rep_idx]
        signs = edge_signs_per_rep[rep_idx]

        _scatter_single_rep(
            edge_vals.astype(np.float64),
            mask.astype(np.bool_),
            signs.astype(np.float64),
            global_edge,
        )

    return global_edge


def compute_divergence_from_edge_field(
    boundary_matrix_d0: csr_matrix,
    edge_field: np.ndarray,
    negate_for_source_positive: bool = True,
) -> np.ndarray:
    div = boundary_matrix_d0 @ edge_field
    if negate_for_source_positive:
        div = -div
    return div
