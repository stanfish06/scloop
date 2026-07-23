# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np

from ..computing.coherence import compute_coherence
from ..computing.homology import (
    compute_loop_geometric_distance,
    compute_loop_homological_equivalence,
)
from ..data.base_components import LoopClassEquivalence
from ..data.boundary import BoundaryMatrixD1
from ..data.constants import (
    DEFAULT_COLUMN_TRIM_METHOD,
    DEFAULT_MAX_N_EDGES_RELAXATION_EQUIVALENCE,
    DEFAULT_N_HUBS_RELAXATION_EQUIVALENCE,
    DEFAULT_N_NEIGHBORS_COLUMN_TRIM,
    DEFAULT_N_PAIRS_CHECK,
    DEFAULT_WITH_RELAXATION_EQUIVALENCE,
)
from ..data.types import (
    ColumnTrimMethod,
    Count_t,
    HomotopyCoherenceMethod,
    LoopDistMethod,
    LoopEdges,
    PositiveFloat,
)
from ..data.utils import loop_vertices_to_edge_ids_with_signs


def loops_to_edge_mask(
    loops: list[list[int]],
    boundary_matrix_d1: BoundaryMatrixD1,
    return_valid_indices: bool = False,
    use_order: bool = False,
) -> np.ndarray | LoopEdges:
    num_vertices = boundary_matrix_d1.num_vertices
    n_edges = boundary_matrix_d1.shape[0]

    edge_lookup = {
        edge_id: row_idx
        for row_idx, edge_id in enumerate(boundary_matrix_d1.row_simplex_ids)
    }

    dtype = np.int32 if use_order else bool
    mask = np.zeros((len(loops), n_edges), dtype=dtype)
    valid_indices_per_rep = []
    valid_edge_ids_per_rep = []
    valid_edge_signs_per_rep = []

    for idx, loop in enumerate(loops):
        loop_arr = np.asarray(loop, dtype=np.int64)
        edge_ids, edge_signs = loop_vertices_to_edge_ids_with_signs(
            loop_arr, num_vertices
        )
        valid_indices = []
        valid_ids = []
        valid_signs = []
        seen_row_ids = set()
        order = 1
        for edge_idx, (eid, sign) in enumerate(zip(edge_ids, edge_signs)):
            row_id = edge_lookup.get(int(eid), -1)
            if row_id >= 0 and row_id not in seen_row_ids:
                mask[idx, row_id] = order if use_order else True
                order += 1
                valid_indices.append(edge_idx)
                valid_ids.append(eid)
                valid_signs.append(sign)
                seen_row_ids.add(row_id)
        valid_indices_per_rep.append(valid_indices)
        valid_edge_ids_per_rep.append(valid_ids)
        valid_edge_signs_per_rep.append(np.array(valid_signs, dtype=np.int8))

    if return_valid_indices:
        return LoopEdges(
            mask=mask,
            indices_per_rep=valid_indices_per_rep,
            edge_ids_per_rep=valid_edge_ids_per_rep,
            edge_signs_per_rep=valid_edge_signs_per_rep,
        )
    return mask


def cocycle_to_edge_mask(
    cocycle: list,
    boundary_matrix_d1: BoundaryMatrixD1,
    vertex_ids: list[int],
) -> np.ndarray | None:
    if not cocycle:
        return None
    num_vertices = boundary_matrix_d1.num_vertices
    n_edges = boundary_matrix_d1.shape[0]
    edge_lookup = {
        edge_id: row_idx
        for row_idx, edge_id in enumerate(boundary_matrix_d1.row_simplex_ids)
    }
    mask = np.zeros(n_edges, dtype=bool)
    for simplex in cocycle:
        try:
            verts, coeff = simplex
        except (ValueError, TypeError):
            continue
        if coeff % 2 == 0 or len(verts) != 2:
            continue
        u_local = int(verts[0])
        v_local = int(verts[1])
        if u_local < 0 or v_local < 0:
            continue
        if u_local >= len(vertex_ids) or v_local >= len(vertex_ids):
            continue
        u_global = vertex_ids[u_local]
        v_global = vertex_ids[v_local]
        if u_global == v_global:
            continue
        if u_global > v_global:
            u_global, v_global = v_global, u_global
        edge_id = u_global * num_vertices + v_global
        row_id = edge_lookup.get(edge_id, -1)
        if row_id >= 0:
            mask[row_id] = not mask[row_id]
    if not mask.any():
        return None
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
    with_relaxation: bool = DEFAULT_WITH_RELAXATION_EQUIVALENCE,
    n_hubs_relaxation: int = DEFAULT_N_HUBS_RELAXATION_EQUIVALENCE,
    max_n_edges_relaxation: int = DEFAULT_MAX_N_EDGES_RELAXATION_EQUIVALENCE,
    max_column_diameter: PositiveFloat | None = None,
    cocycle_edge_mask: np.ndarray | None = None,
    compute_homotopy_coherence: bool = True,
    homotopy_coherence_method: HomotopyCoherenceMethod = "path_finding",
    max_triangles_homotopy_coherence: int = 18,
    column_trim_method: ColumnTrimMethod = DEFAULT_COLUMN_TRIM_METHOD,
    embedding: np.ndarray | None = None,
    n_neighbors_column_trim: int = DEFAULT_N_NEIGHBORS_COLUMN_TRIM,
) -> LoopClassEquivalence:
    if len(source_loops) == 0 or len(target_loops) == 0:
        return LoopClassEquivalence()

    loop_edges_a = loops_to_edge_mask(
        loops=source_loops,
        boundary_matrix_d1=boundary_matrix_d1,
        return_valid_indices=compute_homotopy_coherence,
    )
    loop_edges_b = loops_to_edge_mask(
        loops=target_loops,
        boundary_matrix_d1=boundary_matrix_d1,
        return_valid_indices=compute_homotopy_coherence,
    )

    mask_a = loop_edges_a.mask if isinstance(loop_edges_a, LoopEdges) else loop_edges_a
    mask_b = loop_edges_b.mask if isinstance(loop_edges_b, LoopEdges) else loop_edges_b

    loop_vertex_ids = None
    if column_trim_method == "loop_proximity":
        ids: list[int] = []
        for loop in source_loops:
            ids.extend(int(v) for v in loop)
        for loop in target_loops:
            ids.extend(int(v) for v in loop)
        loop_vertex_ids = np.unique(np.asarray(ids, dtype=np.int64))

    result = compute_loop_homological_equivalence(
        boundary_matrix_d1=boundary_matrix_d1,
        loop_mask_a=mask_a,
        loop_mask_b=mask_b,
        n_pairs_check=n_pairs_check,
        with_relaxation=with_relaxation,
        n_hubs_relaxation=n_hubs_relaxation,
        max_n_edges_relaxation=max_n_edges_relaxation,
        max_column_diameter=max_column_diameter,
        cocycle_edge_mask=cocycle_edge_mask,
        column_trim_method=column_trim_method,
        embedding=embedding,
        loop_vertex_ids=loop_vertex_ids,
        n_neighbors_column_trim=n_neighbors_column_trim,
    )

    if not compute_homotopy_coherence:
        return result

    assert isinstance(loop_edges_a, LoopEdges)
    assert isinstance(loop_edges_b, LoopEdges)
    row_edge_ids = np.asarray(boundary_matrix_d1.row_simplex_ids, dtype=int)
    edge_lengths = dict(
        zip(row_edge_ids.tolist(), boundary_matrix_d1.row_simplex_diams)
    )
    row_indices = np.asarray(boundary_matrix_d1.data[0], dtype=int)
    column_indices = np.asarray(boundary_matrix_d1.data[1], dtype=int)
    triangle_edges = {
        triangle_id: tuple(row_edge_ids[row_indices[column_indices == column]].tolist())
        for column, triangle_id in enumerate(boundary_matrix_d1.col_simplex_ids)
    }

    for (source_index, target_index), deformation in zip(
        result.loop_pairs_matched,
        result.mapping_deformation_matched,
    ):
        result.homotopy_coherence_matched.append(
            compute_coherence(
                source_edges=loop_edges_a.edge_ids_per_rep[source_index],
                target_edges=loop_edges_b.edge_ids_per_rep[target_index],
                triangles=[
                    triangle_edges[triangle_id]
                    for triangle_id in deformation["triangle_ids"]
                ],
                edge_lengths=edge_lengths,
                num_vertices=boundary_matrix_d1.num_vertices,
                method=homotopy_coherence_method,
                max_triangles=max_triangles_homotopy_coherence,
            )
        )

    for (source_index, target_index), deformation in zip(
        result.loop_pairs_matched_relax,
        result.mapping_deformation_matched_relax,
    ):
        source = set(loop_edges_a.edge_ids_per_rep[source_index])
        target = set(loop_edges_b.edge_ids_per_rep[target_index])
        relaxation_edges = set(deformation["relaxation_edge_ids"])
        triangles = [
            triangle_edges[triangle_id] for triangle_id in deformation["triangle_ids"]
        ]
        scores = [
            compute_coherence(
                source_edges=tuple(source ^ relaxation_edges),
                target_edges=tuple(target),
                triangles=triangles,
                edge_lengths=edge_lengths,
                num_vertices=boundary_matrix_d1.num_vertices,
                method=homotopy_coherence_method,
                max_triangles=max_triangles_homotopy_coherence,
            ),
            compute_coherence(
                source_edges=tuple(source),
                target_edges=tuple(target ^ relaxation_edges),
                triangles=triangles,
                edge_lengths=edge_lengths,
                num_vertices=boundary_matrix_d1.num_vertices,
                method=homotopy_coherence_method,
                max_triangles=max_triangles_homotopy_coherence,
            ),
        ]
        valid_scores = [score for score in scores if score is not None]
        result.homotopy_coherence_matched_relax.append(
            float(np.mean(valid_scores)) if valid_scores else None
        )

    return result
