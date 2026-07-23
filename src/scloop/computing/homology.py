# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING, Literal

import numpy as np
from anndata import AnnData
from numba import jit
from scipy.sparse import csr_matrix
from scipy.spatial.distance import cdist, directed_hausdorff
from sklearn.neighbors import radius_neighbors_graph

from ..data.base_components import LoopClassEquivalence, PersistencePair
from ..data.constants import (
    DEFAULT_COLUMN_TRIM_METHOD,
    DEFAULT_LOOP_DIST_METHOD,
    DEFAULT_N_MAX_WORKERS,
    DEFAULT_N_NEIGHBORS_COLUMN_TRIM,
)
from ..data.metadata import ScloopMeta
from ..data.ripser_lib import (  # type: ignore[import-not-found]
    get_boundary_matrix,
    ripser,
    ripser_image,
)
from ..data.types import (
    ColumnTrimMethod,
    Count_t,
    Diameter_t,
    IndexListDistMatrix,
    LoopDistMethod,
    Percent_t,
)
from ..data.utils import encode_triangles_and_edges
from ..preprocessing.delve.kh import kernel_herding_main
from ..preprocessing.downsample import (
    sample_farthest_points,
    sample_farthest_points_randomized,
)
from ..utils.denoise.Sanity_py import sample_posterior_predictive_counts
from ..utils.distance_metrics.frechet_py import compute_pairwise_loop_frechet
from ..utils.linear_algebra_gf2 import (  # type: ignore
    solve_multiple_gf2_m4ri,  # type: ignore[import-not-found]
)

if TYPE_CHECKING:
    from ..data.boundary import BoundaryMatrixD1


@dataclass
class ImageBootstrapHomologyResult:
    persistence_diagram: list
    persistence_pair_simplices: list
    cocycles: list
    indices_resample: list[int]
    bootstrap_distance_matrix: csr_matrix
    reference_image_result: object
    bootstrap_image_result: object
    n_reference_vertices: int


@dataclass
class CrossDatasetImageHomologyResult:
    source_image_result: object
    target_image_result: object
    n_source_vertices: int


def _cap_infinite_deaths(diagrams: list, cap: float | None) -> list:
    if cap is None or not np.isfinite(cap):
        return diagrams
    capped = []
    for dim_pd in diagrams:
        if len(dim_pd) < 2:
            capped.append(dim_pd)
            continue
        births = np.asarray(dim_pd[0])
        deaths = np.asarray(dim_pd[1])
        deaths = np.where(np.isinf(deaths), cap, deaths)
        capped.append([births, deaths])
    return capped


def _persistence_pairs_from_ripser_result(
    result: object, dim: int = 1
) -> list[PersistencePair]:
    births, deaths = result.births_and_deaths_by_dim[dim]
    birth_simplices, death_simplices = result.births_and_deaths_simplex_by_dim[dim]
    if not (len(births) == len(deaths) == len(birth_simplices) == len(death_simplices)):
        raise ValueError(
            "Ripser persistence pairs and critical simplices are misaligned"
        )
    return [
        PersistencePair(
            birth=float(birth),
            death=float(death),
            birth_simplex=list(birth_simplex),
            death_simplex=list(death_simplex),
        )
        for birth, death, birth_simplex, death_simplex in zip(
            births, deaths, birth_simplices, death_simplices
        )
    ]


def _sample_bootstrap_embedding(
    adata: AnnData,
    meta: ScloopMeta,
    source_indices: list[int],
    bootstrap_indices: list[int],
    bootstrap_noise_model: str,
    noise_scale: float,
    sanity_n_posterior: int = 1000,
) -> np.ndarray:
    assert meta.preprocess is not None
    assert meta.preprocess.embedding_method is not None

    emb = np.asarray(adata.obsm[f"X_{meta.preprocess.embedding_method}"])

    X = emb[source_indices]
    if bootstrap_noise_model == "sanity":
        if meta.preprocess.embedding_method == "pca":
            X = sample_posterior_predictive_counts(
                adata=adata,
                cell_idx=np.asarray(bootstrap_indices, dtype=np.int64),
                scale_before_pca=meta.preprocess.scale_before_pca,
                n_pca_comps=meta.preprocess.n_pca_comps,
                n_posterior=sanity_n_posterior,
                ltq_var_scale=noise_scale,
            )
        elif meta.preprocess.embedding_method == "diffmap":
            if meta.preprocess.embedding_neighbors == "pca":
                X = sample_posterior_predictive_counts(
                    adata=adata,
                    cell_idx=np.asarray(bootstrap_indices, dtype=np.int64),
                    scale_before_pca=meta.preprocess.scale_before_pca,
                    n_pca_comps=meta.preprocess.n_pca_comps,
                    n_posterior=sanity_n_posterior,
                    ltq_var_scale=noise_scale,
                )
                assert meta.preprocess.diffmap_operator
                diffmap_operator = meta.preprocess.diffmap_operator
                emb_reference = np.asarray(
                    adata.obsm[f"X_{meta.preprocess.embedding_neighbors}"]
                )
                X = diffmap_operator.project_query_data(
                    emb_reference=emb_reference, emb_query=X
                )
    else:
        X_ref = emb[source_indices]
        X = emb[bootstrap_indices]
        std_X = np.std(X_ref, axis=0)
        X = X + np.random.normal(scale=std_X * noise_scale, size=X.shape)

    return X


def select_bootstrap_sample(
    source_indices: list[int],
    source_embedding: np.ndarray,
    bootstrap_sampling: Literal[
        "resample", "downsample", "fps", "fps_random", "herding"
    ] = "resample",
    bootstrap_downsample_fraction: Percent_t = 2 / 3,
    bootstrap_fps_top_k: int = 5,
    bootstrap_fps_alpha: float = 1.0,
    bootstrap_herding_n_features: int = 1000,
    bootstrap_herding_seed: int | None = None,
):
    match bootstrap_sampling:
        case "resample" | "downsample":
            sample_idx = np.random.choice(
                len(source_indices),
                size=len(source_indices)
                if bootstrap_sampling == "resample"
                else int(len(source_indices) * bootstrap_downsample_fraction),
                replace=True if bootstrap_sampling == "resample" else False,
            )
        case "fps":
            n_keep = max(
                2, int(round(len(source_indices) * bootstrap_downsample_fraction))
            )
            n_keep = min(n_keep, len(source_indices))
            sample_idx = sample_farthest_points(source_embedding, n_keep)
        case "fps_random":
            if bootstrap_fps_top_k <= 0:
                raise ValueError("bootstrap_fps_top_k must be > 0.")
            if bootstrap_fps_alpha < 0:
                raise ValueError("bootstrap_fps_alpha must be >= 0.")
            n_keep = max(
                2, int(round(len(source_indices) * bootstrap_downsample_fraction))
            )
            n_keep = min(n_keep, len(source_indices))
            sample_idx = sample_farthest_points_randomized(
                source_embedding,
                n_keep,
                top_k=bootstrap_fps_top_k,
                alpha=bootstrap_fps_alpha,
            )
        case "herding":
            n_keep = max(
                2, int(round(len(source_indices) * bootstrap_downsample_fraction))
            )
            n_keep = min(n_keep, len(source_indices))
            if bootstrap_herding_seed is None:
                bootstrap_herding_seed = int(np.random.randint(0, 1_000_000))
            sample_idx = kernel_herding_main(
                sample_set_ind=np.arange(len(source_indices)),
                X=source_embedding,
                num_subsamples=n_keep,
                frequency_seed=bootstrap_herding_seed,
                n_features=int(bootstrap_herding_n_features),
            )
    return [source_indices[int(i)] for i in sample_idx.tolist()]


def compute_sparse_pairwise_distance(
    adata: AnnData,
    meta: ScloopMeta,
    bootstrap: bool = False,
    noise_scale: float = 1e-3,
    sanity_n_posterior: int = 1000,
    bootstrap_noise_model: str = "gaussian",
    thresh: Diameter_t | None = None,
    bootstrap_sampling: Literal[
        "resample", "downsample", "fps", "fps_random", "herding"
    ] = "resample",
    bootstrap_downsample_fraction: Percent_t = 2 / 3,
    bootstrap_fps_top_k: int = 5,
    bootstrap_fps_alpha: float = 1.0,
    bootstrap_herding_n_features: int = 1000,
    bootstrap_herding_seed: int | None = None,
    **nei_kwargs,
) -> tuple[csr_matrix, IndexListDistMatrix | None]:
    # important, default is binary graph
    nei_kwargs.setdefault("mode", "distance")
    assert meta.preprocess is not None
    assert meta.preprocess.embedding_method is not None
    emb = adata.obsm[f"X_{meta.preprocess.embedding_method}"]
    selected_indices = (
        meta.preprocess.indices_downsample
        if meta.preprocess.indices_downsample is not None
        else list(range(emb.shape[0]))
    )
    X = emb[selected_indices]
    boot_idx = None
    if bootstrap:
        boot_idx = select_bootstrap_sample(
            source_indices=selected_indices,
            source_embedding=X,
            bootstrap_sampling=bootstrap_sampling,
            bootstrap_downsample_fraction=bootstrap_downsample_fraction,
            bootstrap_fps_top_k=bootstrap_fps_top_k,
            bootstrap_fps_alpha=bootstrap_fps_alpha,
            bootstrap_herding_n_features=bootstrap_herding_n_features,
            bootstrap_herding_seed=bootstrap_herding_seed,
        )
        X = _sample_bootstrap_embedding(
            adata=adata,
            meta=meta,
            source_indices=selected_indices,
            bootstrap_indices=boot_idx,
            bootstrap_noise_model=bootstrap_noise_model,
            noise_scale=noise_scale,
            sanity_n_posterior=sanity_n_posterior,
        )
    else:
        boot_idx = selected_indices
    return (
        radius_neighbors_graph(
            X=X,
            radius=thresh,
            **nei_kwargs,
        ),
        boot_idx,
    )


def _remap_subfiltration_simplices_to_local(
    simplices_by_dim: list, vertex_offset: int
) -> list:
    remapped = []
    for births, deaths in simplices_by_dim:
        remapped.append(
            [
                [[int(v) - vertex_offset for v in simplex] for simplex in births],
                [[int(v) - vertex_offset for v in simplex] for simplex in deaths],
            ]
        )
    return remapped


def _remap_subfiltration_cocycles_to_local(
    cocycles_by_dim: list, vertex_offset: int
) -> list:
    remapped = []
    for dim_cocycles in cocycles_by_dim:
        remapped_dim = []
        for cocycle in dim_cocycles:
            remapped_dim.append(
                [
                    [
                        [int(v) - vertex_offset for v in simplex_vertices],
                        int(coefficient),
                    ]
                    for simplex_vertices, coefficient in cocycle
                ]
            )
        remapped.append(remapped_dim)
    return remapped


def compute_image_bootstrap_homology(
    adata: AnnData,
    meta: ScloopMeta,
    thresh: Diameter_t,
    noise_scale: float = 1e-3,
    sanity_n_posterior: int = 1000,
    bootstrap_noise_model: str = "gaussian",
    bootstrap_sampling: Literal[
        "resample", "downsample", "fps", "fps_random", "herding"
    ] = "resample",
    bootstrap_downsample_fraction: Percent_t = 2 / 3,
    bootstrap_fps_top_k: int = 5,
    bootstrap_fps_alpha: float = 1.0,
    bootstrap_herding_n_features: int = 1000,
    bootstrap_herding_seed: int | None = None,
    **nei_kwargs,
) -> ImageBootstrapHomologyResult:
    assert meta.preprocess is not None
    assert meta.preprocess.embedding_method is not None

    embedding = np.asarray(adata.obsm[f"X_{meta.preprocess.embedding_method}"])
    source_indices = (
        meta.preprocess.indices_downsample
        if meta.preprocess.indices_downsample is not None
        else list(range(embedding.shape[0]))
    )
    source_embedding = embedding[source_indices]
    bootstrap_indices = select_bootstrap_sample(
        source_indices=source_indices,
        source_embedding=source_embedding,
        bootstrap_sampling=bootstrap_sampling,
        bootstrap_downsample_fraction=bootstrap_downsample_fraction,
        bootstrap_fps_top_k=bootstrap_fps_top_k,
        bootstrap_fps_alpha=bootstrap_fps_alpha,
        bootstrap_herding_n_features=bootstrap_herding_n_features,
        bootstrap_herding_seed=bootstrap_herding_seed,
    )
    bootstrap_embedding = _sample_bootstrap_embedding(
        adata=adata,
        meta=meta,
        source_indices=source_indices,
        bootstrap_indices=bootstrap_indices,
        bootstrap_noise_model=bootstrap_noise_model,
        noise_scale=noise_scale,
        sanity_n_posterior=sanity_n_posterior,
    )

    n_reference_vertices = len(source_indices)
    union_embedding = np.vstack([source_embedding, bootstrap_embedding])
    nei_kwargs.setdefault("mode", "distance")
    union_distance_matrix = radius_neighbors_graph(
        X=union_embedding,
        radius=thresh,
        **nei_kwargs,
    ).tocsr()
    bootstrap_offset = n_reference_vertices
    bootstrap_union_indices = np.arange(
        bootstrap_offset,
        bootstrap_offset + len(bootstrap_indices),
        dtype=np.intc,
    )

    reference_image_result = ripser_image(
        distance_matrix=union_distance_matrix.tocoo(copy=False),
        sub_indices=np.arange(n_reference_vertices, dtype=np.intc),
        modulus=2,
        dim_max=1,
        threshold=thresh,
        do_subfiltration_cocycles=False,
    )
    bootstrap_image_result = ripser_image(
        distance_matrix=union_distance_matrix.tocoo(copy=False),
        sub_indices=bootstrap_union_indices,
        modulus=2,
        dim_max=1,
        threshold=thresh,
        do_subfiltration_cocycles=True,
    )
    bootstrap_subfiltration = bootstrap_image_result.subfiltration
    persistence_pair_simplices = _remap_subfiltration_simplices_to_local(
        bootstrap_subfiltration.births_and_deaths_simplex_by_dim,
        bootstrap_offset,
    )
    cocycles = _remap_subfiltration_cocycles_to_local(
        bootstrap_subfiltration.cocycles_by_dim,
        bootstrap_offset,
    )

    return ImageBootstrapHomologyResult(
        persistence_diagram=_cap_infinite_deaths(
            bootstrap_subfiltration.births_and_deaths_by_dim, thresh
        ),
        persistence_pair_simplices=persistence_pair_simplices,
        cocycles=cocycles,
        indices_resample=bootstrap_indices,
        bootstrap_distance_matrix=union_distance_matrix[
            bootstrap_offset:, bootstrap_offset:
        ].tocsr(),
        reference_image_result=reference_image_result,
        bootstrap_image_result=bootstrap_image_result,
        n_reference_vertices=n_reference_vertices,
    )


def compute_cross_dataset_image_homology(
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
    thresh: Diameter_t,
    dim_max: int = 1,
    **nei_kwargs,
) -> CrossDatasetImageHomologyResult:
    n_source_vertices = len(source_embedding)
    union_embedding = np.vstack([source_embedding, target_embedding])
    nei_kwargs.setdefault("mode", "distance")
    union_distance_matrix = radius_neighbors_graph(
        X=union_embedding,
        radius=thresh,
        **nei_kwargs,
    ).tocsr()

    source_indices = np.arange(n_source_vertices, dtype=np.intc)
    target_indices = np.arange(n_source_vertices, len(union_embedding), dtype=np.intc)

    source_image_result = ripser_image(
        distance_matrix=union_distance_matrix.tocoo(copy=False),
        sub_indices=source_indices,
        modulus=2,
        dim_max=dim_max,
        threshold=thresh,
        do_subfiltration_cocycles=False,
        do_image_cocycles=True,
    )
    target_image_result = ripser_image(
        distance_matrix=union_distance_matrix.tocoo(copy=False),
        sub_indices=target_indices,
        modulus=2,
        dim_max=dim_max,
        threshold=thresh,
        do_subfiltration_cocycles=False,
        do_image_cocycles=True,
    )

    return CrossDatasetImageHomologyResult(
        source_image_result=source_image_result,
        target_image_result=target_image_result,
        n_source_vertices=n_source_vertices,
    )


def compute_persistence_diagram_and_cocycles(
    adata: AnnData,
    meta: ScloopMeta,
    thresh: Diameter_t | None = None,
    bootstrap: bool = False,
    noise_scale: float = 1e-3,
    **nei_kwargs,
) -> tuple[list, list, list, IndexListDistMatrix | None, csr_matrix]:
    sparse_pairwise_distance_matrix, boot_idx = compute_sparse_pairwise_distance(
        adata=adata,
        meta=meta,
        bootstrap=bootstrap,
        noise_scale=noise_scale,
        thresh=thresh,
        **nei_kwargs,
    )
    result = ripser(
        distance_matrix=sparse_pairwise_distance_matrix.tocoo(copy=False),
        modulus=2,
        dim_max=1,
        threshold=thresh,
        do_cocycles=True,
    )
    return (
        _cap_infinite_deaths(result.births_and_deaths_by_dim, thresh),
        result.births_and_deaths_simplex_by_dim,
        result.cocycles_by_dim,
        boot_idx,
        sparse_pairwise_distance_matrix,
    )


def compute_boundary_matrix_data(
    adata: AnnData, meta: ScloopMeta, thresh: Diameter_t | None = None, **nei_kwargs
) -> tuple:
    """
    Compute both D0 and D1 boundary matrices
    """
    assert meta.preprocess is not None
    assert meta.preprocess.num_vertices is not None
    sparse_pairwise_distance_matrix, vertex_indices = compute_sparse_pairwise_distance(
        adata=adata, meta=meta, bootstrap=False, thresh=thresh, **nei_kwargs
    )
    result = get_boundary_matrix(sparse_pairwise_distance_matrix.tocoo(), thresh)
    triangles_local = np.asarray(result.triangle_vertices, dtype=np.int64)
    if len(triangles_local) == 0:
        edge_ids, trig_ids, edge_diameters, vertex_indices_np = [], [], [], np.array([])
    else:
        if vertex_indices is None:
            assert sparse_pairwise_distance_matrix.shape is not None
            vertex_indices_np = np.arange(sparse_pairwise_distance_matrix.shape[0])
        else:
            vertex_indices_np = np.asarray(vertex_indices, dtype=np.int64)
        # important: must convert triangle vertex ids to global indices
        triangles = vertex_indices_np[triangles_local]
        # CRITICAL: sort triangle vertices to ensure consistent orientation
        # Without this, d1 @ d2 != 0 when downsampling is used
        triangles = np.sort(triangles, axis=1)
        # NOTE: edges and triangles are encoded based on the total number of vertices, not the downsampled number
        edge_ids, trig_ids = encode_triangles_and_edges(
            triangles, meta.preprocess.num_vertices
        )
        edge_diameters = []
        for tri_local in triangles_local:
            i0, i1, i2 = int(tri_local[0]), int(tri_local[1]), int(tri_local[2])
            edge_diameters.extend(
                [
                    sparse_pairwise_distance_matrix[i0, i1],
                    sparse_pairwise_distance_matrix[i0, i2],
                    sparse_pairwise_distance_matrix[i1, i2],
                ]
            )
    return (
        result,
        edge_ids,
        trig_ids,
        edge_diameters,
        sparse_pairwise_distance_matrix,
        vertex_indices_np.tolist(),
    )


@jit(nopython=True, cache=True)
def find_incident_edges(
    one_ridx_A: np.ndarray,
    ncol_A: int,
    b: np.ndarray,
    n_hubs: int,
):
    columns_include = np.zeros(ncol_A, dtype=np.bool_)
    rows_include = b.copy()
    for _ in range(n_hubs):
        for i in range(ncol_A):
            if columns_include[i]:
                continue
            one_ridx_col_i = one_ridx_A[3 * i : 3 * (i + 1)]
            if rows_include[one_ridx_col_i].any():
                rows_include[one_ridx_col_i] = True
                columns_include[i] = True
    return rows_include ^ b


def _decode_deformation(
    solution: list[int], column_ids: list[int], columns_are_triangles: list[bool]
) -> dict[str, tuple[int, ...]]:
    selected = [
        (simplex_id, is_triangle)
        for value, simplex_id, is_triangle in zip(
            solution, column_ids, columns_are_triangles
        )
        if value % 2 == 1
    ]
    return {
        "triangle_ids": tuple(
            simplex_id for simplex_id, is_triangle in selected if is_triangle
        ),
        "relaxation_edge_ids": tuple(
            simplex_id for simplex_id, is_triangle in selected if not is_triangle
        ),
    }


def _loop_proximity_column_scores(
    centroids: np.ndarray,
    loop_coords: np.ndarray,
    n_neighbors: int,
) -> np.ndarray:
    """Mean distance from each triangle centroid to its k nearest loop points."""
    n_tri = centroids.shape[0]
    n_loop = loop_coords.shape[0]
    if n_tri == 0:
        return np.empty(0, dtype=np.float64)
    if n_loop == 0:
        return np.full(n_tri, np.inf, dtype=np.float64)

    k = max(1, min(int(n_neighbors), n_loop))
    if n_loop <= 64:
        d = cdist(centroids, loop_coords)
        if k == n_loop:
            return d.mean(axis=1)
        idx = np.argpartition(d, kth=k - 1, axis=1)[:, :k]
        return np.take_along_axis(d, idx, axis=1).mean(axis=1)

    from pynndescent import NNDescent

    index = NNDescent(loop_coords, n_neighbors=k)
    _, dists = index.query(centroids, k=k)
    return np.asarray(dists, dtype=np.float64).mean(axis=1)


def compute_loop_homological_equivalence(
    boundary_matrix_d1: "BoundaryMatrixD1",
    loop_mask_a: np.ndarray,
    loop_mask_b: np.ndarray,
    n_pairs_check: int = 3,
    with_relaxation: bool = True,
    n_hubs_relaxation: int = 2,
    max_n_edges_relaxation: int = 50,
    max_column_diameter: float | None = None,
    cocycle_edge_mask: np.ndarray | None = None,
    column_trim_method: ColumnTrimMethod = DEFAULT_COLUMN_TRIM_METHOD,
    embedding: np.ndarray | None = None,
    loop_vertex_ids: np.ndarray | None = None,
    n_neighbors_column_trim: int = DEFAULT_N_NEIGHBORS_COLUMN_TRIM,
) -> LoopClassEquivalence:
    """
    Parameters
    ---------
    loop_mask_a: np.ndarray
        Boolean mask of shape (n_a, n_edges); True where edge (row) is in the loop
    loop_mask_b: np.ndarray
        Boolean mask of shape (n_b, n_edges)
    max_column_diameter: float | None
        If provided, restrict the boundary matrix to columns (triangles) with diameter
        no larger than this value.
    column_trim_method: {"diameter", "loop_proximity"}
        When more triangle columns remain than edge rows, trim to ``nrow`` columns.
        ``diameter`` keeps the largest-diameter triangles (legacy).
        ``loop_proximity`` keeps triangles whose centroids are nearest to the
        loop vertices (requires ``embedding`` and ``loop_vertex_ids``).
    """
    assert loop_mask_a.shape[1] == boundary_matrix_d1.shape[0]
    assert loop_mask_b.shape[1] == boundary_matrix_d1.shape[0]
    if column_trim_method not in ("diameter", "loop_proximity"):
        raise ValueError(f"unknown column_trim_method: {column_trim_method}")
    if column_trim_method == "loop_proximity":
        if embedding is None:
            raise ValueError("column_trim_method='loop_proximity' requires embedding")
        if loop_vertex_ids is None:
            raise ValueError(
                "column_trim_method='loop_proximity' requires loop_vertex_ids"
            )

    # in F2, sum is just xor
    loop_sums = loop_mask_a[:, None, :] ^ loop_mask_b[None, :, :]
    loop_sums = loop_sums.reshape(-1, loop_sums.shape[-1])
    result = LoopClassEquivalence()
    if loop_sums.shape[0] == 0:
        return result
    n_a = loop_mask_a.shape[0]
    n_b = loop_mask_b.shape[0]
    pairs_kept = [(i, j) for i in range(n_a) for j in range(n_b)]
    # early stoping, if sum is not a boundary according to cocycle, then skip it
    if cocycle_edge_mask is not None:
        mask = cocycle_edge_mask.astype(bool)
        if mask.shape[0] != loop_sums.shape[1]:
            return result
        if mask.any():
            keep = np.where((loop_sums[:, mask].sum(axis=1) % 2) == 0)[0]
            result.n_loop_pairs_checked = loop_sums.shape[0]
            if len(keep) == 0:
                return result
            loop_sums = loop_sums[keep]
            pairs_kept = [pairs_kept[i] for i in keep]
    n_pairs_check = min(n_pairs_check, loop_sums.shape[0])
    pairs_kept = pairs_kept[:n_pairs_check]
    one_idx_b_list = [
        np.flatnonzero(loop_sums[i]).astype(int).tolist() for i in range(n_pairs_check)
    ]
    result.n_loop_pairs_checked = n_pairs_check

    one_ridx_A = np.asarray(boundary_matrix_d1.data[0], dtype=int)
    one_cidx_A = np.asarray(boundary_matrix_d1.data[1], dtype=int)
    nrow_A = boundary_matrix_d1.shape[0]
    ncol_A = boundary_matrix_d1.shape[1]

    col_diams = np.asarray(boundary_matrix_d1.col_simplex_diams, dtype=float)
    col_simplex_ids = np.asarray(boundary_matrix_d1.col_simplex_ids, dtype=int)
    row_simplex_ids = np.asarray(boundary_matrix_d1.row_simplex_ids, dtype=int)
    column_rows = [one_ridx_A[one_cidx_A == i] for i in range(ncol_A)]
    if any(len(rows) != 3 for rows in column_rows):
        raise ValueError("every D1 boundary-matrix column must contain three edges")

    cols_keep = np.arange(ncol_A, dtype=int)
    if max_column_diameter is not None:
        cols_keep = cols_keep[col_diams <= max_column_diameter]
        if cols_keep.size == 0:
            return result

    if cols_keep.size > nrow_A:
        if column_trim_method == "loop_proximity":
            assert embedding is not None and loop_vertex_ids is not None
            centroids = boundary_matrix_d1.compute_col_centroids(embedding)[cols_keep]
            loop_ids = np.unique(np.asarray(loop_vertex_ids, dtype=np.int64))
            loop_coords = np.asarray(embedding)[loop_ids].astype(np.float64, copy=False)
            scores = _loop_proximity_column_scores(
                centroids=centroids,
                loop_coords=loop_coords,
                n_neighbors=n_neighbors_column_trim,
            )
            # keep nearest (lowest score); stable for ties
            keep_order = np.argsort(scores, kind="stable")[:nrow_A]
            cols_keep = cols_keep[keep_order]
        else:
            # legacy: keep the larger remaining triangles
            keep_order = np.argsort(col_diams[cols_keep], kind="stable")[-nrow_A:]
            cols_keep = cols_keep[keep_order]

    # order columns by increasing diameter (relaxation replaces the largest tail)
    cols_keep = cols_keep[np.argsort(col_diams[cols_keep], kind="stable")]
    ncol_A = int(cols_keep.size)
    one_ridx_A = (
        np.concatenate([column_rows[i] for i in cols_keep])
        if ncol_A > 0
        else np.empty(0, dtype=int)
    )
    one_cidx_A = np.repeat(np.arange(ncol_A, dtype=int), 3)
    solver_column_ids = [int(col_simplex_ids[i]) for i in cols_keep]
    solver_columns_are_triangles = [True] * ncol_A

    states, solutions = solve_multiple_gf2_m4ri(
        one_ridx_A=one_ridx_A.tolist(),
        one_cidx_A=one_cidx_A.tolist(),
        nrow_A=nrow_A,
        ncol_A=ncol_A,
        one_idx_b_list=one_idx_b_list,
    )
    solved = [i for i, s in enumerate(states) if s == 0]
    if len(solved) > 0:
        result.loop_pairs_matched = [pairs_kept[i] for i in solved]
        result.mapping_deformation_matched = [
            _decode_deformation(
                solutions[i], solver_column_ids, solver_columns_are_triangles
            )
            for i in solved
        ]

    states_relax, solutions_relax = None, None
    if with_relaxation:
        max_n_edges_relaxation = min(max_n_edges_relaxation, ncol_A)
        n_hubs_edges = np.where(
            np.logical_or.reduce(
                [
                    find_incident_edges(
                        one_ridx_A=one_ridx_A,
                        ncol_A=ncol_A,
                        b=loop_sums[i],
                        n_hubs=n_hubs_relaxation,
                    )
                    for i in range(n_pairs_check)
                ]
            )
        )[0]
        # replace last columns (already sorted by diameters) with identity columns
        n_extra_edges = min(len(n_hubs_edges), max_n_edges_relaxation)
        if n_extra_edges > 0:
            replace_start = ncol_A - n_extra_edges
            one_ridx_A[3 * replace_start :] = np.repeat(n_hubs_edges[:n_extra_edges], 3)
            solver_column_ids_relax = list(solver_column_ids)
            solver_columns_are_triangles_relax = list(solver_columns_are_triangles)
            for offset, row in enumerate(n_hubs_edges[:n_extra_edges]):
                edge_id = int(row_simplex_ids[row])
                column = replace_start + offset
                solver_column_ids_relax[column] = edge_id
                solver_columns_are_triangles_relax[column] = False
            states_relax, solutions_relax = solve_multiple_gf2_m4ri(
                one_ridx_A=one_ridx_A.tolist(),
                one_cidx_A=one_cidx_A.tolist(),
                nrow_A=nrow_A,
                ncol_A=ncol_A,
                one_idx_b_list=one_idx_b_list,
            )
            solved_relax = [i for i, s in enumerate(states_relax) if s == 0]
            if len(solved_relax) > 0:
                result.loop_pairs_matched_relax = [pairs_kept[i] for i in solved_relax]
                result.mapping_deformation_matched_relax = [
                    _decode_deformation(
                        solutions_relax[i],
                        solver_column_ids_relax,
                        solver_columns_are_triangles_relax,
                    )
                    for i in solved_relax
                ]

    return result


def compute_loop_geometric_distance(
    source_coords_list: list[list[list[float]]] | list[np.ndarray],
    target_coords_list: list[list[list[float]]] | list[np.ndarray],
    method: LoopDistMethod = DEFAULT_LOOP_DIST_METHOD,
    n_workers: Count_t = DEFAULT_N_MAX_WORKERS,
) -> np.ndarray:
    if len(source_coords_list) == 0 or len(target_coords_list) == 0:
        return np.array([np.nan])

    match method:
        case "frechet":
            try:
                distances_arr = compute_pairwise_loop_frechet(
                    source_coords_list, target_coords_list, n_workers=n_workers
                )
                return distances_arr
            except Exception:
                return np.full(
                    len(source_coords_list) * len(target_coords_list), np.nan
                )
        case "hausdorff":
            distances = []
            for source_coords in source_coords_list:
                for target_coords in target_coords_list:
                    try:
                        dist = max(
                            directed_hausdorff(source_coords, target_coords)[0],
                            directed_hausdorff(target_coords, source_coords)[0],
                        )
                        distances.append(dist)
                    except (ValueError, IndexError):
                        distances.append(np.nan)
            return np.array(distances)
        case _:
            return np.full(len(source_coords_list) * len(target_coords_list), np.nan)
