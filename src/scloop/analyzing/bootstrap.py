# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from concurrent.futures import (
    ThreadPoolExecutor,
    as_completed,
)
from typing import Literal

import numpy as np
from anndata import AnnData
from loguru import logger
from pydantic import BaseModel, ConfigDict, Field
from rich.progress import Progress
from scipy.sparse import csr_matrix

from ..computing.homology import (
    _persistence_pairs_from_ripser_result,
    compute_image_bootstrap_homology,
    compute_persistence_diagram_and_cocycles,
    compute_sparse_pairwise_distance,
)
from ..computing.loops import (
    compute_loop_representatives,
    remap_cocycles_for_full_reconstruction,
)
from ..computing.matching import (
    check_homological_equivalence,
    cocycle_to_edge_mask,
    compute_geometric_distance,
)
from ..data.analysis_containers import LoopMatch
from ..data.base_components import (
    ImagePairRecord,
    LoopClass,
    PersistencePair,
)
from ..data.boundary import BoundaryMatrixD1
from ..data.constants import (
    DEFAULT_COLUMN_TRIM_METHOD,
    DEFAULT_EXTRA_DIAM_EQUIVALENCE,
    DEFAULT_FOREIGN_CHORD_MULT,
    DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
    DEFAULT_K_YEN,
    DEFAULT_LIFE_PCT,
    DEFAULT_LOOP_DIST_METHOD,
    DEFAULT_MAX_N_EDGES_RELAXATION_EQUIVALENCE,
    DEFAULT_MAX_PERIMETER_MULT,
    DEFAULT_N_COCYCLES_USED,
    DEFAULT_N_FORCE_DEVIATE,
    DEFAULT_N_HUBS_RELAXATION_EQUIVALENCE,
    DEFAULT_N_MAX_WORKERS,
    DEFAULT_N_NEIGHBORS_COLUMN_TRIM,
    DEFAULT_N_PAIRS_CHECK_EQUIVALENCE,
    DEFAULT_N_REPS_PER_LOOP,
    DEFAULT_NOISE_SCALE,
    DEFAULT_WITH_RELAXATION_EQUIVALENCE,
)
from ..data.metadata import ScloopMeta
from ..data.types import ColumnTrimMethod, Count_t, LoopDistMethod, PositiveFloat
from ..data.utils import nearest_neighbor_per_row


class BootstrapResult(BaseModel):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    idx_bootstrap: int
    persistence_diagram: list | None
    persistence_pair_simplices: list | None
    cocycles: list | None
    indices_resample: list[int]
    loop_classes: list[LoopClass | None]
    matches: dict[int, list[LoopMatch]]
    reference_image_pairs: list[ImagePairRecord] = Field(default_factory=list)
    bootstrap_image_pairs: list[ImagePairRecord] = Field(default_factory=list)


class _MatchCandidate(BaseModel):
    target_class_idx: int
    geometric_distance: float | None = None
    neighbor_rank: int | None = None
    image_death_simplex: list[int] | None = None


def _select_geometric_candidates(
    pairwise_distances: np.ndarray,
    k: int,
    require_homological_equivalence: bool,
) -> dict[int, list[_MatchCandidate]]:
    if k <= 0:
        raise ValueError("k must be positive")
    n_candidates = k if require_homological_equivalence else 1
    neighbor_indices, neighbor_distances = nearest_neighbor_per_row(
        pairwise_distances, n_candidates
    )
    candidates: dict[int, list[_MatchCandidate]] = {}
    for source_idx in range(pairwise_distances.shape[0]):
        for rank in range(n_candidates):
            target_idx = int(neighbor_indices[source_idx, rank])
            if target_idx < 0:
                continue
            candidates.setdefault(source_idx, []).append(
                _MatchCandidate(
                    target_class_idx=target_idx,
                    geometric_distance=float(neighbor_distances[source_idx, rank]),
                    neighbor_rank=rank,
                )
            )
    return candidates


def _select_image_candidates(
    reference_records: list[ImagePairRecord],
    bootstrap_records: list[ImagePairRecord],
) -> dict[int, list[_MatchCandidate]]:
    targets_by_death_simplex: dict[tuple[int, ...], list[int]] = {}
    for record in bootstrap_records:
        if record.image_pair is None or not record.image_pair.death_simplex:
            continue
        death_simplex = tuple(sorted(record.image_pair.death_simplex))
        targets_by_death_simplex.setdefault(death_simplex, []).append(
            record.source_class_idx
        )

    candidates: dict[int, list[_MatchCandidate]] = {}
    for record in reference_records:
        if record.image_pair is None or not record.image_pair.death_simplex:
            continue
        death_simplex = tuple(sorted(record.image_pair.death_simplex))
        for target_idx in sorted(targets_by_death_simplex.get(death_simplex, [])):
            candidates.setdefault(record.source_class_idx, []).append(
                _MatchCandidate(
                    target_class_idx=target_idx,
                    image_death_simplex=list(death_simplex),
                )
            )
    return candidates


def _build_image_pair_records(
    loop_classes: list[LoopClass | None],
    image_pairs: list[PersistencePair],
    vertex_offset: int,
) -> list[ImagePairRecord]:
    image_pairs_by_birth = {
        tuple(sorted(pair.birth_simplex)): pair
        for pair in image_pairs
        if pair.birth_simplex
    }
    records: list[ImagePairRecord] = []
    for class_idx, loop_class in enumerate(loop_classes):
        if loop_class is None:
            continue
        source_pair = PersistencePair(
            birth=loop_class.birth,
            death=loop_class.death,
            birth_simplex=[v + vertex_offset for v in loop_class.birth_simplex],
            death_simplex=[v + vertex_offset for v in loop_class.death_simplex],
        )
        image_pair = image_pairs_by_birth.get(tuple(sorted(source_pair.birth_simplex)))
        records.append(
            ImagePairRecord(
                source_class_idx=class_idx,
                source_pair=source_pair,
                image_pair=image_pair,
            )
        )
    return records


def run_single_bootstrap(
    idx_bootstrap: int,
    adata: AnnData,
    meta: ScloopMeta,
    original_loop_classes: list[LoopClass | None],
    original_boundary_matrix_d1: BoundaryMatrixD1,
    thresh: float | None = None,
    noise_scale: float = DEFAULT_NOISE_SCALE,
    top_k: int = 1,
    n_reps_per_loop: int = DEFAULT_N_REPS_PER_LOOP,
    life_pct: float = DEFAULT_LIFE_PCT,
    n_cocycles_used: int = DEFAULT_N_COCYCLES_USED,
    n_force_deviate: int = DEFAULT_N_FORCE_DEVIATE,
    k_yen: int = DEFAULT_K_YEN,
    loop_lower_t_pct: float = 2.5,
    loop_upper_t_pct: float = 97.5,
    do_random_walk: bool = False,
    n_random_graphs: Count_t = 10,
    decay_random_walk: PositiveFloat = 1.0,
    noise_random_walk: PositiveFloat = 1.0,
    seed_random_walk: int = 1,
    do_force_deviate_random_walk: bool = False,
    foreign_chord_mult: float = DEFAULT_FOREIGN_CHORD_MULT,
    max_perimeter_mult: float = DEFAULT_MAX_PERIMETER_MULT,
    k_neighbors_check_equivalence: int = DEFAULT_K_NEIGHBORS_CHECK_EQUIVALENCE,
    method_geometric_equivalence: LoopDistMethod = DEFAULT_LOOP_DIST_METHOD,
    n_pairs_check_equivalence: int = DEFAULT_N_PAIRS_CHECK_EQUIVALENCE,
    with_relaxation_equivalence: bool = DEFAULT_WITH_RELAXATION_EQUIVALENCE,
    n_hubs_relaxation_equivalence: int = DEFAULT_N_HUBS_RELAXATION_EQUIVALENCE,
    max_n_edges_relaxation_equivalence: int = DEFAULT_MAX_N_EDGES_RELAXATION_EQUIVALENCE,
    extra_diameter_homology_equivalence: float = DEFAULT_EXTRA_DIAM_EQUIVALENCE,
    filter_column_homology_equivalence: bool = True,
    column_trim_method: ColumnTrimMethod = DEFAULT_COLUMN_TRIM_METHOD,
    n_neighbors_column_trim: int = DEFAULT_N_NEIGHBORS_COLUMN_TRIM,
    full_pairwise_distance_matrix: csr_matrix | None = None,
    full_vertex_ids: list[int] | None = None,
    reconstruct_on_full_data: bool = False,
    candidate_method: Literal["geometric", "image"] = "geometric",
    require_homological_equivalence: bool = True,
    **kwargs,
) -> BootstrapResult:
    if candidate_method not in {"geometric", "image"}:
        raise ValueError(f"unknown candidate method: {candidate_method}")
    reference_image_pairs: list[ImagePairRecord] = []
    bootstrap_image_pairs: list[ImagePairRecord] = []
    image_homology_result = None
    if candidate_method == "image":
        if thresh is None:
            raise ValueError("image matching requires a finite homology threshold")
        image_homology_result = compute_image_bootstrap_homology(
            adata=adata,
            meta=meta,
            thresh=thresh,
            noise_scale=noise_scale,
            **kwargs,
        )
        (
            persistence_diagram,
            persistence_pair_simplices,
            cocycles,
            indices_resample,
            sparse_pairwise_distance_matrix,
        ) = (
            image_homology_result.persistence_diagram,
            image_homology_result.persistence_pair_simplices,
            image_homology_result.cocycles,
            image_homology_result.indices_resample,
            image_homology_result.bootstrap_distance_matrix,
        )
    else:
        (
            persistence_diagram,
            persistence_pair_simplices,
            cocycles,
            indices_resample,
            sparse_pairwise_distance_matrix,
        ) = compute_persistence_diagram_and_cocycles(
            adata=adata,
            meta=meta,
            thresh=thresh,
            bootstrap=True,
            noise_scale=noise_scale,
            **kwargs,
        )

    assert meta.preprocess is not None
    assert meta.preprocess.embedding_method is not None
    assert indices_resample is not None

    embedding = np.array(adata.obsm[f"X_{meta.preprocess.embedding_method}"])

    if (
        reconstruct_on_full_data
        and full_pairwise_distance_matrix is not None
        and full_vertex_ids is not None
    ):
        remapped_cocycles = remap_cocycles_for_full_reconstruction(
            cocycles=cocycles[1],
            bootstrap_vertex_ids=indices_resample,
            full_vertex_ids=full_vertex_ids,
        )
        bootstrap_loop_classes = compute_loop_representatives(
            embedding=embedding,
            pairwise_distance_matrix=full_pairwise_distance_matrix,
            persistence_diagram=persistence_diagram[1],
            persistence_pair_simplices=persistence_pair_simplices[1],
            cocycles=remapped_cocycles,
            boundary_matrix_d1=original_boundary_matrix_d1,
            vertex_ids=full_vertex_ids,
            top_k=top_k,
            n_reps_per_loop=n_reps_per_loop,
            life_pct=life_pct,
            n_cocycles_used=n_cocycles_used,
            n_force_deviate=n_force_deviate,
            k_yen=k_yen,
            loop_lower_t_pct=loop_lower_t_pct,
            loop_upper_t_pct=loop_upper_t_pct,
            do_random_walk=do_random_walk,
            n_random_graphs=n_random_graphs,
            decay_random_walk=decay_random_walk,
            noise_random_walk=noise_random_walk,
            seed_random_walk=seed_random_walk,
            do_force_deviate_random_walk=do_force_deviate_random_walk,
            bootstrap=False,
            do_clean_cocycle_region=True,
            foreign_chord_mult=foreign_chord_mult,
            max_perimeter_mult=max_perimeter_mult,
        )
    else:
        bootstrap_loop_classes = compute_loop_representatives(
            embedding=embedding,
            pairwise_distance_matrix=sparse_pairwise_distance_matrix,
            persistence_diagram=persistence_diagram[1],
            persistence_pair_simplices=persistence_pair_simplices[1],
            cocycles=cocycles[1],
            boundary_matrix_d1=original_boundary_matrix_d1,
            vertex_ids=indices_resample,
            top_k=top_k,
            n_reps_per_loop=n_reps_per_loop,
            life_pct=life_pct,
            n_cocycles_used=n_cocycles_used,
            n_force_deviate=n_force_deviate,
            k_yen=k_yen,
            loop_lower_t_pct=loop_lower_t_pct,
            loop_upper_t_pct=loop_upper_t_pct,
            do_random_walk=do_random_walk,
            n_random_graphs=n_random_graphs,
            decay_random_walk=decay_random_walk,
            noise_random_walk=noise_random_walk,
            seed_random_walk=seed_random_walk,
            do_force_deviate_random_walk=do_force_deviate_random_walk,
            bootstrap=True,
            foreign_chord_mult=foreign_chord_mult,
            max_perimeter_mult=max_perimeter_mult,
        )

    if image_homology_result is not None:
        reference_image_pairs = _build_image_pair_records(
            original_loop_classes,
            _persistence_pairs_from_ripser_result(
                image_homology_result.reference_image_result.image
            ),
            vertex_offset=0,
        )
        bootstrap_image_pairs = _build_image_pair_records(
            bootstrap_loop_classes,
            _persistence_pairs_from_ripser_result(
                image_homology_result.bootstrap_image_result.image
            ),
            vertex_offset=image_homology_result.n_reference_vertices,
        )

    n_original = len(original_loop_classes)
    n_bootstrap = len(bootstrap_loop_classes)

    matches: dict[int, list[LoopMatch]] = {}
    if n_original == 0 or n_bootstrap == 0:
        return BootstrapResult(
            idx_bootstrap=idx_bootstrap,
            persistence_diagram=persistence_diagram,
            persistence_pair_simplices=persistence_pair_simplices,
            cocycles=cocycles,
            indices_resample=indices_resample,
            loop_classes=bootstrap_loop_classes,
            matches=matches,
            reference_image_pairs=reference_image_pairs,
            bootstrap_image_pairs=bootstrap_image_pairs,
        )

    if candidate_method == "geometric":
        pairwise_geo_dist = np.full((n_original, n_bootstrap), np.nan)
        for i, src_loop in enumerate(original_loop_classes):
            if src_loop is None:
                continue
            src_coords = src_loop.coordinates_vertices_representatives
            if src_coords is None:
                continue
            for j, tgt_loop in enumerate(bootstrap_loop_classes):
                if tgt_loop is None:
                    continue
                tgt_coords = tgt_loop.coordinates_vertices_representatives
                if tgt_coords is None:
                    continue
                pairwise_geo_dist[i, j] = compute_geometric_distance(
                    source_coords_list=src_coords,
                    target_coords_list=tgt_coords,
                    method=method_geometric_equivalence,
                    n_workers=1,
                )
        candidates = _select_geometric_candidates(
            pairwise_geo_dist,
            k=k_neighbors_check_equivalence,
            require_homological_equivalence=require_homological_equivalence,
        )
    else:
        candidates = _select_image_candidates(
            reference_image_pairs, bootstrap_image_pairs
        )

    cocycle_edge_masks: list[np.ndarray | None] = [None] * n_original
    if require_homological_equivalence:
        assert meta.preprocess is not None
        assert meta.preprocess.num_vertices is not None
        original_vertex_ids = (
            meta.preprocess.indices_downsample
            if meta.preprocess.indices_downsample is not None
            else list(range(meta.preprocess.num_vertices))
        )
        for i, loop_class in enumerate(original_loop_classes):
            if loop_class is None or loop_class.cocycles is None:
                continue
            cocycle_edge_masks[i] = cocycle_to_edge_mask(
                cocycle=loop_class.cocycles,
                boundary_matrix_d1=original_boundary_matrix_d1,
                vertex_ids=original_vertex_ids,
            )

    for source_idx, source_candidates in candidates.items():
        source_loop = original_loop_classes[source_idx]
        if source_loop is None:
            continue
        for candidate in source_candidates:
            target_loop = bootstrap_loop_classes[candidate.target_class_idx]
            if target_loop is None:
                continue

            match = LoopMatch(
                idx_bootstrap=idx_bootstrap,
                target_class_idx=candidate.target_class_idx,
                candidate_method=candidate_method,
                geometric_distance=candidate.geometric_distance,
                neighbor_rank=candidate.neighbor_rank,
                image_death_simplex=candidate.image_death_simplex,
            )
            matches.setdefault(source_idx, []).append(match)

            if not require_homological_equivalence:
                continue
            if (
                source_loop.representatives is None
                or target_loop.representatives is None
            ):
                continue
            max_column_diameter = None
            if filter_column_homology_equivalence:
                if extra_diameter_homology_equivalence < 0:
                    raise ValueError(
                        "extra_diameter_homology_equivalence must be nonnegative"
                    )
                max_lifetime = max(source_loop.lifetime, target_loop.lifetime)
                max_column_diameter = (
                    max(source_loop.death, target_loop.death)
                    + float(extra_diameter_homology_equivalence) * max_lifetime
                )
            match.topological_equivalence = check_homological_equivalence(
                source_loops=source_loop.representatives,
                target_loops=target_loop.representatives,
                boundary_matrix_d1=original_boundary_matrix_d1,
                n_pairs_check=n_pairs_check_equivalence,
                with_relaxation=with_relaxation_equivalence,
                n_hubs_relaxation=n_hubs_relaxation_equivalence,
                max_n_edges_relaxation=max_n_edges_relaxation_equivalence,
                max_column_diameter=max_column_diameter,
                cocycle_edge_mask=cocycle_edge_masks[source_idx],
                column_trim_method=column_trim_method,
                column_scores=(
                    source_loop.column_proximity_scores(
                        original_boundary_matrix_d1,
                        embedding,
                        n_neighbors_column_trim,
                    )
                    if column_trim_method == "loop_proximity"
                    else None
                ),
            )
            match.boundary_checked = True

    return BootstrapResult(
        idx_bootstrap=idx_bootstrap,
        persistence_diagram=persistence_diagram,
        persistence_pair_simplices=persistence_pair_simplices,
        cocycles=cocycles,
        indices_resample=indices_resample,
        loop_classes=bootstrap_loop_classes,
        matches=matches,
        reference_image_pairs=reference_image_pairs,
        bootstrap_image_pairs=bootstrap_image_pairs,
    )


def run_bootstrap_pipeline(
    n_bootstrap: int,
    adata: AnnData,
    meta: ScloopMeta,
    original_loop_classes: list[LoopClass | None],
    original_boundary_matrix_d1: BoundaryMatrixD1,
    n_max_workers: int = DEFAULT_N_MAX_WORKERS,
    verbose: bool = False,
    progress: Progress | None = None,
    reconstruct_on_full_data: bool = False,
    thresh: float | None = None,
    with_relaxation_equivalence: bool = DEFAULT_WITH_RELAXATION_EQUIVALENCE,
    n_hubs_relaxation_equivalence: int = DEFAULT_N_HUBS_RELAXATION_EQUIVALENCE,
    max_n_edges_relaxation_equivalence: int = DEFAULT_MAX_N_EDGES_RELAXATION_EQUIVALENCE,
    do_force_deviate_random_walk: bool = False,
    **kwargs,
) -> list[BootstrapResult]:
    results: list[BootstrapResult] = []

    full_pairwise_distance_matrix: csr_matrix | None = None
    full_vertex_ids: list[int] | None = None
    if reconstruct_on_full_data:
        full_pairwise_distance_matrix, full_vertex_ids = (
            compute_sparse_pairwise_distance(
                adata=adata,
                meta=meta,
                bootstrap=False,
                thresh=thresh,
            )
        )

    if kwargs.get("column_trim_method", DEFAULT_COLUMN_TRIM_METHOD) == "loop_proximity":
        assert meta.preprocess is not None
        assert meta.preprocess.embedding_method is not None
        warm_embedding = np.array(
            adata.obsm[f"X_{meta.preprocess.embedding_method}"]
        )
        for loop_class in original_loop_classes:
            if loop_class is not None and loop_class.representatives is not None:
                loop_class.column_proximity_scores(
                    original_boundary_matrix_d1,
                    warm_embedding,
                    kwargs.get(
                        "n_neighbors_column_trim", DEFAULT_N_NEIGHBORS_COLUMN_TRIM
                    ),
                )

    ExecutorClass = ThreadPoolExecutor

    with ExecutorClass(max_workers=n_max_workers) as executor:
        tasks = {}
        for i in range(n_bootstrap):
            task = executor.submit(
                run_single_bootstrap,
                idx_bootstrap=i,
                adata=adata,
                meta=meta,
                original_loop_classes=original_loop_classes,
                original_boundary_matrix_d1=original_boundary_matrix_d1,
                full_pairwise_distance_matrix=full_pairwise_distance_matrix,
                full_vertex_ids=full_vertex_ids,
                reconstruct_on_full_data=reconstruct_on_full_data,
                thresh=thresh,
                with_relaxation_equivalence=with_relaxation_equivalence,
                n_hubs_relaxation_equivalence=n_hubs_relaxation_equivalence,
                max_n_edges_relaxation_equivalence=max_n_edges_relaxation_equivalence,
                do_force_deviate_random_walk=do_force_deviate_random_walk,
                **kwargs,
            )
            tasks[task] = i

        task_id = None
        if progress:
            task_id = progress.add_task("Working...", total=n_bootstrap)

        for task in as_completed(tasks):
            i = tasks[task]
            try:
                res = task.result()
                results.append(res)
                if verbose:
                    logger.success(f"[Bootstrap {i + 1}/{n_bootstrap}] finished")
            except Exception as e:
                logger.opt(exception=True).warning(
                    f"[Bootstrap {i + 1}/{n_bootstrap}] failed: {e}"
                )

            if progress and task_id is not None:
                progress.advance(task_id)

    results.sort(key=lambda x: x.idx_bootstrap)
    return results
