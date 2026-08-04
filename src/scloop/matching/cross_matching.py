# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import csr_matrix

from ..computing.homology import (
    _persistence_pairs_from_ripser_result,
    compute_cross_dataset_image_homology,
)
from ..computing.loops import compute_loop_representatives
from ..computing.matching import check_homological_equivalence, cocycle_to_edge_mask
from ..data.base_components import LoopClass, LoopClassEquivalence, PersistencePair
from ..data.boundary import BoundaryMatrixD1
from ..data.constants import (
    DEFAULT_COLUMN_TRIM_METHOD,
    DEFAULT_EXTRA_DIAM_EQUIVALENCE,
    DEFAULT_N_NEIGHBORS_COLUMN_TRIM,
    DEFAULT_WITH_RELAXATION_EQUIVALENCE,
)
from ..data.types import LOOP_ATTRIBUTE_MODE, Diameter_t


@dataclass
class CrossMatchSide:
    embedding: np.ndarray
    loop_classes: list[LoopClass | None]
    vertex_ids: list[int] = field(default_factory=list)
    boundary_matrix_d1: BoundaryMatrixD1 | None = None
    loop_attribute_mode: LOOP_ATTRIBUTE_MODE = "exact"


def _match_image_pairs_by_death_simplex(
    source_pairs: list[PersistencePair],
    target_pairs: list[PersistencePair],
) -> list[tuple[int, int]]:
    targets_by_death: dict[tuple[int, ...], list[int]] = {}
    for j, pair in enumerate(target_pairs):
        if not pair.death_simplex:
            continue
        targets_by_death.setdefault(tuple(sorted(pair.death_simplex)), []).append(j)

    matches: list[tuple[int, int]] = []
    for i, pair in enumerate(source_pairs):
        if not pair.death_simplex:
            continue
        for j in targets_by_death.get(tuple(sorted(pair.death_simplex)), []):
            matches.append((i, j))
    return matches


def _reconstruct_image_loop_classes(
    side: CrossMatchSide,
    pair_indices: set[int],
    diagram: tuple,
    pair_simplices: list,
    cocycles: list,
    distance_matrix: csr_matrix,
    kwargs_reconstruct: dict,
) -> dict[int, LoopClass]:
    if side.loop_attribute_mode == "exact":
        return {}
    if side.boundary_matrix_d1 is None:
        raise ValueError(
            f"loop attribute mode '{side.loop_attribute_mode}' needs boundary_matrix_d1"
        )

    selected = sorted(idx for idx in pair_indices if idx < len(cocycles))
    if not selected:
        return {}

    births, deaths = diagram
    birth_simplices, death_simplices = pair_simplices
    image_loop_classes = compute_loop_representatives(
        embedding=side.embedding,
        pairwise_distance_matrix=distance_matrix,
        persistence_diagram=(
            [births[idx] for idx in selected],
            [deaths[idx] for idx in selected],
        ),
        cocycles=[cocycles[idx] for idx in selected],
        boundary_matrix_d1=side.boundary_matrix_d1,
        vertex_ids=side.vertex_ids,
        persistence_pair_simplices=(
            [birth_simplices[idx] for idx in selected],
            [death_simplices[idx] for idx in selected],
        ),
        **kwargs_reconstruct,
    )
    return {
        selected[image_loop_class.persistence_index]: image_loop_class
        for image_loop_class in image_loop_classes
        if image_loop_class is not None
        and image_loop_class.persistence_index is not None
    }


def _attribute_to_loop_class(
    side: CrossMatchSide,
    image_pair: PersistencePair,
    image_loop_class: LoopClass | None,
    offset: int,
    extra_diameter: float,
    with_relaxation: bool,
    kwargs_equivalence: dict,
) -> tuple[int | None, LoopClassEquivalence | None, LoopClass | None]:
    """Return the attributed loop-class index and, for the boundary mode, the
    equivalence certificate that justified it."""
    match side.loop_attribute_mode:
        case "exact":
            birth_simplex = tuple(sorted(v - offset for v in image_pair.birth_simplex))
            for idx, loop_class in enumerate(side.loop_classes):
                if loop_class is None:
                    continue
                if tuple(sorted(loop_class.birth_simplex)) == birth_simplex:
                    return idx, None, None
            return None, None, None
        case "boundary":
            if image_loop_class is None or image_loop_class.representatives is None:
                return None, None, None
            assert side.boundary_matrix_d1 is not None
            column_scores = None
            if (
                kwargs_equivalence.get("column_trim_method", DEFAULT_COLUMN_TRIM_METHOD)
                == "loop_proximity"
            ):
                column_scores = image_loop_class.column_proximity_scores(
                    side.boundary_matrix_d1,
                    side.embedding,
                    kwargs_equivalence.get(
                        "n_neighbors_column_trim", DEFAULT_N_NEIGHBORS_COLUMN_TRIM
                    ),
                )
            cocycle_edge_mask = None
            if image_loop_class.cocycles is not None:
                cocycle_edge_mask = cocycle_to_edge_mask(
                    cocycle=image_loop_class.cocycles,
                    boundary_matrix_d1=side.boundary_matrix_d1,
                    vertex_ids=side.vertex_ids,
                )
            for idx, loop_class in enumerate(side.loop_classes):
                if loop_class is None or loop_class.representatives is None:
                    continue
                max_lifetime = max(image_loop_class.lifetime, loop_class.lifetime)
                equivalence = check_homological_equivalence(
                    source_loops=image_loop_class.representatives,
                    target_loops=loop_class.representatives,
                    boundary_matrix_d1=side.boundary_matrix_d1,
                    with_relaxation=with_relaxation,
                    max_column_diameter=max(image_loop_class.death, loop_class.death)
                    + extra_diameter * max_lifetime,
                    cocycle_edge_mask=cocycle_edge_mask,
                    column_scores=column_scores,
                    **kwargs_equivalence,
                )
                if equivalence.is_equivalent(relax=with_relaxation):
                    return idx, equivalence, image_loop_class
            return None, None, None
        case "vertex":
            if image_loop_class is None or image_loop_class.representatives is None:
                return None, None, None
            vertices = {v for rep in image_loop_class.representatives for v in rep}
            best_idx = None
            best_overlap = 0
            for idx, loop_class in enumerate(side.loop_classes):
                if loop_class is None or loop_class.representatives is None:
                    continue
                rep_vertices = {v for rep in loop_class.representatives for v in rep}
                overlap = len(vertices & rep_vertices)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            return best_idx, None, None


def cross_match_loops_image(
    source: CrossMatchSide,
    target: CrossMatchSide,
    thresh: Diameter_t,
    kwargs_reconstruct: dict | None = None,
    kwargs_equivalence: dict | None = None,
    **nei_kwargs,
) -> list[tuple[int, int, LoopClassEquivalence | None, LoopClass | None]]:
    kwargs_reconstruct = dict(kwargs_reconstruct or {})
    kwargs_equivalence = dict(kwargs_equivalence or {})
    extra_diameter = kwargs_equivalence.pop(
        "extra_diameter_homology_equivalence", DEFAULT_EXTRA_DIAM_EQUIVALENCE
    )
    with_relaxation = kwargs_equivalence.pop(
        "with_relaxation", DEFAULT_WITH_RELAXATION_EQUIVALENCE
    )

    homology = compute_cross_dataset_image_homology(
        source_embedding=source.embedding[source.vertex_ids],
        target_embedding=target.embedding[target.vertex_ids],
        thresh=thresh,
        **nei_kwargs,
    )
    source_image = homology.source_image_result.image
    target_image = homology.target_image_result.image
    source_pairs = _persistence_pairs_from_ripser_result(source_image)
    target_pairs = _persistence_pairs_from_ripser_result(target_image)

    candidates = _match_image_pairs_by_death_simplex(source_pairs, target_pairs)
    if not candidates:
        return []

    source_image_classes = _reconstruct_image_loop_classes(
        side=source,
        pair_indices={i for i, _ in candidates},
        diagram=source_image.births_and_deaths_by_dim[1],
        pair_simplices=homology.source_image_pair_simplices[1],
        cocycles=homology.source_image_cocycles[1],
        distance_matrix=homology.source_distance_matrix,
        kwargs_reconstruct=kwargs_reconstruct,
    )
    target_image_classes = _reconstruct_image_loop_classes(
        side=target,
        pair_indices={j for _, j in candidates},
        diagram=target_image.births_and_deaths_by_dim[1],
        pair_simplices=homology.target_image_pair_simplices[1],
        cocycles=homology.target_image_cocycles[1],
        distance_matrix=homology.target_distance_matrix,
        kwargs_reconstruct=kwargs_reconstruct,
    )

    source_attribution: dict[int, tuple] = {}
    target_attribution: dict[int, tuple] = {}
    matches: list[tuple[int, int, LoopClassEquivalence | None, LoopClass | None]] = []
    seen: set[tuple[int, int]] = set()
    for i, j in candidates:
        if i not in source_attribution:
            source_attribution[i] = _attribute_to_loop_class(
                side=source,
                image_pair=source_pairs[i],
                image_loop_class=source_image_classes.get(i),
                offset=0,
                extra_diameter=extra_diameter,
                with_relaxation=with_relaxation,
                kwargs_equivalence=kwargs_equivalence,
            )
        if j not in target_attribution:
            target_attribution[j] = _attribute_to_loop_class(
                side=target,
                image_pair=target_pairs[j],
                image_loop_class=target_image_classes.get(j),
                offset=homology.n_source_vertices,
                extra_diameter=extra_diameter,
                with_relaxation=with_relaxation,
                kwargs_equivalence=kwargs_equivalence,
            )
        source_idx, source_equivalence, source_image_lc = source_attribution[i]
        target_idx, target_equivalence, target_image_lc = target_attribution[j]
        if source_idx is None or target_idx is None:
            continue
        if (source_idx, target_idx) in seen:
            continue
        seen.add((source_idx, target_idx))
        matches.append(
            (
                source_idx,
                target_idx,
                target_equivalence or source_equivalence,
                target_image_lc or source_image_lc,
            )
        )
    return matches
