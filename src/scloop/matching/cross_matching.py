# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import numpy as np

from ..computing.homology import (
    _persistence_pairs_from_ripser_result,
    compute_cross_dataset_image_homology,
)
from ..data.base_components import LoopClass, PersistencePair
from ..data.types import LOOP_ATTRIBUTE_MODE, Diameter_t


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


def _image_class_vertices(image_result, pair_idx: int, offset: int) -> set[int]:
    vertices: set[int] = set()
    for verts, _coeff in image_result.cocycles_by_dim[1][pair_idx]:
        for v in verts:
            vertices.add(int(v) - offset)
    return vertices


def _attribute_to_loop_class(
    image_pair: PersistencePair,
    vertices: set[int],
    offset: int,
    loop_classes: list[LoopClass | None],
    mode: LOOP_ATTRIBUTE_MODE,
    vertex_ids: list[int] | None = None,
) -> int | None:
    match mode:
        case "exact":
            birth_simplex = tuple(sorted(v - offset for v in image_pair.birth_simplex))
            for idx, loop_class in enumerate(loop_classes):
                if loop_class is None:
                    continue
                if tuple(sorted(loop_class.birth_simplex)) == birth_simplex:
                    return idx
            return None
        case "boundary":
            return
        case "vertex":
            if vertex_ids is not None:
                vertices = {vertex_ids[v] for v in vertices}
            best_idx = None
            best_overlap = 0
            for idx, loop_class in enumerate(loop_classes):
                if loop_class is None or loop_class.representatives is None:
                    continue
                rep_vertices = {
                    int(v) for rep in loop_class.representatives for v in rep
                }
                overlap = len(vertices & rep_vertices)
                if overlap > best_overlap:
                    best_overlap = overlap
                    best_idx = idx
            return best_idx


def cross_match_loops_image(
    source_embedding: np.ndarray,
    target_embedding: np.ndarray,
    source_loop_classes: list[LoopClass | None],
    target_loop_classes: list[LoopClass | None],
    thresh: Diameter_t,
    source_loop_attribute_mode: LOOP_ATTRIBUTE_MODE = "exact",
    target_loop_attribute_mode: LOOP_ATTRIBUTE_MODE = "exact",
    source_vertex_ids: list[int] | None = None,
    target_vertex_ids: list[int] | None = None,
    **nei_kwargs,
) -> list[tuple[int, int]]:
    homology = compute_cross_dataset_image_homology(
        source_embedding=source_embedding,
        target_embedding=target_embedding,
        thresh=thresh,
        **nei_kwargs,
    )
    source_pairs = _persistence_pairs_from_ripser_result(
        homology.source_image_result.image
    )
    target_pairs = _persistence_pairs_from_ripser_result(
        homology.target_image_result.image
    )

    matches: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for i, j in _match_image_pairs_by_death_simplex(source_pairs, target_pairs):
        source_idx = _attribute_to_loop_class(
            source_pairs[i],
            _image_class_vertices(homology.source_image_result.image, i, 0),
            0,
            source_loop_classes,
            source_loop_attribute_mode,
            source_vertex_ids,
        )
        target_idx = _attribute_to_loop_class(
            target_pairs[j],
            _image_class_vertices(
                homology.target_image_result.image, j, homology.n_source_vertices
            ),
            homology.n_source_vertices,
            target_loop_classes,
            target_loop_attribute_mode,
            target_vertex_ids,
        )
        if source_idx is None or target_idx is None:
            continue
        if (source_idx, target_idx) in seen:
            continue
        seen.add((source_idx, target_idx))
        matches.append((source_idx, target_idx))
    return matches
