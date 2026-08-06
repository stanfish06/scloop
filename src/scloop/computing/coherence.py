# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from ..data.types import HomotopyCoherenceMethod
from ..data.utils import edge_idx_decode
from ..utils.distance_metrics import compute_loop_frechet


def global_h1_death_scale(persistence_diagram: list | None) -> float | None:
    if not persistence_diagram or len(persistence_diagram) < 2:
        return None
    deaths = np.asarray(persistence_diagram[1][1], dtype=float)
    finite = deaths[np.isfinite(deaths)]
    return float(finite.max()) if finite.size else None


def _flip_triangle(
    cycle: frozenset[int], triangle: frozenset[int], num_vertices: int
) -> frozenset[int] | None:
    """Apply one elementary triangle deformation to a simple cycle."""

    shared = cycle & triangle
    if len(shared) == 2:
        return cycle ^ triangle
    if len(shared) != 1:
        return None

    shared_vertices = set(edge_idx_decode(next(iter(shared)), num_vertices))
    triangle_vertices = {
        vertex for edge in triangle for vertex in edge_idx_decode(edge, num_vertices)
    }
    new_vertices = triangle_vertices - shared_vertices
    if len(new_vertices) != 1:
        return None
    cycle_vertices = {
        vertex for edge in cycle for vertex in edge_idx_decode(edge, num_vertices)
    }
    if next(iter(new_vertices)) in cycle_vertices:
        return None
    return cycle ^ triangle


def _cycle_vertices(cycle: frozenset[int], num_vertices: int) -> list[int]:
    adjacency: dict[int, list[int]] = {}
    for edge in cycle:
        tail, head = edge_idx_decode(edge, num_vertices)
        adjacency.setdefault(tail, []).append(head)
        adjacency.setdefault(head, []).append(tail)
    start = next(iter(adjacency))
    order = [start]
    previous, current = -1, start
    while len(order) <= len(adjacency):
        neighbours = adjacency[current]
        if len(neighbours) != 2:
            break
        first, second = neighbours
        following = first if first != previous else second
        if following == start:
            break
        order.append(following)
        previous, current = current, following
    return order


def _cycle_coords(
    cycle: frozenset[int], num_vertices: int, embedding: np.ndarray
) -> np.ndarray:
    vertices = _cycle_vertices(cycle, num_vertices)
    return np.ascontiguousarray(embedding[vertices], dtype=np.float64)


def _frechet_to_target(
    cycle: frozenset[int],
    target_coords: np.ndarray,
    num_vertices: int,
    embedding: np.ndarray,
) -> float:
    cycle_coords = _cycle_coords(cycle, num_vertices, embedding)
    return float(
        min(
            compute_loop_frechet(cycle_coords, target_coords),
            compute_loop_frechet(target_coords, cycle_coords),
        )
    )


def _greedy_reversal(
    source: frozenset[int],
    target: frozenset[int],
    triangles: tuple[frozenset[int], ...],
    num_vertices: int,
    embedding: np.ndarray,
    death_scale: float | None = None,
) -> float | None:
    target_coords = _cycle_coords(target, num_vertices, embedding)
    source_coords = _cycle_coords(source, num_vertices, embedding)
    cycle = source
    reversal = 0.0
    total_step_abs = 0.0
    max_step_abs = 0.0
    n_steps = 0
    remaining = set(range(len(triangles)))

    while remaining:
        best_key = None
        best_step = None
        for triangle_index in remaining:
            triangle = triangles[triangle_index]
            next_cycle = _flip_triangle(cycle, triangle, num_vertices)
            if next_cycle is None:
                continue

            current_remaining_cost = _frechet_to_target(
                cycle, target_coords, num_vertices, embedding
            )
            next_remaining_cost = _frechet_to_target(
                next_cycle, target_coords, num_vertices, embedding
            )
            current_departed_cost = _frechet_to_target(
                cycle, source_coords, num_vertices, embedding
            )
            next_departed_cost = _frechet_to_target(
                next_cycle, source_coords, num_vertices, embedding
            )
            step_reversal = max(
                next_remaining_cost - current_remaining_cost,
                0.0,
            )
            step_abs = abs(next_remaining_cost - current_remaining_cost) + abs(
                next_departed_cost - current_departed_cost
            )
            key = (next_remaining_cost, triangle_index)
            if best_key is None or key < best_key:
                best_key = key
                best_step = (
                    triangle_index,
                    next_cycle,
                    step_reversal,
                    step_abs,
                )

        if best_step is None:
            return None

        triangle_index, cycle, step_reversal, step_abs = best_step
        remaining.remove(triangle_index)
        reversal += step_reversal
        total_step_abs += step_abs
        max_step_abs = max(max_step_abs, step_abs)
        n_steps += 1

    if cycle != target:
        return None
    # multiply by 2 due to two way frechet
    return 1 - min(max_step_abs / (2 * death_scale), 1)


def path_finding_coherence(
    source_edges: Sequence[int],
    target_edges: Sequence[int],
    triangles: Sequence[Sequence[int]],
    num_vertices: int,
    embedding: np.ndarray,
    death_scale: float | None = None,
) -> float | None:

    source = frozenset(source_edges)
    target = frozenset(target_edges)
    triangle_sets = tuple(frozenset(triangle) for triangle in triangles)
    forward = _greedy_reversal(
        source,
        target,
        triangle_sets,
        num_vertices,
        embedding,
        death_scale,
    )

    reverse = _greedy_reversal(
        target,
        source,
        triangle_sets,
        num_vertices,
        embedding,
        death_scale,
    )

    candidates = [cost for cost in (forward, reverse) if cost is not None]
    if not candidates:
        return None
    return min(candidates)


def compute_coherence(
    source_edges: Sequence[int],
    target_edges: Sequence[int],
    triangles: Sequence[Sequence[int]],
    num_vertices: int,
    embedding: np.ndarray,
    death_scale: float | None = None,
    method: HomotopyCoherenceMethod = "path_finding",
) -> float | None:
    return path_finding_coherence(
        source_edges,
        target_edges,
        triangles,
        num_vertices,
        embedding,
        death_scale,
    )
