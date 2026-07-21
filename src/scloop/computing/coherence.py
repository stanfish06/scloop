# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from math import sqrt
from typing import Mapping, Sequence

from ..data.types import HomotopyCoherenceMethod
from ..data.utils import edge_idx_decode


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


def _perimeter(cycle: frozenset[int], edge_lengths: Mapping[int, float]) -> float:
    return sum(edge_lengths[edge] for edge in cycle)


def _coherence(reversal_squared_sum: float, n_steps: int) -> float | None:
    if n_steps == 0:
        return None
    return 1.0 - sqrt(reversal_squared_sum / n_steps)


def exact_coherence(
    source_edges: Sequence[int],
    target_edges: Sequence[int],
    triangles: Sequence[Sequence[int]],
    edge_lengths: Mapping[int, float],
    num_vertices: int,
    max_triangles: int = 18,
) -> float | None:
    """Find the best coherence over all valid orders using subset dynamic programming."""

    if len(triangles) > max_triangles:
        raise ValueError(
            f"exact search is limited to {max_triangles} triangles; use path finding"
        )

    source = frozenset(source_edges)
    target = frozenset(target_edges)
    triangle_sets = tuple(frozenset(triangle) for triangle in triangles)
    full_mask = (1 << len(triangle_sets)) - 1

    cycles = {0: source}
    lengths = {0: _perimeter(source, edge_lengths)}
    reversals = {0: 0.0}
    target_length = _perimeter(target, edge_lengths)

    for mask in range(full_mask + 1):
        if mask not in cycles:
            continue
        cycle = cycles[mask]
        length = lengths[mask]
        for triangle_index, triangle in enumerate(triangle_sets):
            bit = 1 << triangle_index
            if mask & bit:
                continue
            next_cycle = _flip_triangle(cycle, triangle, num_vertices)
            if next_cycle is None:
                continue

            next_mask = mask | bit
            next_length = _perimeter(next_cycle, edge_lengths)
            cycles[next_mask] = next_cycle
            lengths[next_mask] = next_length
            step_reversal = max(
                abs(target_length - next_length) - abs(target_length - length),
                0.0,
            ) / ((length + next_length) / 2)
            reversals[next_mask] = min(
                reversals.get(next_mask, float("inf")),
                reversals[mask] + step_reversal**2,
            )

    if cycles.get(full_mask) != target:
        return None
    return _coherence(reversals[full_mask], len(triangle_sets))


def _greedy_reversal(
    source: frozenset[int],
    target: frozenset[int],
    triangles: tuple[frozenset[int], ...],
    edge_lengths: Mapping[int, float],
    num_vertices: int,
) -> float | None:
    """Follow one valid triangle order greedily by target perimeter distance."""

    target_length = _perimeter(target, edge_lengths)
    source_length = _perimeter(source, edge_lengths)
    cycle = source
    length = source_length
    reversal = 0.0
    remaining = set(range(len(triangles)))

    while remaining:
        best_key = None
        best_step = None
        for triangle_index in remaining:
            triangle = triangles[triangle_index]
            next_cycle = _flip_triangle(cycle, triangle, num_vertices)
            if next_cycle is None:
                continue

            next_length = _perimeter(next_cycle, edge_lengths)
            next_remaining_cost = abs(target_length - next_length)
            step_reversal = max(
                next_remaining_cost - abs(target_length - length),
                0.0,
            ) / max(length, next_length)
            key = (next_remaining_cost, triangle_index)
            if best_key is None or key < best_key:
                best_key = key
                best_step = (triangle_index, next_cycle, next_length, step_reversal)

        if best_step is None:
            return None

        triangle_index, cycle, length, step_reversal = best_step
        remaining.remove(triangle_index)
        reversal += step_reversal**2

    return reversal if cycle == target else None


def path_finding_coherence(
    source_edges: Sequence[int],
    target_edges: Sequence[int],
    triangles: Sequence[Sequence[int]],
    edge_lengths: Mapping[int, float],
    num_vertices: int,
) -> float | None:
    """Return coherence from a greedy valid triangle order."""

    source = frozenset(source_edges)
    target = frozenset(target_edges)
    triangle_sets = tuple(frozenset(triangle) for triangle in triangles)
    forward = _greedy_reversal(
        source,
        target,
        triangle_sets,
        edge_lengths,
        num_vertices,
    )
    reverse = _greedy_reversal(
        target,
        source,
        triangle_sets,
        edge_lengths,
        num_vertices,
    )

    candidates = [cost for cost in (forward, reverse) if cost is not None]
    if not candidates:
        return None
    return _coherence(min(candidates), len(triangle_sets))


def compute_coherence(
    source_edges: Sequence[int],
    target_edges: Sequence[int],
    triangles: Sequence[Sequence[int]],
    edge_lengths: Mapping[int, float],
    num_vertices: int,
    method: HomotopyCoherenceMethod = "path_finding",
    max_triangles: int = 18,
) -> float | None:
    match method:
        case "exact":
            return exact_coherence(
                source_edges,
                target_edges,
                triangles,
                edge_lengths,
                num_vertices,
                max_triangles,
            )
        case "path_finding":
            return path_finding_coherence(
                source_edges,
                target_edges,
                triangles,
                edge_lengths,
                num_vertices,
            )
