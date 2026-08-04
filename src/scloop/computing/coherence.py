# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

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
    total_step_abs = 0.0
    remaining = set(range(len(triangles)))

    while remaining:
        best_key = None
        best_step = None
        for triangle_index in remaining:
            triangle = triangles[triangle_index]
            next_cycle = _flip_triangle(cycle, triangle, num_vertices)
            if next_cycle is None:
                continue

            current_remaining_cost = abs(target_length - length)
            next_length = _perimeter(next_cycle, edge_lengths)
            next_remaining_cost = abs(target_length - next_length)
            step_reversal = max(
                next_remaining_cost - current_remaining_cost,
                0.0,
            )
            step_abs = abs(next_length - length)
            key = (next_remaining_cost, triangle_index)
            if best_key is None or key < best_key:
                best_key = key
                best_step = (
                    triangle_index,
                    next_cycle,
                    next_length,
                    step_reversal,
                    step_abs,
                )

        if best_step is None:
            return None

        triangle_index, cycle, length, step_reversal, step_abs = best_step
        remaining.remove(triangle_index)
        reversal += step_reversal
        total_step_abs += step_abs

    return 1 - reversal / total_step_abs if cycle == target else None


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
    return max(candidates)


def compute_coherence(
    source_edges: Sequence[int],
    target_edges: Sequence[int],
    triangles: Sequence[Sequence[int]],
    edge_lengths: Mapping[int, float],
    num_vertices: int,
    method: HomotopyCoherenceMethod = "path_finding",
) -> float | None:
    return path_finding_coherence(
        source_edges,
        target_edges,
        triangles,
        edge_lengths,
        num_vertices,
    )
