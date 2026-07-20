# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

from heapq import heappop, heappush
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


def _coherence(source_length: float, target_length: float, reversal: float) -> float:
    if reversal == 0:
        return 1.0
    net_change = abs(target_length - source_length)
    return net_change / (net_change + 2 * reversal)


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
    penalize_contraction = lengths[0] <= _perimeter(target, edge_lengths)

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
            change = next_length - length
            cycles[next_mask] = next_cycle
            lengths[next_mask] = next_length
            step_reversal = (
                max(-change, 0.0) if penalize_contraction else max(change, 0.0)
            )
            reversals[next_mask] = min(
                reversals.get(next_mask, float("inf")),
                reversals[mask] + step_reversal,
            )

    if cycles.get(full_mask) != target:
        return None
    return _coherence(lengths[0], lengths[full_mask], reversals[full_mask])


def _path_finding_reversal(
    source: frozenset[int],
    target: frozenset[int],
    triangles: tuple[frozenset[int], ...],
    edge_lengths: Mapping[int, float],
    num_vertices: int,
    max_states: int,
) -> float | None:
    """Find a low-reversal valid triangle order within a state budget."""

    full_mask = (1 << len(triangles)) - 1
    target_length = _perimeter(target, edge_lengths)
    source_length = _perimeter(source, edge_lengths)
    penalize_contraction = source_length <= target_length

    queue = [(0.0, len(source ^ target), 0)]
    best_cost = {0: 0.0}
    cycles = {0: source}
    lengths = {0: source_length}
    best_complete = None
    states_expanded = 0

    while queue and states_expanded < max_states:
        cost, _, mask = heappop(queue)
        if cost != best_cost.get(mask):
            continue
        states_expanded += 1
        if mask == full_mask:
            if cycles[mask] == target:
                return cost
            continue

        cycle = cycles[mask]
        length = lengths[mask]
        for triangle_index, triangle in enumerate(triangles):
            bit = 1 << triangle_index
            if mask & bit:
                continue
            next_cycle = _flip_triangle(cycle, triangle, num_vertices)
            if next_cycle is None:
                continue

            next_mask = mask | bit
            next_length = _perimeter(next_cycle, edge_lengths)
            change = next_length - length
            step_cost = max(-change, 0.0) if penalize_contraction else max(change, 0.0)
            next_cost = cost + step_cost
            if next_cost >= best_cost.get(next_mask, float("inf")):
                continue

            best_cost[next_mask] = next_cost
            cycles[next_mask] = next_cycle
            lengths[next_mask] = next_length
            heappush(queue, (next_cost, len(next_cycle ^ target), next_mask))
            if next_mask == full_mask and next_cycle == target:
                best_complete = (
                    next_cost
                    if best_complete is None
                    else min(best_complete, next_cost)
                )

    return best_complete


def path_finding_coherence(
    source_edges: Sequence[int],
    target_edges: Sequence[int],
    triangles: Sequence[Sequence[int]],
    edge_lengths: Mapping[int, float],
    num_vertices: int,
    max_states: int,
) -> float | None:
    """Return the best coherence found within a state budget."""

    if max_states <= 0:
        raise ValueError("max_states must be positive")

    source = frozenset(source_edges)
    target = frozenset(target_edges)
    triangle_sets = tuple(frozenset(triangle) for triangle in triangles)
    forward = _path_finding_reversal(
        source,
        target,
        triangle_sets,
        edge_lengths,
        num_vertices,
        max_states,
    )
    reverse = _path_finding_reversal(
        target,
        source,
        triangle_sets,
        edge_lengths,
        num_vertices,
        max_states,
    )

    candidates = [cost for cost in (forward, reverse) if cost is not None]
    if not candidates:
        return None
    reversal = min(candidates)
    return _coherence(
        _perimeter(source, edge_lengths),
        _perimeter(target, edge_lengths),
        reversal,
    )


def compute_coherence(
    source_edges: Sequence[int],
    target_edges: Sequence[int],
    triangles: Sequence[Sequence[int]],
    edge_lengths: Mapping[int, float],
    num_vertices: int,
    method: HomotopyCoherenceMethod = "path_finding",
    max_states: int = 10_000,
    max_triangles: int = 18,
) -> float | None:
    if method == "exact":
        return exact_coherence(
            source_edges,
            target_edges,
            triangles,
            edge_lengths,
            num_vertices,
            max_triangles,
        )
    if method == "path_finding":
        return path_finding_coherence(
            source_edges,
            target_edges,
            triangles,
            edge_lengths,
            num_vertices,
            max_states,
        )
    raise ValueError(f"unknown coherence method: {method}")
