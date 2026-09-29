# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
from __future__ import annotations

import math
from typing import Iterable, Iterator, List, Sequence, Tuple

import igraph as ig
import numpy as np
from loguru import logger
from numba import jit
from pydantic import PositiveFloat
from scipy.sparse import csr_matrix, triu
from sklearn.neighbors import NearestNeighbors

from ..data.base_components import LoopClass
from ..data.boundary import BoundaryMatrixD1
from ..data.constants import (
    DEFAULT_FOREIGN_CHORD_MULT,
    DEFAULT_K_LOCAL_SCALE,
    DEFAULT_K_YEN,
    DEFAULT_LIFE_PCT,
    DEFAULT_MAX_INSERT_PER_EDGE,
    DEFAULT_MAX_PERIMETER_MULT,
    DEFAULT_N_COCYCLES_USED,
    DEFAULT_N_FORCE_DEVIATE,
    DEFAULT_N_REPS_PER_LOOP,
    DEFAULT_SPLIT_EDGE_LENGTH_MULT,
    DEFAULT_SPLIT_POINT_DISTANCE_MULT,
    NUMERIC_EPSILON,
)
from ..data.types import Count_t, Percent_t
from ..data.utils import extract_edges_from_coo, loops_to_coords


def remap_cocycles_for_full_reconstruction(
    cocycles: list,
    bootstrap_vertex_ids: list[int],
    full_vertex_ids: list[int],
) -> list:
    global_to_full_local = {gid: lid for lid, gid in enumerate(full_vertex_ids)}

    remapped_cocycles = []
    for cocycle in cocycles:
        remapped_simplex = []
        for simplex in cocycle:
            try:
                verts, coeff = simplex
            except ValueError:
                continue
            if coeff == 0 or len(verts) != 2:
                continue
            global_u = bootstrap_vertex_ids[int(verts[0])]
            global_v = bootstrap_vertex_ids[int(verts[1])]
            if global_u == global_v:
                continue
            if global_u in global_to_full_local and global_v in global_to_full_local:
                full_local_u = global_to_full_local[global_u]
                full_local_v = global_to_full_local[global_v]
                remapped_simplex.append(((full_local_u, full_local_v), coeff))
        remapped_cocycles.append(remapped_simplex)
    return remapped_cocycles


# ISSUE: this often mess up loop reconstruction
@jit(nopython=True, cache=True)
def _clean_cocycle_region_impl(
    edges: np.ndarray,
    cocycle_vertices: np.ndarray,
    n_vertices: int,
) -> np.ndarray:
    n_edges = edges.shape[0]
    n_cocycle = len(cocycle_vertices)

    is_cocycle_vertex = np.zeros(n_vertices, dtype=np.bool_)
    for i in range(n_cocycle):
        v = cocycle_vertices[i]
        if v < n_vertices:
            is_cocycle_vertex[v] = True

    block_mask = np.zeros(n_edges, dtype=np.bool_)
    for i in range(n_edges):
        u, v = edges[i, 0], edges[i, 1]
        if is_cocycle_vertex[u] and is_cocycle_vertex[v]:
            block_mask[i] = True

    return block_mask


def clean_cocycle_region(
    edges: np.ndarray,
    cocycle_edges: list[tuple[int, int]],
) -> set[tuple[int, int]]:
    if len(cocycle_edges) == 0 or len(edges) == 0:
        return set()

    cocycle_vertices_set = set()
    for u, v in cocycle_edges:
        cocycle_vertices_set.add(u)
        cocycle_vertices_set.add(v)
    cocycle_vertices = np.array(list(cocycle_vertices_set), dtype=np.int64)

    n_vertices = max(edges[:, 0].max(), edges[:, 1].max()) + 1
    n_vertices = max(
        n_vertices, cocycle_vertices.max() + 1 if len(cocycle_vertices) > 0 else 0
    )

    block_mask = _clean_cocycle_region_impl(edges, cocycle_vertices, n_vertices)

    edges_to_block = set()
    for i in range(len(edges)):
        if block_mask[i]:
            u, v = int(edges[i, 0]), int(edges[i, 1])
            edges_to_block.add((min(u, v), max(u, v)))

    return edges_to_block


def _iter_cocycle_edges(cocycles_dim1: Iterable) -> Iterator[tuple[object, int, int]]:
    for simplex in cocycles_dim1:
        try:
            verts, coeff = simplex
        except ValueError:
            continue
        if coeff == 0 or len(verts) != 2:
            continue
        yield simplex, int(verts[0]), int(verts[1])


def compute_loop_representatives(
    embedding: np.ndarray,
    pairwise_distance_matrix: csr_matrix,
    persistence_diagram: tuple,
    cocycles: list,
    boundary_matrix_d1: BoundaryMatrixD1,
    vertex_ids: list[int],
    persistence_pair_simplices: tuple[list, list] | list | None = None,
    top_k: Count_t | None = None,
    n_reps_per_loop: int = DEFAULT_N_REPS_PER_LOOP,
    life_pct: Percent_t = DEFAULT_LIFE_PCT,
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
    bootstrap: bool = False,
    rank_offset: int = 0,
    do_clean_cocycle_region: bool = False,
    foreign_chord_mult: float = DEFAULT_FOREIGN_CHORD_MULT,
    max_perimeter_mult: float = DEFAULT_MAX_PERIMETER_MULT,
    validate_representatives: bool = True,
) -> list[LoopClass | None]:
    assert pairwise_distance_matrix.shape is not None

    loop_births = np.array(persistence_diagram[0], dtype=np.float32)
    loop_deaths = np.array(persistence_diagram[1], dtype=np.float32)

    if loop_births.size == 0:
        return []
    if top_k is None:
        top_k = loop_births.size
    if top_k <= 0:
        return []
    top_k = min(top_k, loop_births.size)

    persistence = loop_deaths - loop_births
    indices_top_k = np.argsort(persistence)[::-1][:top_k]

    dm_upper = triu(pairwise_distance_matrix, k=1).tocoo()
    edges_array, edge_diameters = extract_edges_from_coo(
        dm_upper.row, dm_upper.col, dm_upper.data
    )

    if len(edges_array) == 0:
        return []

    boundary_edge_set = boundary_matrix_d1.edge_set

    results: list[LoopClass | None] = [None] * len(indices_top_k)

    build_foreign = foreign_chord_mult > 1.0
    check_reps = validate_representatives and persistence_pair_simplices is not None
    death_keys: list[tuple[float, tuple[int, ...]]] = []
    if check_reps:
        assert persistence_pair_simplices is not None
        death_keys = [
            (float(loop_deaths[c]), tuple(-v for v in sorted(simplex, reverse=True)))
            if len(simplex) > 0
            else (math.inf, ())
            for c, simplex in enumerate(persistence_pair_simplices[1])
        ]
    cocycle_edges_per_class: dict[int, set[tuple[int, int]]] = {}
    if build_foreign or check_reps:
        for cls_idx in range(len(cocycles)):
            cocycle_edges_per_class[cls_idx] = {
                (min(a, b), max(a, b))
                for _, a, b in _iter_cocycle_edges(cocycles[cls_idx])
                if a != b
            }

    for i, loop_idx in enumerate(indices_top_k):
        loop_birth = loop_births[loop_idx].item()
        loop_death = loop_deaths[loop_idx].item()
        birth_simplex: list[int] = []
        death_simplex: list[int] = []
        if persistence_pair_simplices is not None:
            births, deaths = persistence_pair_simplices
            if loop_idx < len(births):
                birth_simplex = list(births[loop_idx])
            if loop_idx < len(deaths):
                death_simplex = list(deaths[loop_idx])

        valid_cocycles = []
        n_cocycles_original = len(cocycles[loop_idx])

        for simplex, u_local, v_local in _iter_cocycle_edges(cocycles[loop_idx]):
            u_global = vertex_ids[u_local]
            v_global = vertex_ids[v_local]
            edge_global = (min(u_global, v_global), max(u_global, v_global))

            if bootstrap:
                if edge_global in boundary_edge_set or u_global == v_global:
                    valid_cocycles.append(simplex)
            else:
                if edge_global in boundary_edge_set:
                    valid_cocycles.append(simplex)

        if len(valid_cocycles) == 0:
            logger.warning(
                f"Loop class {i + rank_offset}: All {n_cocycles_original} cocycle edges "
                f"filtered (not in boundary matrix). Skipping reconstruction."
            )
            continue

        if len(valid_cocycles) < n_cocycles_original:
            logger.info(
                f"Loop class {i + rank_offset}: Filtered {n_cocycles_original - len(valid_cocycles)}/"
                f"{n_cocycles_original} cocycle edges not in boundary matrix"
            )

        valid_edge_mask = []
        for edge_local in edges_array:
            u_global = vertex_ids[int(edge_local[0])]
            v_global = vertex_ids[int(edge_local[1])]
            edge_global = (min(u_global, v_global), max(u_global, v_global))
            if bootstrap:
                valid = (edge_global in boundary_edge_set) or (u_global == v_global)
            else:
                valid = edge_global in boundary_edge_set
            valid_edge_mask.append(valid)

        valid_edge_mask = np.array(valid_edge_mask, dtype=bool)
        edges_array_filtered = edges_array[valid_edge_mask]
        edge_diameters_filtered = edge_diameters[valid_edge_mask]

        foreign_cocycle_edges: list[tuple[int, int]] | None = None
        if build_foreign:
            foreign_set: set[tuple[int, int]] = set()
            for cls_idx, edge_set in cocycle_edges_per_class.items():
                if cls_idx == int(loop_idx):
                    continue
                foreign_set |= edge_set
            foreign_cocycle_edges = list(foreign_set)

        loops_local, _ = reconstruct_n_loop_representatives(
            cocycles_dim1=valid_cocycles,
            edges=edges_array_filtered,
            edge_diameters=edge_diameters_filtered,
            loop_birth=loop_birth,
            loop_death=loop_death,
            n=n_reps_per_loop,
            life_pct=life_pct,
            n_force_deviate=n_force_deviate,
            k_yen=k_yen,
            loop_lower_pct=loop_lower_t_pct,
            loop_upper_pct=loop_upper_t_pct,
            n_cocycles_used=n_cocycles_used,
            do_random_walk=do_random_walk,
            n_random_graphs=n_random_graphs,
            decay_random_walk=decay_random_walk,
            noise_random_walk=noise_random_walk,
            seed_random_walk=seed_random_walk,
            do_force_deviate_random_walk=do_force_deviate_random_walk,
            do_clean_cocycle_region=do_clean_cocycle_region,
            foreign_cocycle_edges=foreign_cocycle_edges,
            foreign_chord_mult=foreign_chord_mult,
            max_perimeter_mult=max_perimeter_mult,
        )

        representatives_valid = None
        if check_reps:
            own_key = death_keys[loop_idx]
            alive = [
                c
                for c in range(len(death_keys))
                if loop_births[c] <= own_key[0] and death_keys[c] >= own_key
            ]
            representatives_valid = [
                _is_death_cycle(loop, int(loop_idx), alive, cocycle_edges_per_class)
                for loop in loops_local
            ]

        loops = [[vertex_ids[v] for v in loop] for loop in loops_local]
        loops_coords = loops_to_coords(embedding=embedding, loops_vertices=loops)

        results[i] = LoopClass(
            rank=i + rank_offset,
            persistence_index=int(loop_idx),
            birth=loop_birth,
            death=loop_death,
            birth_simplex=birth_simplex,
            death_simplex=death_simplex,
            cocycles=cocycles[loop_idx],
            representatives=loops,
            representatives_valid=representatives_valid,
            coordinates_vertices_representatives=loops_coords,
        )

    return results


def _is_death_cycle(
    loop: Sequence[int],
    own_class: int,
    alive_classes: list[int],
    cocycle_edges: dict[int, set[tuple[int, int]]],
) -> bool:
    """[loop] = [∂τ] just before the death triangle τ enters, via the alive cocycle basis."""
    edges: set[tuple[int, int]] = set()
    for u, v in zip(loop, [*loop[1:], loop[0]]):
        if u != v:
            edges ^= {(min(u, v), max(u, v))}
    return all(
        len(edges & cocycle_edges[c]) % 2 == int(c == own_class) for c in alive_classes
    )


def reconstruct_n_loop_representatives(
    cocycles_dim1: List,
    edges: np.ndarray,
    edge_diameters: np.ndarray,
    loop_birth: PositiveFloat,
    loop_death: PositiveFloat,
    n: Count_t,
    life_pct: Percent_t = DEFAULT_LIFE_PCT,
    n_force_deviate: Count_t = DEFAULT_N_FORCE_DEVIATE,
    k_yen: Count_t = DEFAULT_K_YEN,
    loop_lower_pct: float = 5,
    loop_upper_pct: float = 95,
    n_cocycles_used: Count_t = DEFAULT_N_COCYCLES_USED,
    do_random_walk: bool = False,  # random walk works but still less robust than the force deviate branch
    n_random_graphs: Count_t = 10,
    decay_random_walk: PositiveFloat = 1.0,
    noise_random_walk: PositiveFloat = 1.0,
    seed_random_walk: int = 1,
    do_force_deviate_random_walk: bool = False,
    *,
    do_clean_cocycle_region: bool = False,
    foreign_cocycle_edges: list[tuple[int, int]] | None = None,
    foreign_chord_mult: float = DEFAULT_FOREIGN_CHORD_MULT,
    max_perimeter_mult: float = DEFAULT_MAX_PERIMETER_MULT,
) -> Tuple[List[List[int]], List[float]]:
    """
    Reconstruct diverse loop representatives using shortest paths or random walks
    """
    if n <= 0 or len(edges) == 0:
        return [], []
    filt_t = loop_birth + (loop_death - loop_birth) * life_pct

    all_cocycle_edges: list[tuple[int, int]] = [
        (a, b) for _, a, b in _iter_cocycle_edges(cocycles_dim1)
    ]

    if not all_cocycle_edges:
        return [], []

    cocycle_edges_for_paths = all_cocycle_edges[:n_cocycles_used]

    edge_diameters = np.asarray(edge_diameters)
    mask = edge_diameters <= filt_t
    if not np.any(mask):
        return [], []
    edges_filt = edges[mask]
    weights_filt = edge_diameters[mask]

    edge_weight_dict: dict[tuple[int, int], float] = {}
    for i in range(len(edges_filt)):
        key = (
            min(edges_filt[i, 0], edges_filt[i, 1]),
            max(edges_filt[i, 0], edges_filt[i, 1]),
        )
        edge_weight_dict[key] = max(
            edge_weight_dict.get(key, -math.inf), weights_filt[i]
        )

    for e in all_cocycle_edges:
        key = (min(e), max(e))
        edge_weight_dict[key] = math.inf

    if do_clean_cocycle_region:
        edges_to_block = clean_cocycle_region(
            edges=edges_filt,
            cocycle_edges=all_cocycle_edges,
        )
        for edge_key in edges_to_block:
            if edge_key in edge_weight_dict:
                edge_weight_dict[edge_key] = math.inf

    cycles_pool: list[list[int]] = []
    cycles_dist: list[float] = []

    _n_trials: Count_t = n_random_graphs if do_random_walk else n_force_deviate

    edge_list = list(edge_weight_dict.keys())
    if not edge_list:
        return [], []
    n_vertices = max(max(e) for e in edge_list) + 1

    factor_array = np.ones(len(edge_list), dtype=np.float64)
    if foreign_cocycle_edges and foreign_chord_mult > 1.0:
        own_chord_keys = {(min(e), max(e)) for e in all_cocycle_edges}
        foreign_keys = {
            (min(u, v), max(u, v)) for u, v in foreign_cocycle_edges
        } - own_chord_keys
        for idx, e in enumerate(edge_list):
            if e in foreign_keys:
                factor_array[idx] = foreign_chord_mult

    g = ig.Graph(n=n_vertices, edges=edge_list, directed=False)
    base_weight_list = [edge_weight_dict[e] for e in edge_list]
    g.es["base_weight"] = base_weight_list
    rng = np.random.default_rng(seed_random_walk)
    for _ in range(_n_trials):
        weight_array = (
            np.array([edge_weight_dict[e] for e in edge_list], dtype=np.float64)
            * factor_array
        )

        if do_random_walk:
            # Gumbel max trick to simulate/approximate random walk bridge
            weight_list_perturbed = (
                decay_random_walk * weight_array
                - noise_random_walk
                * rng.gumbel(loc=0.0, scale=1.0, size=len(weight_array))
            )
            weight_list_perturbed -= np.min(weight_list_perturbed)
            weight_list_perturbed += NUMERIC_EPSILON
            g.es["weight"] = list(weight_list_perturbed)
        else:
            g.es["weight"] = list(weight_array)

        paths_this_round: list[list[int]] = []
        for i, j in cocycle_edges_for_paths:
            paths = _k_shortest_paths(g, i, j, k_yen)
            if not paths:
                continue
            for path in paths:
                dist = _path_weight(g, path)
                if math.isfinite(dist):
                    cycles_pool.append(path)
                    paths_this_round.append(path)
                    cycles_dist.append(dist)

        if (not do_random_walk) or do_force_deviate_random_walk:
            for path in paths_this_round:
                for u, v in zip(path[:-1], path[1:]):
                    key = (min(u, v), max(u, v))
                    edge_weight_dict[key] = math.inf

    return _select_diverse_loops(
        cycles=cycles_pool,
        distances=cycles_dist,
        n=n,
        lower_pct=loop_lower_pct,
        upper_pct=loop_upper_pct,
        max_perimeter_mult=max_perimeter_mult,
    )


def _k_shortest_paths(g: ig.Graph, source: int, target: int, k: int) -> list[list[int]]:
    if source == target:
        return []
    try:
        return g.get_k_shortest_paths(
            source, target, k=k, weights=g.es["weight"], mode="ALL"
        )
    except ig.InternalError:
        return []


def _path_weight(g: ig.Graph, path: Sequence[int]) -> float:
    if len(path) < 2:
        return math.inf
    weight = 0.0
    for u, v in zip(path[:-1], path[1:]):
        try:
            eid = g.get_eid(u, v, directed=False)
        except ig.InternalError:
            return math.inf
        w = g.es[eid]["base_weight"]
        if not math.isfinite(w):
            return math.inf
        weight += float(w)
    return weight


def _select_diverse_loops(
    cycles: Iterable[Sequence[int]],
    distances: Iterable[float],
    n: int,
    lower_pct: float,
    upper_pct: float,
    max_perimeter_mult: float = DEFAULT_MAX_PERIMETER_MULT,
) -> Tuple[List[List[int]], List[float]]:
    pairs = sorted(
        [(d, list(c)) for d, c in zip(distances, cycles) if math.isfinite(d)],
        key=lambda x: x[0],
    )
    if not pairs:
        return [], []

    # Gate long/off-rail candidates against the shortest (most trusted) rep,
    # then run the usual percentile diversity sampling on what remains.
    if math.isfinite(max_perimeter_mult) and max_perimeter_mult > 0.0:
        l_min = pairs[0][0]
        max_len = max_perimeter_mult * l_min
        pairs = [p for p in pairs if p[0] <= max_len]
        if not pairs:
            return [], []

    n_total = len(pairs)
    n_return = min(n_total, n)
    if n_return == 1:
        idxs = [n_total // 2]
    else:
        step = (upper_pct - lower_pct) / (n_return - 1)
        idxs = []
        for i in range(n_return):
            pct = (lower_pct + step * i) / 100
            idx = min(math.floor(n_total * pct), n_total - 1)
            idxs.append(idx)

    selected = [pairs[i] for i in idxs]
    dists = [p[0] for p in selected]
    loops = [p[1] for p in selected]
    return loops, dists


def _distances_to_edge(points: np.ndarray, a: np.ndarray, b: np.ndarray) -> np.ndarray:
    ab = b - a
    denom = float(ab @ ab)
    if denom == 0.0:
        return np.linalg.norm(points - a, axis=1)
    s = np.clip((points - a) @ ab / denom, 0.0, 1.0)
    return np.linalg.norm(points - (a + s[:, None] * ab), axis=1)


def _select_split_point(
    a: int,
    b: int,
    u: int,
    v: int,
    embedding: np.ndarray,
    used_mask: np.ndarray,
    max_diameter: float,
    limit: float | None,
    pool: np.ndarray | None,
    pool_embedding: np.ndarray,
) -> int | None:
    points = pool_embedding
    d_a = np.linalg.norm(points - embedding[a], axis=1)
    mask = d_a <= max_diameter
    mask &= ~used_mask if pool is None else ~used_mask[pool]
    idx = np.flatnonzero(mask)
    if idx.size == 0:
        return None
    candidate_points = points[idx]
    keep = np.linalg.norm(candidate_points - embedding[b], axis=1) <= max_diameter
    if not keep.any():
        return None
    candidates = (idx if pool is None else pool[idx])[keep]
    candidate_points = candidate_points[keep]

    if limit is not None:
        keep = _distances_to_edge(candidate_points, embedding[u], embedding[v]) <= limit
        if not keep.any():
            return None
        candidates = candidates[keep]
        candidate_points = candidate_points[keep]

    midpoint = 0.5 * (embedding[a] + embedding[b])
    return int(
        candidates[np.argmin(np.linalg.norm(candidate_points - midpoint, axis=1))]
    )


def _densify_loops(
    vertices: list[int],
    embedding: np.ndarray,
    local_scale: np.ndarray,
    max_diameter: float,
    max_insert_per_edge: int,
    split_edge_length_mult: float | None = None,
    split_point_distance_mult: float | None = None,
) -> list[int]:
    if len(vertices) < 2:
        return list(vertices)

    refined: list[int] = [vertices[0]]
    used_mask = np.zeros(embedding.shape[0], dtype=bool)
    used_mask[vertices] = True

    if split_edge_length_mult is None:
        split_edge_length_mult = 1.0

    ball_cutoff = 0.25 * float(
        np.linalg.norm(embedding.max(axis=0) - embedding.min(axis=0))
    )

    def _target_length(a: int, b: int) -> float:
        return float(split_edge_length_mult * 0.5 * (local_scale[a] + local_scale[b]))

    for u, v in zip(vertices[:-1], vertices[1:]):
        poly = [u, v]
        lengths = [float(np.linalg.norm(embedding[u] - embedding[v]))]
        targets = [_target_length(u, v)]
        limit = (
            split_point_distance_mult * 0.5 * (local_scale[u] + local_scale[v])
            if split_point_distance_mult is not None
            else None
        )
        pool: np.ndarray | None = None
        pool_embedding: np.ndarray = embedding
        limit_select = limit
        pool_ready = limit is None
        stalled_pairs: set[tuple[int, int]] = set()
        n_inserted = 0
        while n_inserted < max_insert_per_edge:
            edges_to_split = []
            for i in range(len(poly) - 1):
                a, b = poly[i], poly[i + 1]
                if (a, b) in stalled_pairs:
                    continue
                if lengths[i] > targets[i]:
                    edges_to_split.append((targets[i] / lengths[i], i, a, b))
            if not edges_to_split:
                break
            edges_to_split.sort(key=lambda x: -x[0])
            if not pool_ready:
                assert limit is not None
                half_len = 0.5 * float(np.linalg.norm(embedding[u] - embedding[v]))
                if limit + half_len < ball_cutoff:
                    mid_uv = 0.5 * (embedding[u] + embedding[v])
                    near_mask = np.linalg.norm(embedding - mid_uv, axis=1) <= (
                        limit + half_len
                    ) * (1.0 + 1e-12)
                    if 2 * int(near_mask.sum()) <= embedding.shape[0]:
                        near = np.flatnonzero(near_mask)
                        pool = near[
                            _distances_to_edge(
                                embedding[near], embedding[u], embedding[v]
                            )
                            <= limit
                        ]
                        pool_embedding = embedding[pool]
                        limit_select = None
                pool_ready = True

            inserted_this_round = False
            for _, i, a, b in edges_to_split:
                p = _select_split_point(
                    a=a,
                    b=b,
                    u=u,
                    v=v,
                    embedding=embedding,
                    used_mask=used_mask,
                    max_diameter=max_diameter,
                    limit=limit_select,
                    pool=pool,
                    pool_embedding=pool_embedding,
                )
                if p is None:
                    stalled_pairs.add((a, b))
                    continue
                poly.insert(i + 1, p)
                lengths[i] = float(np.linalg.norm(embedding[a] - embedding[p]))
                lengths.insert(
                    i + 1, float(np.linalg.norm(embedding[p] - embedding[b]))
                )
                targets[i] = _target_length(a, p)
                targets.insert(i + 1, _target_length(p, b))
                used_mask[p] = True
                n_inserted += 1
                inserted_this_round = True
                break
            if not inserted_this_round:
                break
        refined.extend(poly[1:])

    return refined


def refine_loop_representatives(
    loop_classes: list[LoopClass],
    embedding: np.ndarray,
    local_scale: np.ndarray | None = None,
    max_insert_per_edge: int = DEFAULT_MAX_INSERT_PER_EDGE,
    split_edge_length_mult: float | None = DEFAULT_SPLIT_EDGE_LENGTH_MULT,
    split_point_distance_mult: float | None = DEFAULT_SPLIT_POINT_DISTANCE_MULT,
    life_pct: float = 0.0,
    k_local_scale: int = DEFAULT_K_LOCAL_SCALE,
) -> None:
    if local_scale is None:
        nn = NearestNeighbors(n_neighbors=k_local_scale + 1).fit(embedding)
        knn_distances, _ = nn.kneighbors(embedding)
        local_scale = np.asarray(knn_distances[:, 1:].mean(axis=1), dtype=np.float64)
    for loop_class in loop_classes:
        if loop_class is None or not loop_class.representatives:
            continue
        max_diameter = loop_class.birth + life_pct * (
            loop_class.death - loop_class.birth
        )
        refined_all = []
        for rep in loop_class.representatives:
            refined = _densify_loops(
                vertices=list(rep),
                embedding=embedding,
                local_scale=local_scale,
                max_diameter=max_diameter,
                max_insert_per_edge=max_insert_per_edge,
                split_edge_length_mult=split_edge_length_mult,
                split_point_distance_mult=split_point_distance_mult,
            )
            refined_all.append(refined)
        loop_class.representatives_refined = refined_all
