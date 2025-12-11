# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Mchigan)
from .types import Index_t, Size_t
from numba import jit


@jit(nopython=True)
def edge_idx_encode(i: Index_t, j: Index_t, num_vertices: Size_t) -> Index_t:
    if i > j:
        i, j = j, i
    return i * num_vertices + j


@jit(nopython=True)
def edge_idx_decode(i: Index_t, num_vertices: Size_t) -> tuple[Index_t, Index_t]:
    return i // num_vertices, i % num_vertices


@jit(nopython=True)
def triangle_idx_encode(
    i: Index_t, j: Index_t, k: Index_t, num_vertices: Size_t
) -> Index_t:
    i, j, k = sorted([i, j, k])
    return i * num_vertices**2 + j * num_vertices + k


@jit(nopython=True)
def triangle_idx_decode(
    i: Index_t, num_vertices: Size_t
) -> tuple[Index_t, Index_t, Index_t]:
    return i // num_vertices**2, (i // num_vertices) % num_vertices, i % num_vertices
