# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
"""Sparse GF2 solver for A x = b with few nonzeros per column (boundary matrices).

Columns are reduced left to right with a pivot table keyed by each column's lowest
(largest-index) row, tracking which original columns were combined. The columns that
do not reduce to zero form the column rank profile of A (its leftmost independent
columns); each b is reduced against them and every other variable is set to 0.
"""

from __future__ import annotations

import numpy as np
from numba import njit, types
from numba.typed import List

_INT_ARRAY = types.int64[::1]


@njit(cache=True)
def _symmetric_difference(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """XOR of two sorted, duplicate-free index arrays."""
    out = np.empty(a.size + b.size, np.int64)
    i = j = k = 0
    while i < a.size and j < b.size:
        if a[i] < b[j]:
            out[k] = a[i]
            i += 1
            k += 1
        elif a[i] > b[j]:
            out[k] = b[j]
            j += 1
            k += 1
        else:
            i += 1
            j += 1
    while i < a.size:
        out[k] = a[i]
        i += 1
        k += 1
    while j < b.size:
        out[k] = b[j]
        j += 1
        k += 1
    return out[:k].copy()


@njit(cache=True)
def _reduce_and_solve(
    indptr: np.ndarray,
    indices: np.ndarray,
    nrow: int,
    ncol: int,
    b_indptr: np.ndarray,
    b_indices: np.ndarray,
    solutions: np.ndarray,
    states: np.ndarray,
) -> None:
    pivot_of_row = np.full(nrow, -1, np.int64)
    reduced = List.empty_list(_INT_ARRAY)
    combination = List.empty_list(_INT_ARRAY)
    for c in range(ncol):
        col = indices[indptr[c] : indptr[c + 1]].copy()
        comb = np.array([c], dtype=np.int64)
        while col.size > 0:
            p = pivot_of_row[col[-1]]
            if p < 0:
                pivot_of_row[col[-1]] = len(reduced)
                reduced.append(col)
                combination.append(comb)
                break
            col = _symmetric_difference(col, reduced[p])
            comb = _symmetric_difference(comb, combination[p])

    for j in range(b_indptr.size - 1):
        rhs = b_indices[b_indptr[j] : b_indptr[j + 1]].copy()
        x = np.empty(0, np.int64)
        states[j] = 0
        while rhs.size > 0:
            p = pivot_of_row[rhs[-1]]
            if p < 0:
                states[j] = -1
                break
            rhs = _symmetric_difference(rhs, reduced[p])
            x = _symmetric_difference(x, combination[p])
        if states[j] == 0:
            for t in x:
                solutions[j, t] = 1


def solve_multiple_gf2(one_ridx_A, one_cidx_A, nrow_A, ncol_A, one_idx_b_list):
    """Solve A x = b over GF2 for every b in one_idx_b_list.

    Same interface as the M4RI binding: A is given by the row/column indices of its
    ones (repeated entries count once), each b by the indices of its ones. Returns
    (states, solutions): state 0 with x as a list of bits, or -1 with None when
    A x = b has no solution.
    """
    n_systems = len(one_idx_b_list)
    if n_systems == 0:
        return [], []
    if ncol_A == 0:
        states = [0 if len(b) == 0 else -1 for b in one_idx_b_list]
        return states, [[] if s == 0 else None for s in states]

    ridx = np.asarray(one_ridx_A, dtype=np.int64)
    cidx = np.asarray(one_cidx_A, dtype=np.int64)
    # CSC layout: unique (col, row) keys sorted by column, then row
    keys = np.unique(cidx * nrow_A + ridx)
    cols, rows = keys // nrow_A, keys % nrow_A
    indptr = np.searchsorted(cols, np.arange(ncol_A + 1)).astype(np.int64)

    b_list = [np.unique(np.asarray(b, dtype=np.int64)) for b in one_idx_b_list]
    b_indptr = np.zeros(n_systems + 1, dtype=np.int64)
    b_indptr[1:] = np.cumsum([b.size for b in b_list])
    b_indices = np.concatenate(b_list).astype(np.int64)

    solutions = np.zeros((n_systems, ncol_A), dtype=np.uint8)
    states = np.zeros(n_systems, dtype=np.int64)
    _reduce_and_solve(
        indptr, rows, nrow_A, ncol_A, b_indptr, b_indices, solutions, states
    )
    state_list = states.tolist()
    return state_list, [
        solutions[j].tolist() if state_list[j] == 0 else None for j in range(n_systems)
    ]


@njit(cache=True)
def _reduce_columns(
    indptr: np.ndarray, indices: np.ndarray, nrow: int, ncol: int
) -> tuple:
    pivot_of_row = np.full(nrow, -1, np.int64)
    pivot_position = np.empty(min(nrow, ncol), np.int64)
    reduced = List.empty_list(_INT_ARRAY)
    combination = List.empty_list(_INT_ARRAY)
    used = np.empty(nrow, np.int64)
    for c in range(ncol):
        col = indices[indptr[c] : indptr[c + 1]].copy()
        n_used = 0
        while col.size > 0:
            p = pivot_of_row[col[-1]]
            if p < 0:
                # combination built only for columns that become pivots
                comb = np.array([c], dtype=np.int64)
                for u in range(n_used):
                    comb = _symmetric_difference(comb, combination[used[u]])
                pivot_position[len(reduced)] = c
                pivot_of_row[col[-1]] = len(reduced)
                reduced.append(col)
                combination.append(comb)
                break
            used[n_used] = p
            n_used += 1
            col = _symmetric_difference(col, reduced[p])
    return pivot_of_row, pivot_position[: len(reduced)].copy(), reduced, combination


@njit(cache=True)
def _solve_on_reduction(
    pivot_of_row,
    pivot_position,
    reduced,
    combination,
    cutoff,
    extra_rows,
    b_indptr,
    b_indices,
    states,
):
    n_cached = len(reduced)
    local = pivot_of_row.copy()
    for r in range(local.size):
        if local[r] >= 0 and pivot_position[local[r]] >= cutoff:
            local[r] = -1
    ext_reduced = List.empty_list(_INT_ARRAY)
    ext_combination = List.empty_list(_INT_ARRAY)
    for k in range(extra_rows.size):
        col = np.array([extra_rows[k]], dtype=np.int64)
        comb = np.array([cutoff + k], dtype=np.int64)
        while col.size > 0:
            p = local[col[-1]]
            if p < 0:
                local[col[-1]] = n_cached + len(ext_reduced)
                ext_reduced.append(col)
                ext_combination.append(comb)
                break
            if p < n_cached:
                col = _symmetric_difference(col, reduced[p])
                comb = _symmetric_difference(comb, combination[p])
            else:
                col = _symmetric_difference(col, ext_reduced[p - n_cached])
                comb = _symmetric_difference(comb, ext_combination[p - n_cached])
    supports = List.empty_list(_INT_ARRAY)
    for j in range(b_indptr.size - 1):
        rhs = b_indices[b_indptr[j] : b_indptr[j + 1]].copy()
        x = np.empty(0, np.int64)
        states[j] = 0
        while rhs.size > 0:
            p = local[rhs[-1]]
            if p < 0:
                states[j] = -1
                break
            if p < n_cached:
                rhs = _symmetric_difference(rhs, reduced[p])
                x = _symmetric_difference(x, combination[p])
            else:
                rhs = _symmetric_difference(rhs, ext_reduced[p - n_cached])
                x = _symmetric_difference(x, ext_combination[p - n_cached])
        supports.append(x)
    return supports


def reduce_columns(column_rows: np.ndarray, nrow: int) -> tuple:
    """Reduce the columns of A (row indices per column, shape (ncol, k)) left to right once."""
    ncol, k = column_rows.shape
    indptr = np.arange(0, k * ncol + 1, k, dtype=np.int64)
    indices = np.ascontiguousarray(np.sort(column_rows, axis=1), dtype=np.int64).ravel()
    return _reduce_columns(indptr, indices, nrow, ncol)


def solve_on_reduction(reduction, cutoff, extra_rows, one_idx_b_list):
    """Solve A x = b where A = first cutoff reduced columns followed by one-entry columns at extra_rows.

    Returns (states, supports): supports[j] holds the column positions with x = 1, None if unsolved.
    """
    n_systems = len(one_idx_b_list)
    if n_systems == 0:
        return [], []
    b_list = [np.unique(np.asarray(b, dtype=np.int64)) for b in one_idx_b_list]
    b_indptr = np.zeros(n_systems + 1, dtype=np.int64)
    b_indptr[1:] = np.cumsum([b.size for b in b_list])
    b_indices = np.concatenate(b_list).astype(np.int64)
    states = np.zeros(n_systems, dtype=np.int64)
    supports = _solve_on_reduction(
        *reduction,
        int(cutoff),
        np.asarray(extra_rows, dtype=np.int64),
        b_indptr,
        b_indices,
        states,
    )
    state_list = states.tolist()
    return state_list, [
        supports[j] if state_list[j] == 0 else None for j in range(n_systems)
    ]
