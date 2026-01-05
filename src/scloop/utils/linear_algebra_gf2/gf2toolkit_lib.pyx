# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdint cimport int32_t

cdef extern from "gf2toolkit_wrapper.hpp":
    int gf2toolkit_solve(
        const int32_t* row_indices,
        const int32_t* col_indices,
        int32_t nnz_A,
        int32_t nrow_A,
        int32_t ncol_A,
        const int32_t* b_indices,
        int32_t nnz_b,
        int32_t* solution_out,
        int32_t check_rank
    ) nogil

    int gf2toolkit_solve_multiple(
        const int32_t* row_indices,
        const int32_t* col_indices,
        int32_t nnz_A,
        int32_t nrow_A,
        int32_t ncol_A,
        const int32_t* b_indices_flat,
        const int32_t* b_lengths,
        const int32_t* b_offsets,
        int32_t n_systems,
        int32_t* results_out,
        int32_t* solutions_out,
        int32_t check_rank
    ) nogil


# ~2x slower than m4ri as of now (probably need native trig-solver)
def solve_gf2(one_ridx_A, one_cidx_A, nrow_A, ncol_A, one_idx_b):
    assert nrow_A >= ncol_A, "number of rows must be greater than or equal to the number of columns"

    cdef int32_t[:] ridx_view
    cdef int32_t[:] cidx_view
    cdef int32_t[:] b_idx_view
    cdef int32_t* solution
    cdef int result
    cdef list sol_list
    cdef int32_t nrow_c
    cdef int32_t ncol_c
    cdef int32_t nnz_A
    cdef int32_t nnz_b

    import numpy as np
    ridx_view = np.asarray(one_ridx_A, dtype=np.int32)
    cidx_view = np.asarray(one_cidx_A, dtype=np.int32)
    b_idx_view = np.asarray(one_idx_b, dtype=np.int32)
    nrow_c = nrow_A
    ncol_c = ncol_A
    nnz_A = len(ridx_view)
    nnz_b = len(b_idx_view)
    solution = <int32_t*>malloc(ncol_A * sizeof(int32_t))

    if solution == NULL:
        raise MemoryError("Failed to allocate solution array")

    try:
        with nogil:
            # rank deficiency is fine, in our case, we just need to know if a system is solvable or not
            result = gf2toolkit_solve(
                &ridx_view[0] if nnz_A > 0 else NULL,
                &cidx_view[0] if nnz_A > 0 else NULL,
                nnz_A,
                nrow_c, ncol_c,
                &b_idx_view[0] if nnz_b > 0 else NULL,
                nnz_b,
                solution, 0
            )

        if result == 0:
            sol_list = [solution[i] for i in range(ncol_A)]
        else:
            sol_list = None
        return (result == 0, sol_list)
    finally:
        free(solution)

# has bug
def solve_multiple_gf2(one_ridx_A, one_cidx_A, nrow_A, ncol_A, one_idx_b_list):
    cdef int32_t[:] ridx_view
    cdef int32_t[:] cidx_view
    cdef int32_t[:] b_flat_view
    cdef int32_t[:] b_lengths_view
    cdef int32_t[:] b_offsets_view
    cdef int32_t* results_arr
    cdef int32_t* solutions_arr
    cdef int32_t nrow_c
    cdef int32_t ncol_c
    cdef int32_t nnz_A
    cdef int32_t n_systems
    cdef size_t i, j

    import numpy as np

    ridx_view = np.asarray(one_ridx_A, dtype=np.int32)
    cidx_view = np.asarray(one_cidx_A, dtype=np.int32)
    nrow_c = nrow_A
    ncol_c = ncol_A
    nnz_A = len(ridx_view)
    n_systems = len(one_idx_b_list)

    b_flat = []
    b_lengths = []
    b_offsets = []
    offset = 0
    for b in one_idx_b_list:
        b_arr = np.asarray(b, dtype=np.int32)
        b_flat.extend(b_arr.tolist())
        b_lengths.append(len(b_arr))
        b_offsets.append(offset)
        offset += len(b_arr)

    b_flat_view = np.asarray(b_flat, dtype=np.int32)
    b_lengths_view = np.asarray(b_lengths, dtype=np.int32)
    b_offsets_view = np.asarray(b_offsets, dtype=np.int32)

    results_arr = <int32_t*>malloc(n_systems * sizeof(int32_t))
    solutions_arr = <int32_t*>malloc(n_systems * ncol_A * sizeof(int32_t))

    if results_arr == NULL or solutions_arr == NULL:
        if results_arr != NULL:
            free(results_arr)
        if solutions_arr != NULL:
            free(solutions_arr)
        raise MemoryError("Failed to allocate arrays")

    try:
        with nogil:
            gf2toolkit_solve_multiple(
                &ridx_view[0] if nnz_A > 0 else NULL,
                &cidx_view[0] if nnz_A > 0 else NULL,
                nnz_A,
                nrow_c, ncol_c,
                &b_flat_view[0],
                &b_lengths_view[0],
                &b_offsets_view[0],
                n_systems,
                results_arr, solutions_arr, 0
            )

        results = []
        sols = []
        for i in range(n_systems):
            results.append(results_arr[i])
            if results_arr[i] == 0:
                sol = [solutions_arr[i * ncol_A + j] for j in range(ncol_A)]
            else:
                sol = None
            sols.append(sol)

        return results, sols
    finally:
        free(results_arr)
        free(solutions_arr)
