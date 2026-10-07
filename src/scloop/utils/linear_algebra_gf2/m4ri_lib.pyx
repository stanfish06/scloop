# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
cimport cython
from libc.stdlib cimport malloc, free

cdef extern from "m4ri/m4ri.h":
    ctypedef int rci_t
    ctypedef int wi_t
    ctypedef unsigned long long m4ri_word "word"
    ctypedef int BIT
    ctypedef struct mzd_t:
        rci_t nrows
        rci_t ncols
        wi_t width

    mzd_t *mzd_init(rci_t, rci_t) nogil
    void mzd_free(mzd_t *) nogil
    mzd_t *mzd_copy(mzd_t *DST, const mzd_t *A) nogil
    void mzd_write_bit(mzd_t *m, rci_t row, rci_t col, BIT value) nogil
    BIT mzd_read_bit(mzd_t *M, rci_t row, rci_t col) nogil

    void mzd_print(mzd_t *) nogil
    int mzd_solve_left(mzd_t *A, mzd_t *B, int cutoff, int inconsistency_check) nogil

cdef int solve_gf2_nogil(
    const rci_t[:] one_ridx_A,
    const rci_t[:] one_cidx_A,
    rci_t nrow_A,
    rci_t ncol_A,
    const rci_t[:] one_idx_b,
    BIT* solution_out
) noexcept nogil:
    cdef mzd_t *A = mzd_init(nrow_A, ncol_A)
    cdef mzd_t *b = mzd_init(nrow_A, 1)
    cdef size_t i, nnz_A = one_ridx_A.shape[0], nnz_b = one_idx_b.shape[0]
    cdef int result

    for i in range(nnz_A):
        mzd_write_bit(A, one_ridx_A[i], one_cidx_A[i], 1)

    for i in range(nnz_b):
        mzd_write_bit(b, one_idx_b[i], 0, 1)

    result = mzd_solve_left(A, b, 0, 1)

    if result == 0:
        for i in range(<size_t>ncol_A):
            solution_out[i] = mzd_read_bit(b, i, 0)

    mzd_free(A)
    mzd_free(b)
    return result


def solve_gf2(one_ridx_A, one_cidx_A, nrow_A, ncol_A, one_idx_b):
    assert nrow_A >= ncol_A, "number of rows must be greater than or equal to the number of columns"

    cdef rci_t[:] ridx_view
    cdef rci_t[:] cidx_view
    cdef rci_t[:] b_idx_view
    cdef BIT* solution
    cdef int result
    cdef list sol_list
    cdef mzd_t *A
    cdef mzd_t *b
    cdef rci_t nrow_c
    cdef rci_t ncol_c

    try:
        import numpy as np
        ridx_view = np.asarray(one_ridx_A, dtype=np.int32)
        cidx_view = np.asarray(one_cidx_A, dtype=np.int32)
        b_idx_view = np.asarray(one_idx_b, dtype=np.int32)
        nrow_c = nrow_A
        ncol_c = ncol_A
        solution = <BIT*>malloc(ncol_A * sizeof(BIT))

        if solution == NULL:
            raise MemoryError("Failed to allocate solution array")
        try:
            with nogil:
                result = solve_gf2_nogil(ridx_view, cidx_view, nrow_c, ncol_c, b_idx_view, solution)
            if result == 0:
                sol_list = [solution[i] for i in range(ncol_A)]
            else:
                sol_list = None
            return (result == 0, sol_list)
        finally:
            free(solution)
    except:
        A = mzd_init(nrow_A, ncol_A)
        b = mzd_init(nrow_A, 1)
        for (i, j) in zip(one_ridx_A, one_cidx_A):
            mzd_write_bit(A, i, j, 1)
        for i in one_idx_b:
            mzd_write_bit(b, i, 0, 1)
        try:
            result = mzd_solve_left(A, b, 0, 1)
            if result == 0:
                sol_list = [mzd_read_bit(b, i, 0) for i in range(ncol_A)]
            else:
                sol_list = None
            return (result == 0, sol_list)
        finally:
            mzd_free(A)
            mzd_free(b)


def solve_multiple_gf2(one_ridx_A, one_cidx_A, nrow_A, ncol_A, one_idx_b_list):
    """Solve A x = b over GF2 for every b in one_idx_b_list with one PLUQ of A.

    Returns (states, solutions): state 0 with x as a list of bits, or -1 with
    None when A x = b has no solution.
    """
    import numpy as np

    cdef mzd_t *A
    cdef mzd_t *B
    cdef Py_ssize_t i, j
    cdef rci_t m_c, ncol_c, k_c
    cdef rci_t[::1] a_r, a_c, b_r, b_c
    cdef unsigned char[:, ::1] x_view

    n_systems = len(one_idx_b_list)
    if n_systems == 0:
        return [], []
    if ncol_A == 0:
        states = [0 if len(b) == 0 else -1 for b in one_idx_b_list]
        return states, [[] if s == 0 else None for s in states]

    # mzd_write_bit sets a bit, so repeated entries count once
    entries = np.unique(
        np.asarray(one_ridx_A, dtype=np.int64) * ncol_A
        + np.asarray(one_cidx_A, dtype=np.int64)
    )
    ridx, cidx = entries // ncol_A, entries % ncol_A
    b_list = [np.unique(np.asarray(b, dtype=np.int64)) for b in one_idx_b_list]

    # drop rows that are zero in A and in every b; pad with zero rows to >= ncol_A
    used = np.unique(np.concatenate([ridx, *b_list]))
    m = max(used.size, ncol_A)
    ridx = np.searchsorted(used, ridx)
    b_list = [np.searchsorted(used, b) for b in b_list]

    a_r = ridx.astype(np.intc)
    a_c = cidx.astype(np.intc)
    b_r = np.concatenate(b_list).astype(np.intc)
    b_c = np.repeat(np.arange(n_systems), [b.size for b in b_list]).astype(np.intc)
    x = np.zeros((n_systems, ncol_A), dtype=np.uint8)
    x_view = x
    m_c, ncol_c, k_c = m, ncol_A, n_systems

    with nogil:
        A = mzd_init(m_c, ncol_c)
        B = mzd_init(m_c, k_c)
        for i in range(a_r.shape[0]):
            mzd_write_bit(A, a_r[i], a_c[i], 1)
        for i in range(b_r.shape[0]):
            mzd_write_bit(B, b_r[i], b_c[i], 1)
        # m4ri's inconsistency check is all-or-nothing across columns of B;
        # per-system consistency is checked below
        mzd_solve_left(A, B, 0, 0)
        for j in range(k_c):
            for i in range(ncol_c):
                x_view[j, i] = mzd_read_bit(B, i, j)
        mzd_free(A)
        mzd_free(B)

    states, sols = [], []
    for j in range(n_systems):
        # exact residual: A x + b == 0 over GF2
        residual = np.bincount(ridx[x[j, cidx] == 1], minlength=m) & 1
        residual[b_list[j]] ^= 1
        if residual.any():
            states.append(-1)
            sols.append(None)
        else:
            states.append(0)
            sols.append(x[j].tolist())
    return states, sols
