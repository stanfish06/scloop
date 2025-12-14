# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
cdef extern from "m4ri/m4ri.h":
    ctypedef int rci_t
    ctypedef int wi_t
    ctypedef unsigned long long m4ri_word "word"
    ctypedef int BIT
    ctypedef struct mzd_t:
        rci_t nrows
        rci_t ncols
        wi_t width

    mzd_t *mzd_init(rci_t, rci_t)
    void mzd_free(mzd_t *)
    mzd_t *mzd_copy(mzd_t *DST, const mzd_t *A)
    void mzd_write_bit(mzd_t *m, rci_t row, rci_t col, BIT value)
    BIT mzd_read_bit(mzd_t *M, rci_t row, rci_t col)

    void mzd_print(mzd_t *)
    int mzd_solve_left(mzd_t *A, mzd_t *B, int cutoff, int inconsistency_check)


def solve_gf2(one_ridx_A, one_cidx_A, nrow_A, ncol_A, one_idx_b):
    assert nrow_A >= ncol_A, "number of rows must be greater than or equal to the number of columns" 
    cdef mzd_t *A = mzd_init(nrow_A, ncol_A);
    cdef mzd_t *b = mzd_init(nrow_A, 1);
    for (i, j) in zip(one_ridx_A, one_cidx_A):
        mzd_write_bit(A, i, j, 1)
    for i in one_idx_b:
        mzd_write_bit(b, i, 0, 1)
    try:
        result = mzd_solve_left(A, b, 0, 1)
        if result == 0:
            sol = [mzd_read_bit(b, i, 0) for i in range(ncol_A)]
        else:
            sol = None
        return (result == 0, sol)
    finally:
        mzd_free(A)
        mzd_free(b)

def solve_multiple_gf2(one_ridx_A, one_cidx_A, nrow_A, ncol_A, one_idx_b_list):
    cdef mzd_t *A_base = mzd_init(nrow_A, ncol_A)
    for (i, j) in zip(one_ridx_A, one_cidx_A):
        mzd_write_bit(A_base, i, j, 1)

    results = []
    sols = []

    try:
        for one_idx_b in one_idx_b_list:
            A = mzd_init(nrow_A, ncol_A)
            b = mzd_init(nrow_A, 1)

            try:
                mzd_copy(A, A_base)
                for i in one_idx_b:
                    mzd_write_bit(b, i, 0, 1)

                result = mzd_solve_left(A, b, 0, 1)
                if result == 0:
                    sol = [mzd_read_bit(b, i, 0) for i in range(ncol_A)]
                else:
                    sol = None
                results.append(result)
                sols.append(sol)
            finally:
                mzd_free(A)
                mzd_free(b)
    finally:
        mzd_free(A_base)

    return results, sols
