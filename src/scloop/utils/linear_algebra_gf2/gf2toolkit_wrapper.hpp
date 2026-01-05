// Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
#ifndef GF2TOOLKIT_WRAPPER_HPP
#define GF2TOOLKIT_WRAPPER_HPP

#include <cstdint>

extern "C" {

int gf2toolkit_solve(const int32_t *row_indices, const int32_t *col_indices,
                     int32_t nnz_A, int32_t nrow_A, int32_t ncol_A,
                     const int32_t *b_indices, int32_t nnz_b,
                     int32_t *solution_out, int32_t check_rank);

int gf2toolkit_solve_multiple(
    const int32_t *row_indices, const int32_t *col_indices, int32_t nnz_A,
    int32_t nrow_A, int32_t ncol_A, const int32_t *b_indices_flat,
    const int32_t *b_lengths, const int32_t *b_offsets, int32_t n_systems,
    int32_t *results_out, int32_t *solutions_out, int32_t check_rank);
}

#endif
