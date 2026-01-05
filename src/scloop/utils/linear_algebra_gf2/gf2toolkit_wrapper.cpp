// Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
#include "gf2toolkit_wrapper.hpp"
#include "GF2toolkit/srcs/GF2toolkit_LU.hh"
#include "GF2toolkit/srcs/GF2toolkit_Matrix.hh"
#include "GF2toolkit/srcs/GF2toolkit_m4ri.hh"

#include <cstring>
#include <m4ri/m4ri.h>
#include <vector>

using UNSIGNED = uint64_t;
static constexpr unsigned NUM_BITS = 64;
static constexpr unsigned BLOCK_BITS = 256;

static unsigned round_up_to_block(unsigned n) {
  return ((n + BLOCK_BITS - 1) / BLOCK_BITS) * BLOCK_BITS;
}

extern "C" {

int gf2toolkit_solve(const int32_t *row_indices, const int32_t *col_indices,
                     int32_t nnz_A, int32_t nrow_A, int32_t ncol_A,
                     const int32_t *b_indices, int32_t nnz_b,
                     int32_t *solution_out, int32_t check_rank) {
  if (nrow_A < ncol_A) {
    return -1;
  }

  unsigned nc_padded = round_up_to_block(static_cast<unsigned>(ncol_A));
  unsigned nr = static_cast<unsigned>(nrow_A);

  GF2toolkit::LU<UNSIGNED> lu;
  lu.resize(nr, nc_padded);

  for (int32_t i = 0; i < nnz_A; i++) {
    if (row_indices[i] < nrow_A && col_indices[i] < ncol_A) {
      lu.set(static_cast<unsigned>(row_indices[i]),
             static_cast<unsigned>(col_indices[i]));
    }
  }

  unsigned rank = lu.computeRank();

  if (check_rank && rank < static_cast<unsigned>(ncol_A)) {
    return -1;
  }

  unsigned nrL, ncL, nrU, ncU;
  unsigned dimRowsL = nr;
  unsigned dimRowsU = nr;
  unsigned numBlocksL = nc_padded / NUM_BITS;
  unsigned numBlocksU = numBlocksL;

  UNSIGNED *L_data = new UNSIGNED[dimRowsL * numBlocksL];
  UNSIGNED *U_data = new UNSIGNED[dimRowsU * numBlocksU];
  memset(L_data, 0, dimRowsL * numBlocksL * sizeof(UNSIGNED));
  memset(U_data, 0, dimRowsU * numBlocksU * sizeof(UNSIGNED));

  lu.extractL(L_data, dimRowsL, numBlocksL, nrL, ncL);
  lu.extractU(U_data, dimRowsU, numBlocksU, nrU, ncU);

  if (ncL > rank)
    ncL = rank;
  if (ncU > static_cast<unsigned>(ncol_A))
    ncU = ncol_A;

  unsigned numBlocksL_actual = (ncL + NUM_BITS - 1) / NUM_BITS;
  unsigned numBlocksU_actual = (ncU + NUM_BITS - 1) / NUM_BITS;

  mzd_t *L_full =
      GF2toolkit::toM4RI<UNSIGNED>(L_data, dimRowsL, nrL, numBlocksL_actual);
  mzd_t *U_full =
      GF2toolkit::toM4RI<UNSIGNED>(U_data, dimRowsU, nrU, numBlocksU_actual);

  mzd_t *L = mzd_init(rank, rank);
  mzd_t *U = mzd_init(nrU, ncU);

  for (unsigned i = 0; i < rank; i++) {
    for (unsigned j = 0; j < rank; j++) {
      mzd_write_bit(L, i, j, mzd_read_bit(L_full, i, j));
    }
  }

  for (unsigned i = 0; i < nrU; i++) {
    for (unsigned j = 0; j < ncU; j++) {
      mzd_write_bit(U, i, j, mzd_read_bit(U_full, i, j));
    }
  }

  mzd_free(L_full);
  mzd_free(U_full);
  delete[] L_data;
  delete[] U_data;

  mzd_t *b_m4ri = mzd_init(nrow_A, 1);
  for (int32_t i = 0; i < nnz_b; i++) {
    if (b_indices[i] >= 0 && b_indices[i] < nrow_A) {
      mzd_write_bit(b_m4ri, b_indices[i], 0, 1);
    }
  }

  unsigned b_nc_blocks = nc_padded / NUM_BITS;
  UNSIGNED *b_gf2 = new UNSIGNED[nr * b_nc_blocks];
  memset(b_gf2, 0, nr * b_nc_blocks * sizeof(UNSIGNED));

  for (unsigned i = 0; i < nr; i++) {
    if (mzd_read_bit(b_m4ri, i, 0)) {
      b_gf2[0 * nr + i] |= (1ULL << 0);
    }
  }

  // this is important, dont forget to permutate b
  lu.applyPermute(b_gf2, nr, b_nc_blocks);

  mzd_t *Pb_m4ri = GF2toolkit::toM4RI<UNSIGNED>(b_gf2, nr, nr, 1);
  delete[] b_gf2;
  mzd_free(b_m4ri);

  mzd_t *Pb = mzd_init(rank, 1);
  for (unsigned i = 0; i < rank; i++) {
    mzd_write_bit(Pb, i, 0, mzd_read_bit(Pb_m4ri, i, 0));
  }

  // If all extra elements of b have 1s, then not solvable
  for (int32_t i = rank; i < nrow_A; i++) {
    if (mzd_read_bit(Pb_m4ri, i, 0) != 0) {
      mzd_free(Pb_m4ri);
      mzd_free(Pb);
      mzd_free(L);
      mzd_free(U);
      return -1;
    }
  }
  mzd_free(Pb_m4ri);

  mzd_trsm_lower_left(L, Pb, 0);

  mzd_t *U_square = mzd_init(rank, rank);
  for (unsigned i = 0; i < rank; i++) {
    for (unsigned j = 0; j < rank && j < ncU; j++) {
      mzd_write_bit(U_square, i, j, mzd_read_bit(U, i, j));
    }
  }

  mzd_trsm_upper_left(U_square, Pb, 0);

  mzd_t *x = mzd_init(ncol_A, 1);
  for (unsigned i = 0; i < rank; i++) {
    mzd_write_bit(x, i, 0, mzd_read_bit(Pb, i, 0));
  }

  mzd_t *A_verify = mzd_init(nrow_A, ncol_A);
  for (int32_t i = 0; i < nnz_A; i++) {
    if (row_indices[i] < nrow_A && col_indices[i] < ncol_A) {
      int bit = mzd_read_bit(A_verify, row_indices[i], col_indices[i]);
      mzd_write_bit(A_verify, row_indices[i], col_indices[i], bit ^ 1);
    }
  }
  
  mzd_t *b_verify = mzd_init(nrow_A, 1);
  for (int32_t i = 0; i < nnz_b; i++) {
    if (b_indices[i] >= 0 && b_indices[i] < nrow_A) {
      int bit = mzd_read_bit(b_verify, b_indices[i], 0);
      mzd_write_bit(b_verify, b_indices[i], 0, bit ^ 1);
    }
  }
  
  mzd_t *Ax = mzd_init(nrow_A, 1);
  mzd_mul(Ax, A_verify, x, 0);
  
  bool solution_valid = true;
  for (int32_t i = 0; i < nrow_A; i++) {
    if (mzd_read_bit(Ax, i, 0) != mzd_read_bit(b_verify, i, 0)) {
      solution_valid = false;
      break;
    }
  }
  
  mzd_free(A_verify);
  mzd_free(b_verify);
  mzd_free(Ax);
  
  if (!solution_valid) {
    mzd_free(L);
    mzd_free(U);
    mzd_free(U_square);
    mzd_free(Pb);
    mzd_free(x);
    return -1;
  }

  for (int32_t i = 0; i < ncol_A; i++) {
    solution_out[i] = mzd_read_bit(x, i, 0);
  }

  mzd_free(L);
  mzd_free(U);
  mzd_free(U_square);
  mzd_free(Pb);
  mzd_free(x);

  return 0;
}

int gf2toolkit_solve_multiple(
    const int32_t *row_indices, const int32_t *col_indices, int32_t nnz_A,
    int32_t nrow_A, int32_t ncol_A, const int32_t *b_indices_flat,
    const int32_t *b_lengths, const int32_t *b_offsets, int32_t n_systems,
    int32_t *results_out, int32_t *solutions_out, int32_t check_rank) {
  if (nrow_A < ncol_A) {
    for (int32_t sys = 0; sys < n_systems; sys++) {
      results_out[sys] = -1;
    }
    return -1;
  }

  unsigned nc_padded = round_up_to_block(static_cast<unsigned>(ncol_A));
  unsigned nr = static_cast<unsigned>(nrow_A);

  GF2toolkit::LU<UNSIGNED> lu;
  lu.resize(nr, nc_padded);

  for (int32_t i = 0; i < nnz_A; i++) {
    if (row_indices[i] < nrow_A && col_indices[i] < ncol_A) {
      lu.set(static_cast<unsigned>(row_indices[i]),
             static_cast<unsigned>(col_indices[i]));
    }
  }

  unsigned rank = lu.computeRank();

  if (check_rank && rank < static_cast<unsigned>(ncol_A)) {
    for (int32_t sys = 0; sys < n_systems; sys++) {
      results_out[sys] = -1;
    }
    return -1;
  }

  unsigned nrL, ncL, nrU, ncU;
  unsigned dimRowsL = nr;
  unsigned dimRowsU = nr;
  unsigned numBlocksL = nc_padded / NUM_BITS;
  unsigned numBlocksU = numBlocksL;

  UNSIGNED *L_data = new UNSIGNED[dimRowsL * numBlocksL];
  UNSIGNED *U_data = new UNSIGNED[dimRowsU * numBlocksU];
  memset(L_data, 0, dimRowsL * numBlocksL * sizeof(UNSIGNED));
  memset(U_data, 0, dimRowsU * numBlocksU * sizeof(UNSIGNED));

  lu.extractL(L_data, dimRowsL, numBlocksL, nrL, ncL);
  lu.extractU(U_data, dimRowsU, numBlocksU, nrU, ncU);

  if (ncL > rank)
    ncL = rank;
  if (ncU > static_cast<unsigned>(ncol_A))
    ncU = ncol_A;

  unsigned numBlocksL_actual = (ncL + NUM_BITS - 1) / NUM_BITS;
  unsigned numBlocksU_actual = (ncU + NUM_BITS - 1) / NUM_BITS;

  mzd_t *L_full =
      GF2toolkit::toM4RI<UNSIGNED>(L_data, dimRowsL, nrL, numBlocksL_actual);
  mzd_t *U_full =
      GF2toolkit::toM4RI<UNSIGNED>(U_data, dimRowsU, nrU, numBlocksU_actual);

  mzd_t *L = mzd_init(rank, rank);
  mzd_t *U_square = mzd_init(rank, rank);

  for (unsigned i = 0; i < rank; i++) {
    for (unsigned j = 0; j < rank; j++) {
      mzd_write_bit(L, i, j, mzd_read_bit(L_full, i, j));
      if (j < ncU) {
        mzd_write_bit(U_square, i, j, mzd_read_bit(U_full, i, j));
      }
    }
  }

  mzd_free(L_full);
  mzd_free(U_full);
  delete[] L_data;
  delete[] U_data;

  // Solve each system
  unsigned b_nc_blocks = nc_padded / NUM_BITS;
  UNSIGNED *b_gf2 = new UNSIGNED[nr * b_nc_blocks];

  for (int32_t sys = 0; sys < n_systems; sys++) {
    int32_t offset = b_offsets[sys];
    int32_t length = b_lengths[sys];

    mzd_t *b_m4ri = mzd_init(nrow_A, 1);
    for (int32_t i = 0; i < length; i++) {
      int32_t idx = b_indices_flat[offset + i];
      if (idx >= 0 && idx < nrow_A) {
        mzd_write_bit(b_m4ri, idx, 0, 1);
      }
    }

    memset(b_gf2, 0, nr * b_nc_blocks * sizeof(UNSIGNED));
    for (unsigned i = 0; i < nr; i++) {
      if (mzd_read_bit(b_m4ri, i, 0)) {
        b_gf2[0 * nr + i] |= (1ULL << 0);
      }
    }

    lu.applyPermute(b_gf2, nr, b_nc_blocks);

    mzd_t *Pb_m4ri = GF2toolkit::toM4RI<UNSIGNED>(b_gf2, nr, nr, 1);
    mzd_free(b_m4ri);

    bool solvable = true;
    for (int32_t i = rank; i < nrow_A; i++) {
      if (mzd_read_bit(Pb_m4ri, i, 0) != 0) {
        solvable = false;
        break;
      }
    }

    if (!solvable) {
      results_out[sys] = -1;
      for (int32_t i = 0; i < ncol_A; i++) {
        solutions_out[sys * ncol_A + i] = 0;
      }
      mzd_free(Pb_m4ri);
      continue;
    }

    mzd_t *Pb = mzd_init(rank, 1);
    for (unsigned i = 0; i < rank; i++) {
      mzd_write_bit(Pb, i, 0, mzd_read_bit(Pb_m4ri, i, 0));
    }
    mzd_free(Pb_m4ri);

    mzd_trsm_lower_left(L, Pb, 0);
    mzd_trsm_upper_left(U_square, Pb, 0);

    results_out[sys] = 0;
    for (unsigned i = 0; i < rank; i++) {
      solutions_out[sys * ncol_A + i] = mzd_read_bit(Pb, i, 0);
    }
    for (int32_t i = rank; i < ncol_A; i++) {
      solutions_out[sys * ncol_A + i] = 0;
    }

    mzd_free(Pb);
  }

  delete[] b_gf2;
  mzd_free(L);
  mzd_free(U_square);
  return 0;
}
}
