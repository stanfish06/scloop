# Copyright 2026 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
cimport cython
from libc.stdlib cimport malloc, free
from libc.stdio cimport fprintf, stderr
import numpy as np
from cython.parallel import prange
from openmp cimport omp_get_max_threads, omp_get_thread_num

cdef inline void increment(int *arr, int idx) noexcept nogil:
    arr[idx] += 1

cdef inline int sum_counts(int *arr, int n) noexcept nogil:
    cdef int i, s = 0
    for i in range(n):
        s += arr[i]
    return s

cdef extern from "Sanity/src/calc_true_variation_parallel_prior_mu_sigma.h":
    void get_gene_expression_level(double *n_c, double *N_c, double n, double vmin, double vmax, double &mu, double &var_mu, double *delta, double *var_delta, int C, int numbin, double a, double b, double *lik, double &v_ml, double &mu_v_ml, double &var_mu_v_ml, double *delta_v_ml, double *var_delta_v_ml) nogil


def run_sanity(double[:, ::1] X, double[::1] library_size, double[::1] gene_total, int nbins, double vmin, double vmax, bint verbose=True, bint use_max_v=False):
    cdef int G = X.shape[0]
    cdef int C = X.shape[1]
    cdef double a = 1.0
    cdef double b = 0.0

    cdef double[:, ::1] log_mean = np.zeros((G, C), dtype=np.float64)
    cdef double[:, ::1] log_var = np.zeros((G, C), dtype=np.float64)

    cdef int g, c, t, tid, total
    cdef int n_threads = omp_get_max_threads()
    cdef int *thread_counts = <int *>malloc(n_threads * sizeof(int))
    cdef double mu_g, var_mu_g, v_ml, mu_v_ml, var_mu_v_ml
    cdef double *delta
    cdef double *var_delta
    cdef double *lik
    cdef double *delta_v_ml
    cdef double *var_delta_v_ml

    for t in range(n_threads):
        thread_counts[t] = 0

    for g in prange(G, nogil=True):
        tid = omp_get_thread_num()
        delta = <double *>malloc(C * sizeof(double))
        var_delta = <double *>malloc(C * sizeof(double))
        lik = <double *>malloc(nbins * sizeof(double))
        delta_v_ml = <double *>malloc(C * sizeof(double))
        var_delta_v_ml = <double *>malloc(C * sizeof(double))

        get_gene_expression_level(
            &X[g, 0], &library_size[0], gene_total[g],
            vmin, vmax,
            mu_g, var_mu_g,
            delta, var_delta,
            C, nbins, a, b, lik,
            v_ml, mu_v_ml, var_mu_v_ml,
            delta_v_ml, var_delta_v_ml,
        )

        for c in range(C):
          if use_max_v:
              log_mean[g, c] = mu_v_ml + delta_v_ml[c]
              log_var[g, c] = var_mu_v_ml + var_delta_v_ml[c]
          else:
              log_mean[g, c] = mu_g + delta[c]
              log_var[g, c] = var_mu_g + var_delta[c]

        free(delta)
        free(var_delta)
        free(lik)
        free(delta_v_ml)
        free(var_delta_v_ml)

        if verbose:
            increment(thread_counts, tid)
            if tid == 0 and thread_counts[0] % 10 == 0:
                fprintf(stderr, "Sanity: %d/%d genes processed\n", sum_counts(thread_counts, n_threads), G)

    free(thread_counts)

    return np.asarray(log_mean), np.asarray(log_var)
