# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
cimport cython
from cython.parallel import prange
from libcpp.vector cimport vector
from libcpp.functional cimport function
import numpy as np

ctypedef vector[double] double_vec_1d
ctypedef vector[vector[double]] double_vec_2d

cdef extern from "Frechet.h":
    ctypedef function[double(const double_vec_1d& point_a, const double_vec_1d& point_b)] distanceFunction
    # ISSUE: cant pass function into computeLoopFrechet due to noexcept issue, check back later
    cdef double computeLoopFrechet(const double_vec_2d& curve_a, const double_vec_2d& curve_b) nogil

cdef double_vec_1d list_to_vec_1d(list l):
    cdef double_vec_1d vec
    vec.reserve(len(l))
    for v in l:
        vec.push_back(v)
    return vec

cdef double_vec_2d list_to_vec_2d(list l):
    cdef size_t m = len(l)
    cdef size_t n = len(l[0])
    cdef double_vec_2d vec = double_vec_2d(m, double_vec_1d(n))
    cdef size_t i, j
    for i in range(m):
        for j in range(n):
            vec[i][j] = l[i][j]
    return vec

cdef double_vec_2d memview_to_vec_2d(double[:, ::1] arr) noexcept nogil:
    cdef size_t m = arr.shape[0]
    cdef size_t n = arr.shape[1]
    cdef double_vec_2d vec = double_vec_2d(m, double_vec_1d(n))
    cdef size_t i, j
    for i in range(m):
        for j in range(n):
            vec[i][j] = arr[i, j]
    return vec

cdef double compute_loop_frechet_nogil(
    double[:, ::1] curve_a,
    double[:, ::1] curve_b
) noexcept nogil:
    cdef double_vec_2d curve_a_vec = memview_to_vec_2d(curve_a)
    cdef double_vec_2d curve_b_vec = memview_to_vec_2d(curve_b)
    return computeLoopFrechet(curve_a_vec, curve_b_vec)

def compute_loop_set_frechet(
    curve_pairs
):
    cdef int n = len(curve_pairs)
    cdef double[:] results = np.empty(n, dtype=np.float64)
    cdef int i

    for i in prange(n, nogil=True):
        with gil:
            # need gil to get data from list
            a, b = curve_pairs[i]
        results[i] = compute_loop_frechet_nogil(a, b)

    return np.asarray(results) 

# Backward-compatible wrapper for Python lists
def compute_loop_frechet(
    curve_a,
    curve_b,
    distance_type: str = "euclidean"
) -> float:
    if distance_type != "euclidean":
        raise ValueError(f"Unsupported distance type: {distance_type}")

    cdef double[:, ::1] curve_a_view
    cdef double[:, ::1] curve_b_view
    cdef double_vec_2d curve_a_vec
    cdef double_vec_2d curve_b_vec
    cdef double result

    try:
        curve_a_view = curve_a
        curve_b_view = curve_b
        with nogil:
            result = compute_loop_frechet_nogil(curve_a_view, curve_b_view)
        return result
    except (TypeError, ValueError):
        curve_a_vec = list_to_vec_2d(curve_a)
        curve_b_vec = list_to_vec_2d(curve_b)
        return computeLoopFrechet(curve_a_vec, curve_b_vec)
