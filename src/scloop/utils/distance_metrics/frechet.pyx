# Copyright 2025 Zhiyuan Yu (Heemskerk's lab, University of Michigan)
cimport cython
from libcpp.vector cimport vector 
from libcpp.functional cimport function

ctypedef vector[double] double_vec_1d
ctypedef vector[vector[double]] double_vec_2d

cdef extern from "Frechet.h":
    ctypedef function[double(const double_vec_1d& point_a, const double_vec_1d& point_b)] distanceFunction
    # ISSUE: cant pass function into computeLoopFrechet due to noexcept issue, check back later
    cdef double computeLoopFrechet(const double_vec_2d& curve_a, const double_vec_2d& curve_b)

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

def compute_loop_frechet(
    curve_a: list,
    curve_b: list,
    distance_type: str = "euclidean"
) -> float:
    cdef double_vec_2d curve_a_vec = list_to_vec_2d(curve_a)
    cdef double_vec_2d curve_b_vec = list_to_vec_2d(curve_b)
    if distance_type == "euclidean":
        return computeLoopFrechet(curve_a_vec, curve_b_vec)
    else:
        raise ValueError(f"Unsupported distance type: {distance_type}")

# TODO: Frechet distance between two sets of loops
