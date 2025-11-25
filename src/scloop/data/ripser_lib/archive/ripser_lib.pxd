from libcpp.vector cimport vector 
ctypedef float value_t

cdef extern from "ripser.hpp":
    cdef cppclass ripserResults:
        vector[vector[value_t]] births_and_deaths_by_dim
        vector[vector[vector[int]]] cocycles_by_dim
        int num_edges
    cdef ripserResults rips_dm_sparse(int* I, int* J, float* V, int NEdges, int N, int modulus, int dim_max, float threshold, int do_cocycles)
