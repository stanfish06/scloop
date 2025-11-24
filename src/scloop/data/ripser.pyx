cimport cython
from ripser cimport ripserResults, rips_dm_sparse, value_t
from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np
import typing
import dataclasses

@dataclasses.dataclass
class RipserResults:
    births_and_deaths_by_dim: list(np.ndarray)
    cocycles_by_dim: np.ndarray
    num_edges: int

cdef list convert_birth_death_to_numpy(vector[vector[value_t]] births_and_deaths_by_dim, int dim):
    '''
       The first variable is a vector of unrolled persistence diagrams
       so, for example births_and_deaths_by_dim[0] contains a list of
                [birth0, death0, birth1, death1, ..., birthk, deathk]
       for k points in the 0D persistence diagram
       and likewise for d-dimensional persistence in births_and_deaths_by_dim[d]
    '''
    cdef list birth = []
    cdef list death = []
    cdef vector[value_t]& birth_death = births_and_deaths_by_dim[dim]
    cdef int n_pairs = birth_death.size() // 2
    for i in range(n_pairs):
        birth.append(birth_death[i * 2])
        death.append(birth_death[i * 2 + 1])
    return [birth, death]

def ripser(
    coo_matrix distance_matrix,
    int modulus,
    int dim_max,
    float threshold,
    bool do_cocycles,
) -> RipserResults:
    pass
    cdef int* I = (int*) distance_matrix.row
    cdef int* J = (int*) distance_matrix.col
    cdef float* V = (float*) distance_matrix.data
    cdef int NEdges = distance_matrix.size
    cdef int N = distance_matrix.shape[0]
    cdef ripserResults res = rips_dm_sparse(I, J, V, NEdges, N, modulus, dim_max, threshold, (int)do_cocycles)
    cdef list persistence_diagrams = [convert_birth_death_to_numpy(ripserResults.births_and_deaths_by_dim, i) for i in range(dim_max)]
    return RipserResults(
        persistence_diagrams,
        np.array(),
        0
    )
