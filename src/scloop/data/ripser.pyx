cimport cython
from ripser cimport ripserResults, rips_dm_sparse, value_t
from scipy.sparse import coo_matrix
import numpy as np
cimport numpy as np
import typing
import dataclasses

@dataclasses.dataclass
class RipserResults:
    births_and_deaths_by_dim: list[np.ndarray]
    cocycles_by_dim: list
    num_edges: int

cdef list converting_cocycles_to_numpy(vector[vector[vector[int]]] cocycles_by_dim, int dim): 
    '''
    This is a vector of representative cocycles for each
    dimension. For now, only cocycles above dimension 0 are added, so
    dimension 0 is an empty list For the others, cocycles_by_dim[d] holds an
    array of representative cocycles for dimension d which is parallel with
    the array of births/deaths for dimension d. Each element of the array is
    itself an array of unrolled information about the cocycle For dimension 1,
    for example, the zeroeth element of the array contains [ccl0_simplex0_idx0
    ccl0_simplex0_idx1 ccl0_simplex0_val, ccl0_simplex1_idx0
    ccl0_simplex1_idx1 ccl0_simplex1_val, ... ccl0_simplexk_idx0
    ccl0_simplexk_idx1 ccl0_simplexk_val] for a cocycle representing the first
    persistence point, which has k simplices with nonzero values in the
    representative cocycle
    '''
    cdef list cocycle_representatives = []
    cdef vector[vector[int]]& cocycles = cocycles_by_dim[dim]
    cdef Py_ssize_t nc = (Py_ssize_t) cocycles.size() 
    for i in range(nc):
        cdef vector[int]& rep_i = cocycles[i]
        cdef int chunk_size = dim + 2
        cdef Py_ssize_t n_simplices = rep_i.size() // chunk_size
        cdef list cocycle_rep_members = []
        for j in range(n_simplices):
            cdef list simplex = []
            cdef int start_idx = j * chunk_size
            cdef int end_idx = start_idx + chunk_size - 1
            for k in range(start_idx, end_idx):
                simplex.append(int(rep_i[k]))
            cocycle_rep_members.append([simplex, int(rep_i[end_idx])])
        cocycle_representatives.append(cocycle_rep_members)
    return cocycle_representatives

cdef list converting_birth_death_to_numpy(vector[vector[value_t]] births_and_deaths_by_dim, int dim):
    '''
    This is a vector of unrolled persistence diagrams
    so, for example births_and_deaths_by_dim[0] contains a list of
    [birth0, death0, birth1, death1, ..., birthk, deathk]
    for k points in the 0D persistence diagram
    and likewise for d-dimensional persistence in births_and_deaths_by_dim[d]
    '''
    cdef list birth = []
    cdef list death = []
    cdef vector[value_t]& birth_death = births_and_deaths_by_dim[dim]
    cdef Py_ssize_t n_pairs = (Py_ssize_t) birth_death.size() // 2
    for i in range(n_pairs):
        birth.append(birth_death[i * 2])
        death.append(birth_death[i * 2 + 1])
    return np.stack((birth, death), axis=1)

def ripser(
    coo_matrix distance_matrix,
    int modulus,
    int dim_max,
    float threshold,
    bool do_cocycles,
) -> RipserResults:
    cdef np.ndarray[np.intc_t, ndim=1, mode='c'] row = distance_matrix.row.astype(np.intc, copy=True)
    cdef int* I = (int*) row.data
    cdef np.ndarray[np.intc_t, ndim=1, mode='c'] col = distance_matrix.col.astype(np.intc, copy=True)
    cdef int* J = (int*) col.data
    cdef np.ndarray[np.float32_t, ndim=1, mode='c'] data = distance_matrix.data.astype(np.float32, copy=True)
    cdef float* V = (float*) data.data
    cdef int NEdges = distance_matrix.nnz
    cdef int N = distance_matrix.shape[0]
    cdef ripserResults res = rips_dm_sparse(I, J, V, NEdges, N, modulus, dim_max, threshold, (int)do_cocycles)
    cdef list persistence_diagrams = [converting_birth_death_to_numpy(res.births_and_deaths_by_dim, i) for i in range(dim_max)]
    cdef list cocycle_representatives = [converting_cocycles_to_numpy(res.cocycles_by_dim, i) for i in range(1, dim_max)]
    return RipserResults(
        persistence_diagrams,
        cocycle_representatives,
        NEdges
    )
