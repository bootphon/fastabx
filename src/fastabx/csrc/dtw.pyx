#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
from cython.parallel import prange
cimport numpy as np


ctypedef np.float32_t CTYPE_t
ctypedef np.intp_t IND_t


cpdef dtw(CTYPE_t[:,:] distances):
    cdef IND_t i, j
    cdef CTYPE_t[:,:] cost = np.zeros_like(distances)
    cdef CTYPE_t final_cost, c_diag, c_left, c_up
    cdef IND_t N = distances.shape[0]
    cdef IND_t M = distances.shape[1]
    cost[0, 0] = distances[0, 0]
    for i in range(1, N):
        cost[i, 0] = distances[i, 0] + cost[i - 1, 0]
    for j in range(1, M):
        cost[0, j] = distances[0, j] + cost[0, j - 1]
    for i in range(1, N):
        for j in range(1, M):
            cost[i, j] = distances[i, j] + min(cost[i - 1, j], cost[i - 1, j - 1], cost[i, j - 1])
    final_cost = cost[N - 1, M - 1]

    path_len = 1
    i = N - 1
    j = M - 1
    while i > 0 and j > 0:
        c_up = cost[i - 1, j]
        c_left = cost[i, j - 1]
        c_diag = cost[i - 1, j - 1]
        if c_diag <= c_left and c_diag <= c_up:
            i -= 1
            j -= 1
        elif c_left <= c_up:
            j -= 1
        else:
            i -= 1
        path_len += 1
    if i == 0:
        path_len += j
    if j == 0:
        path_len += i
    return final_cost / path_len


def dtw_batch(CTYPE_t[:, :, :, :] distances, IND_t[:] sx, IND_t[:] sy, symmetric):
    cdef int i
    cdef int j
    cdef int start_index
 
    nx, ny = distances.shape[0], distances.shape[1]
    out = np.zeros((nx, ny), dtype=np.float32)
    for i in range(nx):
        start_index = i if symmetric else 0
        for j in range(start_index, ny):
            if symmetric and i == j:
                continue
            out[i][j] = dtw(distances[i, j, :sx[i], :sy[j]])
            if symmetric and i != j:
                out[j][i] = out[i][j]
    return out
