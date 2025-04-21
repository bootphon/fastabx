#cython: language_level=3
#cython: boundscheck=False
#cython: wraparound=False
#cython: cdivision=True

import numpy as np
cimport numpy as np
from cython.parallel import parallel, prange
from libc.stdlib cimport calloc, free, malloc


ctypedef np.float32_t CTYPE_t
ctypedef np.intp_t IND_t


cdef CTYPE_t c_dtw(const CTYPE_t* distances, int N, int M, int stride) nogil:
    cdef int i, j
    cdef CTYPE_t* cost
    cdef CTYPE_t final_cost, c_diag, c_left, c_up
    cdef int path_len = 1

    cost = <CTYPE_t*>calloc(N * M, sizeof(CTYPE_t))
    cost[0] = distances[0]
    for i in range(1, N):
        cost[i * M] = distances[i * stride] + cost[(i - 1) * M]
    for j in range(1, M):
        cost[j] = distances[j] + cost[j - 1]
    for i in range(1, N):
        for j in range(1, M):
            cost[i * M + j] = distances[i * stride + j] + min(
                cost[(i - 1) * M + j],
                cost[(i - 1) * M + (j - 1)],
                cost[i * M + (j - 1)],
            )
    final_cost = cost[(N - 1) * M + (M - 1)]

    i = N - 1
    j = M - 1
    while i > 0 and j > 0:
        c_up = cost[(i - 1) * M + j]
        c_left = cost[i * M + (j - 1)]
        c_diag = cost[(i - 1) * M + (j - 1)]
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
    free(cost)
    return final_cost / path_len


cpdef dtw(CTYPE_t[:,:] distances):
    cdef int N = distances.shape[0]
    cdef int M = distances.shape[1]
    cdef int stride = distances.strides[0] // sizeof(CTYPE_t)
    return np.asarray(c_dtw(&distances[0, 0], N, M, stride))


def dtw_batch(CTYPE_t[:, :, :, :] distances, IND_t[:] sx, IND_t[:] sy, bint symmetric):
    cdef int i, j, k, num_tasks
    cdef int nx = distances.shape[0]
    cdef int ny = distances.shape[1]
    cdef CTYPE_t[:,:] out = np.zeros((nx, ny), dtype=np.float32)

    num_tasks = nx * (ny - 1) // 2 if symmetric else nx * ny
    cdef int* task_i = <int*>malloc(num_tasks * sizeof(int))
    cdef int* task_j = <int*>malloc(num_tasks * sizeof(int))
    k = 0
    for i in range(nx):
        for j in range(i + 1 if symmetric else 0, ny):
            task_i[k] = i
            task_j[k] = j
            k += 1

    with nogil, parallel():
        for k in prange(num_tasks, schedule='dynamic'):
            i = task_i[k]
            j = task_j[k]
            out[i, j] = c_dtw(
                &distances[i, j, 0, 0],
                sx[i],
                sy[j],
                distances.strides[2] // sizeof(CTYPE_t),
            )
            if symmetric:
                out[j, i] = out[i, j]
    free(task_i)
    free(task_j)
    return np.asarray(out)
