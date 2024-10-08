"""
# Copyright (C) 2025 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import itertools

## COMPLEX math tools - Complex Laplacian ##

"""
- Generate the components matrix M from the ordered edges set Z 
  and mu_ij -
"""


def gen_compnts_matrix(n, Z, mu_matrix):
    M = np.zeros((n, len(Z)), dtype=complex)
    for i in range(n):
        k = 0
        for edge in Z:  # edge = (Z_k^head, Z_k^tail)
            if i + 1 == edge[1]:  # if tail
                M[i, k] = mu_matrix[i, edge[0] - 1]  # mu_{i Z_k^head}
            elif i + 1 == edge[0]:  # if head
                M[i, k] = -mu_matrix[i, edge[1] - 1]  # mu_{i Z_k^tail}
            k = k + 1
    return M


"""
- Generate the weights accordangly to a desired formation shape
  and incidence matrix -
  REF: https://ieeexplore.ieee.org/abstract/document/6750042
  (p1 is a "random" constant)
"""


def gen_weights(p_star, N_list, p1=(1 + 2j)):
    n = len(p_star)
    W = np.zeros((n, n), dtype=complex)

    for i in range(n):
        neigh_edge_tuples = list(itertools.combinations(N_list[i], 2))

        w_vector = np.zeros(n, dtype=complex)
        for edge in neigh_edge_tuples:
            # compute the weights two by two neightborns
            j, k = edge
            w_vector[j] = w_vector[j] + p1 * (p_star[i] - p_star[k])
            w_vector[k] = w_vector[k] + p1 * (p_star[j] - p_star[i])

        W[i, :] = w_vector

    return W


"""
- Generate the Complex Laplacian matrix -
"""


def gen_laplacian(n, W, N_list):
    L = np.zeros((n, n), dtype=complex)
    for i in range(n):
        N_i = N_list[i]
        for j in range(n):
            if i == j:
                L[i, j] = np.sum(W[i, N_i])
            elif j in N_i:
                L[i, j] = -W[i, j]
    return L
