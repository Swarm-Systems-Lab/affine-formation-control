"""
# Copyright (C) 2025 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

## COMMON math tools (graphs) ##

"""
- Generate the set of neightborns of i (index) from the edges set -
"""


def gen_Ni(i, n, edges_set):
    N_i = []
    for k in range(n):
        if i != k and ((i + 1, k + 1) in edges_set or (k + 1, i + 1) in edges_set):
            N_i.append(k)
    return N_i


"""
- Generate the edges set from Z -
"""


def gen_edges_set(Z):
    E = set()
    for edge in Z:
        E.add(edge)
        E.add((edge[1], edge[0]))
    return E


"""
- Generate the incidence matrix B from the ordered edges set Z -
"""


def gen_inc_matrix(n, Z):
    B = np.zeros((n, len(Z)))
    for i in range(n):
        k = 0
        for edge in Z:  # edge = (Z_k^head, Z_k^tail)
            if i + 1 == edge[1]:  # if tail
                B[i, k] = 1
            elif i + 1 == edge[0]:  # if head
                B[i, k] = -1
            k = k + 1
    return B
