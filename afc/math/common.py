"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np

from numpy.typing import NDArray
from typing import List, Tuple

__all__ = ["toComplex", "toReal", "gen_projectors", 
           "gen_Ni", "gen_edges_set", "gen_inc_matrix"]

## COMMON math tools ##################################################################

## General Use ------------------------------------------------------------------------

def toComplex(array: NDArray):
    """
    2D-real numpy array to 1D-complex
    """
    if isinstance(array.flatten()[0], complex):
        return array
    else:
        return array[:, 0] + array[:, 1] * 1j
    
def toReal(array: NDArray):
    """
    1D-complex numpy array to 2D-real  
    """
    if not isinstance(array.flatten()[0], complex):
        return array
    else:
        return np.array([np.real(array), np.imag(array)]).T


## Graphs -----------------------------------------------------------------------------

def gen_projectors(basis: NDArray) -> List[NDArray]:
    """
    Given a basis $B = \{v1, ..., vn\}$, generate its associated projector matrices
    """
    A = np.array([*basis]).T
    P = A @ np.linalg.inv(A.T @ A) @ A.T
    P_perp = np.eye(A.shape[0]) - P
    return P, P_perp


def gen_Ni(i: int, n: int, edges_set: List[Tuple[int]]):
    """
    Generate the set of neightborns of i (index) from the edges set
    """
    N_i = []
    for k in range(n):
        if i != k and ((i + 1, k + 1) in edges_set or (k + 1, i + 1) in edges_set):
            N_i.append(k)
    return N_i


def gen_edges_set(Z: List[Tuple]):
    """
    Generate the edges set from Z
    """
    E = set()
    for edge in Z:
        E.add(edge)
        E.add((edge[1], edge[0]))
    return E


def gen_inc_matrix(n: int, Z: List[Tuple[int]]):
    """
    Generate the incidence matrix B from the ordered edges set Z
    """
    B = np.zeros((n, len(Z)))
    for i in range(len(Z)):  # edge = (Z_k^head, Z_k^tail)
        B[Z[i][0] - 1, i] = -1  # head
        B[Z[i][1] - 1, i] = 1  # tail
    return B

#######################################################################################
