"""
# Copyright (C) 2025 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import itertools

import picos as pic  # version 1.2.0 !!
from scipy import linalg as la


def rot_mat(z):
    theta = np.arctan2(z.imag, z.real)  # (y,x)
    mod = np.sqrt(z.real**2 + z.imag**2)
    return mod * np.array(
        [[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]
    )


def rot_transf_from_ang(n, theta=np.pi / 3):
    return np.kron(
        np.eye(n),
        np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]]),
    )


def sh_transf_x(n, t):
    return np.kron(np.eye(n), np.array([[1, 0], [np.sinh(t), 1]]))


def sh_transf_y(n, t):
    return np.kron(np.eye(n), np.array([[1, np.sinh(t)], [0, 1]]))


## AFFINE FORMATIONS math tools - Complex Laplacian ##

"""
- Generate the components matrix M from the ordered edges set Z 
  and mu_ij -
"""


def gen_compnts_matrix(n, m, Z, mu_matrix):
    M = np.zeros((n * m, len(Z) * m))
    for i in range(n):
        k = 0
        for edge in Z:  # edge = (Z_k^head, Z_k^tail)
            if i + 1 == edge[1]:  # if tail
                M[i * m : (i + 1) * m, k * m : (k + 1) * m] = mu_matrix[
                    i * m : (i + 1) * m, (edge[0] - 1) * m : (edge[0]) * m
                ]  # mu_{i Z_k^head}
            elif i + 1 == edge[0]:  # if head
                M[i * m : (i + 1) * m, k * m : (k + 1) * m] = -mu_matrix[
                    i * m : (i + 1) * m, (edge[1] - 1) * m : (edge[1]) * m
                ]  # mu_{i Z_k^tail}
            k = k + 1
    return M


"""
- Generate the weights accordangly to a desired formation shape
  and incidence matrix or neightbors set (Lin method but in R^m) -
  (p1 is a "random" constant)
"""


def gen_weights_rm(p_star, N_list, p1=1):
    m = 2
    n = int(p_star.shape[0] / m)

    W_complex = np.zeros((n, n), dtype=complex)
    W_affine = np.zeros((n * m, n * m))

    for i in range(n):
        # compute the weights of the first two neightborns
        j, k = N_list[i][0:2]
        w_vector = p1 * np.array(
            [
                (p_star[i * m] + p_star[i * m + 1] * 1j)
                - (p_star[k * m] + p_star[k * m + 1] * 1j),
                (p_star[j * m] + p_star[j * m + 1] * 1j)
                - (p_star[i * m] + p_star[i * m + 1] * 1j),
            ]
        )

        neigh_edge_tuples = list(itertools.combinations(N_list[i], 2))

        w_vector = np.zeros(n, dtype=complex)
        for edge in neigh_edge_tuples:
            # compute the weights two by two neightborns
            j, k = edge
            w_vector[j] = (
                w_vector[j]
                + p1 * (p_star[i * m] + p_star[i * m + 1] * 1j)
                - (p_star[k * m] + p_star[k * m + 1] * 1j)
            )
            w_vector[k] = (
                w_vector[k]
                + p1 * (p_star[j * m] + p_star[j * m + 1] * 1j)
                - (p_star[i * m] + p_star[i * m + 1] * 1j)
            )

        W_complex[i, :] = w_vector

        for j in N_list[i]:
            W_affine[i * m : (i + 1) * m, j * m : (j + 1) * m] = rot_mat(
                W_complex[i, j]
            )

    return W_affine  # 4 deg of freedom (no shearing)


"""
- Generate the weights accordangly to a desired formation shape
  and incidence matrix (Shiyun method with real weights) -
"""


def gen_weights_r(ps, B, m):
    # Algorithm from "Affine Formation Maneuver Control of Multiagent Systems"
    # Transactions on Automatic Control 2017
    # Author: Shiyu Zhao
    numAgents = B.shape[0]

    P = ps.reshape(numAgents, m)
    Pbar = np.concatenate((P.T, np.ones(numAgents).T), axis=None)
    Pbar = Pbar.reshape(m + 1, numAgents).T

    H = B.T

    E = Pbar.T.dot(H.T).dot(np.diag(H[:, 0]))

    for i in range(1, H.shape[1]):
        aux = Pbar.T.dot(H.T).dot(np.diag(H[:, i]))
        E = np.concatenate((E, aux))

    ker_E = la.null_space(E)

    [U, s, Vh] = la.svd(Pbar)

    U2 = U[:, m + 1 :]

    M = []
    for i in range(ker_E.shape[1]):
        aux = U2.T.dot(H.T).dot(np.diag(ker_E[:, i])).dot(H).dot(U2)
        M.append(aux)

    Mc = pic.new_param("Mc", M)
    lmi_problem = pic.Problem()
    c = lmi_problem.add_variable("c", len(M))
    lmi_problem.add_constraint(pic.sum([c[i] * Mc[i] for i in range(len(M))]) >> 0)
    lmi_problem.set_objective("find", c)
    lmi_problem.solve(verbose=0)

    w = np.zeros(ker_E.shape[0])
    for i in range(ker_E.shape[1]):
        w = w + (c[i].value * ker_E[:, i])

    return np.diag(w)


"""
- Generate the Affine Laplacian matrix -
"""


def gen_laplacian(n, m, W, N_list):
    L = np.zeros((n * m, n * m))
    for i in range(n):
        N_i = N_list[i]
        for j in range(n):
            if i == j:
                for k in N_i:
                    L[i * m : (i + 1) * m, j * m : (j + 1) * m] = (
                        L[i * m : (i + 1) * m, j * m : (j + 1) * m]
                        + W[i * m : (i + 1) * m, k * m : (k + 1) * m]
                    )
            elif j in N_i:
                L[i * m : (i + 1) * m, j * m : (j + 1) * m] = -W[
                    i * m : (i + 1) * m, j * m : (j + 1) * m
                ]
    return L
