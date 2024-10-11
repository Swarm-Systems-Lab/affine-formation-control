"""
# Copyright (C) 2025 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from tqdm import tqdm

from .math.mt_cmn import gen_edges_set, gen_Ni, gen_inc_matrix
from .math.mt_affine import rot_transf_from_ang, sh_transf_x, sh_transf_y, rot_mat
from .math.mt_affine import gen_weights_rm, gen_weights_r, gen_compnts_matrix
from .simulator import simulator

########################################################################################

def check_p_dim(p, var_name=""):
    if isinstance(p, list):
        p = np.array(p)
    if len(p.shape) == 2:
        p = p.flatten()
    elif len(p.shape) != 1:
        print("ERR: {:s} wrong dimension".format(var_name), p.shape)
    return p


def build_B(list_edges, n):
    B = np.zeros((n, len(list_edges)))
    for i in range(len(list_edges)):
        B[list_edges[i][0] - 1, i] = 1
        B[list_edges[i][1] - 1, i] = -1
    return B

########################################################################################

class sim_frame_affine:
    def __init__(
        self, n, Z, p_star, p0, tf, dt=0.001, h=1, K=None, kappa=0.1, debug=False
    ):

        self.data = {"p": None}
        np.random.seed(2024)
        self.debug = debug

        # Graph
        self.n = n
        self.m = 2

        self.Z = Z
        self.E = gen_edges_set(Z)

        # Desired formation
        self.p_star = check_p_dim(p_star, "p0")

        # Initial conditions
        self.p0 = check_p_dim(p0, "p0")
        self.tf = tf
        self.dt = dt

        # Controller
        self.h = h
        self.kappa = kappa

        if K is None:
            self.K = np.kron(np.eye(self.n), np.eye(self.m))
        else:
            self.K = np.kron(K, np.eye(self.m))
        self.K_inv = LA.inv(self.K)

        # Generate all the neighbors sets Ni
        self.Ni_list = []
        for i in range(n):
            self.Ni_list.append(gen_Ni(i, n, self.E))

        # ---- Generating weights and Laplacian matrix -
        self.B = gen_inc_matrix(self.n, self.Z)
        self.B_bar_T = np.kron(self.B.T, np.eye(self.m))

        # - Shiyu Zhao algorithm
        self.W = gen_weights_r(self.p_star, self.B, self.m)
        self.L = np.kron(self.B @ self.W @ self.B.T, np.eye(self.m))

        # - Lin C^1 algorithm but in R^2
        # self.W = gen_weights_rm(self.p_star, self.Ni_list)
        # self.L = gen_laplacian(self.n, self.m, self.W, self.Ni_list)

        # - Debugging print
        if self.debug:
            with np.printoptions(precision=2, suppress=True):
                print("W {:s} = \n".format(str(self.W.shape)), self.W)
                print("L {:s} = \n".format(str(self.L.shape)), self.L)
            print(" --------- Eigen values")
            with np.printoptions(precision=8, suppress=True):
                for i, eigen in enumerate(LA.eig(-self.L)[0]):
                    print("lambda_{:d} = {:f}".format(i, eigen))
        # --------------------------------------------------------

        # Initialise the components matrix and the modified laplacian
        self.L_mod = np.copy(self.L)

        # Initialise the simulator
        self.simulator = simulator(self.p0, self.dt)

    """
    """

    def check_eigen_vectors(self):
        R90 = rot_transf_from_ang(self.n, np.pi / 2)
        Sc = np.kron(np.eye(self.n), np.eye(self.m))
        Shx = sh_transf_x(self.n, 1)
        Shy = sh_transf_y(self.n, 1)

        with np.printoptions(precision=6, suppress=True):
            print("{:<15} = ".format("L@1_n^bar"), self.L @ np.ones(self.n * self.m))
            print("{:<15} = ".format("L@p^*"), self.L @ self.p_star)
            print("{:<15} = ".format("L@(I_mn)@p^*"), self.L @ Sc @ self.p_star)
            print("{:<15} = ".format("L@R(pi/4)@p^*"), self.L @ R90 @ self.p_star)
            print("{:<15} = ".format("L@Shx@p^*"), self.L @ Shx @ self.p_star)
            print("{:<15} = ".format("L@Shy@p^*"), self.L @ Shy @ self.p_star)
            print(" ------------ ")

    """
    """

    def set_manual_mu(self, mu_matrix, debug_eigenvectors=False):
        mu_matrix = np.kron(mu_matrix, np.eye(self.m))
        self.M = gen_compnts_matrix(self.n, self.m, self.Z, mu_matrix)
        self.L_mod = self.L - self.kappa / self.h * self.K_inv @ self.M @ self.B_bar_T

        if self.debug:
            with np.printoptions(precision=2, suppress=True):
                print(
                    "M", self.M.shape, "B.T", self.B.T.shape, "K_inv", self.K_inv.shape
                )
                # print("mu_ij matrix:\n", mu_matrix)
                # print("M:\n", mu_matrix)
                print("M^bar@B^T^bar@p_star:", self.M @ self.B_bar_T @ self.p_star)

            print(" --------- Eigenvalues L_mod")
            with np.printoptions(precision=4, suppress=True):
                eg = LA.eig(-self.h * self.L_mod)
                for i in range(len(eg[0])):
                    print("lambda_{:d} = {:f}".format(i, eg[0][i]))
                    if debug_eigenvectors:
                        print(eg[1][:, i])

    """
    """

    def numerical_simulation(self):
        # Integration steps
        its = int(self.tf / self.dt)

        # Init data arrays
        pdata = np.empty([its, self.n, self.m])

        # Numerical simulation
        for i in tqdm(range(its)):
            pdata[i, :, :] = self.simulator.p.reshape(self.n, self.m)
            # Robots simulator euler step integration
            self.simulator.int_euler(self.h, self.K, self.L_mod)

        self.data["p"] = pdata

    def plot(self):
        # Extract data
        pdata = self.data["p"]

        # Figure init and configuration
        fig = plt.figure()
        ax = fig.subplots()

        ax.grid(True)
        ax.set_xlim([-20, 20])
        ax.set_ylim([-20, 20])

        # Plotting
        colors = ["k", "b", "r", "g"]
        for i in range(self.n):
            ax.plot(pdata[:, i, 0], pdata[:, i, 1], c=colors[i], lw=0.8)
            ax.plot(pdata[0, i, 0], pdata[0, i, 1], "x", c=colors[i])
            ax.plot(pdata[-1, i, 0], pdata[-1, i, 1], ".", c=colors[i], label=i)

        for edge in self.Z:
            i, j = np.array(edge) - 1
            ax.plot(
                [pdata[-1, i, 0], pdata[-1, j, 0]],
                [pdata[-1, i, 1], pdata[-1, j, 1]],
                "k--",
                lw=0.8,
            )

        plt.legend()
        plt.show()
