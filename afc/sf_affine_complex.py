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


def build_B(list_edges, n):
    B = np.zeros((n, len(list_edges)))
    for i in range(len(list_edges)):
        B[list_edges[i][0] - 1, i] = 1
        B[list_edges[i][1] - 1, i] = -1
    return B


########################################################################################


class sim_frame_affine_complex:
    def __init__(
        self, n, Z, p_star, p0, tf, dt=0.001, h=1, K=None, kappa=0.1, debug=False
    ):
        self.data = {"p": None}
        self.debug = debug

        # Graph
        self.n = n
        self.m = 2

        self.Z = Z
        self.E = gen_edges_set(Z)

        # Desired formation
        self.p_star = np.array(p_star)
        if isinstance(self.p_star[0, 0], complex):
            self.p_star_c = self.p_star
        else:
            self.p_star_c = self.p_star[:, 0] + self.p_star[:, 1] * 1j

        # Initial conditions
        self.tf = tf
        self.dt = dt

        self.p0 = np.array(p0)
        if isinstance(self.p0[0, 0], complex):
            self.p0_c = self.p0
        else:
            self.p0_c = self.p0[:, 0] + self.p0[:, 1] * 1j

        # Controller
        self.h = h
        self.kappa = kappa

        if K is None:
            self.K = np.eye(self.n)
        else:
            self.K = K
        self.K_inv = LA.inv(self.K)

        # Generate all the neighbors sets Ni
        self.Ni_list = []
        for i in range(n):
            self.Ni_list.append(gen_Ni(i, n, self.E))

        # ---- Generating weights and Laplacian matrix -
        self.B = gen_inc_matrix(self.n, self.Z)

        # - Shiyu Zhao algorithm
        self.W = gen_weights_r(self.p_star, self.B, self.m)
        self.L = self.B @ self.W @ self.B.T

        # - Debugging print
        if self.debug:
            with np.printoptions(precision=2, suppress=True):
                print("W {:s} = \n".format(str(self.W.shape)), self.W)
                print("L {:s} = \n".format(str(self.L.shape)), self.L)
            print(" --------- Eigenvalues")
            with np.printoptions(precision=8, suppress=True):
                for i, eigen in enumerate(LA.eig(-self.L)[0]):
                    print("lambda_{:d} = {:f}".format(i, eigen))
        # --------------------------------------------------------

        # Initialise the components matrix and the modified laplacian
        self.L_mod = np.copy(self.L)

        # Initialise the simulator
        self.simulator = simulator(self.p0_c, self.dt)

    def check_eigen_vectors(self):
        with np.printoptions(precision=6, suppress=True):
            print("{:<15} = ".format("L@1_n"), self.L @ np.ones(self.n))
            print("{:<15} = ".format("L@p^*"), self.L @ self.p_star_c)
            print("{:<15} = ".format("L@Re(p^*)"), self.L @ np.real(self.p_star_c))
            print("{:<15} = ".format("L@Im(p^*)"), self.L @ np.imag(self.p_star_c))
            print(" ------------ ")

    def set_manual_mu(self, mu_matrix):
        self.M = gen_compnts_matrix(self.n, 1, self.Z, mu_matrix)
        self.L_mod = self.L - self.kappa / self.h * self.K_inv @ self.M @ self.B.T

        if self.debug:
            with np.printoptions(precision=2, suppress=True):
                print(
                    "M", self.M.shape, "B.T", self.B.T.shape, "K_inv", self.K_inv.shape
                )
                print("mu_ij matrix:\n", mu_matrix)
                print("M:\n", mu_matrix)
                print("M@B^T@p_star_c", self.M @ self.B.T @ self.p_star_c)
            print(" --------- Eigenvalues L_mod")
            with np.printoptions(precision=8, suppress=True):
                for i, eigen in enumerate(LA.eig(-self.h * self.L_mod)[0]):
                    print("lambda_{:d} = {:f}".format(i, eigen))

    def numerical_simulation(self):
        # Integration steps
        its = int(self.tf / self.dt)

        # Init data arrays
        pdata_c = np.empty([its, self.n], dtype=complex)
        pdata = np.empty([its, self.n, self.m])

        # Reset the simulator
        self.simulator.reset()

        # Numerical simulation
        for i in tqdm(range(its)):
            pdata_c[i, :] = self.simulator.p
            # Robots simulator euler step integration
            self.simulator.int_euler(self.h, self.K, self.L_mod)

        pdata[:, :, 0] = np.real(pdata_c)
        pdata[:, :, 1] = np.imag(pdata_c)
        self.data["p"] = pdata

    def plot(self, ax=None, lim=20):
        # Extract data
        pdata = self.data["p"]

        flag_newplot = False
        if ax is None:
            flag_newplot = True

            # Figure init and configuration
            fig = plt.figure()
            ax = fig.subplots()

            ax.grid(True)
            ax.set_xlim([-lim, lim])
            ax.set_ylim([-lim, lim])

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

        if flag_newplot:
            ax.set_aspect("equal")
            ax.legend()
