"""
# Copyright (C) 2025 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from tqdm import tqdm

from .math.mt_cmn import gen_edges_set, gen_Ni, gen_inc_matrix
from .math.mt_complex import gen_weights, gen_laplacian, gen_compnts_matrix
from .simulator import simulator

########################################################################################


def check_p_dim(p, var_name=""):
    if isinstance(p, list):
        p = np.array(p, dtype=complex)
    if len(p.shape) != 1:
        print("ERR: {:s} wrong dimension".format(var_name), p.shape)
    return p


########################################################################################


class sim_frame_complex:
    def __init__(
        self,
        n,
        Z,
        p_star,
        p0,
        tf,
        dt=0.001,
        h=1,
        K=None,
        kappa=1,
        p1=(1 + 2j),
        debug=False,
    ):

        self.data = {"p": None}
        np.random.seed(2024)
        self.debug = debug

        # Graph
        self.n = n
        self.Z = Z
        self.E = gen_edges_set(Z)

        # Desired formation
        self.p_star = check_p_dim(p_star, "p_star")

        # Initial conditions
        self.p0 = check_p_dim(p0, "p0")
        self.tf = tf
        self.dt = dt

        # Controller
        self.h = h
        self.kappa = kappa

        if K is None:
            self.K = np.eye(n)
        else:
            self.K = K
        self.K_inv = LA.inv(K)

        # Generate all the neighbors sets Ni
        self.Ni_list = []
        for i in range(n):
            self.Ni_list.append(gen_Ni(i, n, self.E))

        # Generating weights and laplacian matrix
        self.B = gen_inc_matrix(self.n, self.Z)
        self.W = gen_weights(self.p_star, self.Ni_list, p1=p1)
        self.L = gen_laplacian(self.n, self.W, self.Ni_list)

        # ---- Debugging print -
        if self.debug:
            print("B = \n", self.B)
            print("L = \n", self.L)
            print("\nL eigenvalues:")
            for i, eigen in enumerate(LA.eig(-self.L)[0]):
                print("lambda_{:d} = {:f}".format(i, eigen))
        # ----

        # Initialise the components matrix and the modified laplacian
        self.L_mod = np.copy(self.L)

        # Initialise the simulator
        self.simulator = simulator(self.p0, self.dt)

    def check_eigen_vectors(self):
        with np.printoptions(precision=6, suppress=True):
            ps = self.p_star
            print("L@1_n = \n", self.L @ (np.ones((self.n))))
            print("L@p^star = \n", self.L @ ps)
            print("L@Re(p^*) = \n", self.L @ np.real(ps))
            print("L@Im(p^*) = \n", self.L @ np.imag(ps))
            print(" ------------ ")

    def set_velocity(self, vx, vy, a, omega):
        # Stack velocity vector
        vf = (
            (vx + vy * (1j)) * np.ones(self.n)
            + a * self.p_star
            + omega * self.p_star * (1j)
        )

        # Matrix of motion marameters mu_ij
        mu_matrix = np.zeros((self.n, self.n), dtype=complex)
        for i in range(self.n):
            mu_i = np.zeros(self.n, dtype=complex)
            j_neig = self.Ni_list[i][0]
            mu_i[j_neig] = vf[i] / (self.p_star[i] - self.p_star[j_neig])
            mu_matrix[i, :] = mu_i

        M = gen_compnts_matrix(self.n, self.Z, mu_matrix)
        self.L_mod = self.L - self.kappa / self.h * self.K_inv @ M @ self.B.T

        if self.debug:
            print("\nv_f = \n", vf)
            print("\nM = \n", M)
            print("\nL@1_n = \n", self.L @ (np.ones((self.n, 1))))
            print("\nL@v_f = \n", self.L @ vf[:, None])  # TODO: thm. 3
            print("\nL eigenvalues:")
            for i, eigen in enumerate(LA.eig(-self.L_mod)[0]):
                print("lambda_{:d} = {:f}".format(i, eigen))

    def numerical_simulation(self):
        # Integration steps
        its = int(self.tf / self.dt)

        # Init data arrays
        pdata = np.empty([its, self.n], dtype=complex)

        # Numerical simulation
        for i in tqdm(range(its)):
            pdata[i, :] = self.simulator.p
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
            ax.plot(pdata[:, i].real, pdata[:, i].imag, c=colors[i], lw=0.8)
            ax.plot(pdata[0, i].real, pdata[0, i].imag, "x", c=colors[i])
            ax.plot(pdata[-1, i].real, pdata[-1, i].imag, ".", c=colors[i])

        for edge in self.Z:
            i, j = np.array(edge) - 1
            ax.plot(
                [pdata[-1, i].real, pdata[-1, j].real],
                [pdata[-1, i].imag, pdata[-1, j].imag],
                "k--",
                lw=0.8,
            )

        plt.show()
