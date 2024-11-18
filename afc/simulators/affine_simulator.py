"""
# Copyright (C) 2025 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import numpy.linalg as LA
from tqdm import tqdm

from ssl_pysimutils import debug_eig

from .simulator import Simulator
from ..math.common import gen_edges_set, gen_Ni, gen_inc_matrix
from ..math.real_lap import rot_transf_from_ang, sh_transf_x, sh_transf_y, rot_mat
from ..math.real_lap import gen_weights_rm, gen_weights_r, gen_compnts_matrix
from ..plots import plot_xy


def check_p_dim(p):
    if isinstance(p, list):
        p = np.array(p)
    if len(p.shape) == 2:
        p = p.flatten()
    elif len(p.shape) != 1:
        raise ValueError
    return p


#######################################################################################


class AffineSimulator(Simulator):
    def __init__(
        self, Z, p_star, p0, tf, dt=0.001, h=1, K=None, kappa=0.1
    ):  
        super().__init__(x0=check_p_dim(p0), dt=dt, t0=0)

        self.tf = tf
        self.m = 2
        self.n = int(len(self.x0)/2)

        # Set the graph
        self.Z = Z
        self.E = gen_edges_set(Z)

        # Set the desired formation 
        self.p_star = check_p_dim(p_star)

        # Set controller settings
        self.h = h
        self.kappa = kappa

        if K is None:
            self.K = np.kron(np.eye(self.n), np.eye(self.m))
        else:
            self.K = np.kron(K, np.eye(self.m))
        self.K_inv = LA.inv(self.K)

        # Initialise the controller
        self.control_init()

    def control_init(self):
        # Generate all the neighbors sets Ni
        self.Ni_list = []
        for i in range(self.n):
            self.Ni_list.append(gen_Ni(i, self.n, self.E))

        # ---- Generating weights and Laplacian matrix -
        self.B = gen_inc_matrix(self.n, self.Z)
        self.B_bar_T = np.kron(self.B.T, np.eye(self.m))

        # - Shiyu Zhao algorithm
        self.W = gen_weights_r(self.p_star, self.B, self.m)
        self.L = np.kron(self.B @ self.W @ self.B.T, np.eye(self.m))

        # - Lin C^1 algorithm but in R^2
        # self.W = gen_weights_rm(self.p_star, self.Ni_list)
        # self.L = gen_laplacian(self.n, self.m, self.W, self.Ni_list)

        # Initialise the components matrix and the modified laplacian
        self.L_mod = np.copy(self.L)

    def control_single_integrator(self, p: np.ndarray):
        u = -self.h * self.K @ self.L_mod @ p[:, None]
        p_dot = u[:, 0]
        return p_dot
    
    def set_manual_mu(self, mu_matrix):
        mu_matrix = np.kron(mu_matrix, np.eye(self.m))
        self.M = gen_compnts_matrix(self.n, self.m, self.Z, mu_matrix)
        self.L_mod = self.L - self.kappa / self.h * self.K_inv @ self.M @ self.B_bar_T

    def numerical_simulation(self):
        # Reset the simulator
        self.reset()
        self.state_dynamics = self.control_single_integrator

        # Numerical simulation
        its = int(self.tf / self.dt)
        for i in tqdm(range(its), desc="Executing numerical simulation"):
            # update the simulator variables and data
            self.int_euler()
            self.update_data()

    ## Provide data to plot -----------------------------------------------------------
    def plot(self, ax_input=None, lim=20):
        xdata = np.array(self.data["x"])[:,0::2]
        ydata = np.array(self.data["x"])[:,1::2]
        plot_xy(xdata, ydata, self.Z, ax_input=ax_input, lim=lim)

    ## Debugging Functions ------------------------------------------------------------
    def check_W_L(self, eigenvectors=True, prec=[2, 8, 3]):
        with np.printoptions(precision=prec[0], suppress=True):
            print("W {:s} = \n".format(str(self.W.shape)), self.W)
            print("L {:s} = \n".format(str(self.L.shape)), self.L)

        debug_eig(-np.array(self.L, dtype=complex), eigenvectors, *prec[1:])

    def check_kernel(self):
        print("--- Kernel Eigenvectors")
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

    def check_M(self, eigenvectors=False, prec=[2,4,3]):
        with np.printoptions(precision=prec[0], suppress=True):
            print(
                "M", self.M.shape, "B.T", self.B.T.shape, "K_inv", self.K_inv.shape
            )
            # print("mu_ij matrix:\n", mu_matrix)
            # print("M:\n", mu_matrix)
            print("M^bar@B^T^bar@p_star:", self.M @ self.B_bar_T @ self.p_star)

        print("\n -> L_mod")
        debug_eig(np.array(-self.h * self.L_mod), eigenvectors, *prec[1:])

#######################################################################################

