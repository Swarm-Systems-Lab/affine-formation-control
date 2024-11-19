"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import numpy.linalg as LA
from tqdm import tqdm

from ssl_pysimutils import debug_eig

from .simulator import Simulator
from ..math.common import toComplex, toReal
from ..math.common import gen_edges_set, gen_inc_matrix, gen_projectors
from ..math.real_lap import gen_weights_rm, gen_weights_r, gen_compnts_matrix
from ..plots import plot_xy

#######################################################################################


class AffineComplexSimulator(Simulator):
    def __init__(self, Z, p_star, p0, tf, dt=0.001, K=None, kappa=1, h=1):
        super().__init__(x0=toComplex(np.array(p0)), dt=dt, t0=0)

        self.tf = tf
        self.n = len(self.x0)

        # Add new simulator variables
        self.sim_variables["p_perp_norm"] = None
        self.reset()

        # Set the graph
        self.Z = Z
        self.E = gen_edges_set(Z)

        # Set the desired formation and calculate the projection matrices
        self.p_star = toReal(np.array(p_star))
        self.p_star_c = toComplex(np.array(p_star))

        basis_S = [np.ones(self.n), np.real(self.p_star_c), np.imag(self.p_star_c)]
        self.Ps, self.Psp = gen_projectors(basis_S)

        # Set controller settings
        self.h = h
        self.kappa = kappa

        if K is None:
            self.K = np.eye(self.n)
        else:
            self.K = K
        self.K_inv = LA.inv(self.K)

        # Initialise the controller
        self.control_init()

    def control_init(self):
        # Generate the incidence, weights, and Laplacian matrices
        self.B = gen_inc_matrix(self.n, self.Z)
        self.W = gen_weights_r(self.p_star, self.B, 2)  # Shiyu Zhao algorithm
        self.L = self.B @ self.W @ self.B.T

        # Initialise the modified laplacian
        self.L_mod = np.copy(self.L)

    def control_single_integrator(self, p: np.ndarray):
        u = -self.h * self.K @ self.L_mod @ p[:, None]
        p_dot = u[:, 0]
        return p_dot

    def set_manual_mu(self, mu_matrix):
        self.M = gen_compnts_matrix(self.n, 1, self.Z, mu_matrix)
        self.L_mod = self.L - self.kappa / self.h * self.K_inv @ self.M @ self.B.T
        # for debugging
        self.mu_matrix = mu_matrix

    def numerical_simulation(self, reset=True):
        # Reset the simulator
        if reset:
            self.reset()
        self.state_dynamics = self.control_single_integrator

        # Numerical simulation
        its = int(self.tf / self.dt)
        for i in tqdm(range(its), desc="Executing numerical simulation"):
            p = self.sim_variables["x"]
            # update the simulator variables and data
            self.sim_variables["p_perp_norm"] = LA.norm(self.Psp @ p)
            self.int_euler()
            self.update_data()

    ## Provide data to plot -----------------------------------------------------------
    def plot(self, ax_input=None, lim=20, alpha1=1, alpha2=1, colored=True, tail=True, lw_net=0.8):
        xdata = np.real(self.data["x"])
        ydata = np.imag(self.data["x"])
        plot_xy(xdata, ydata, self.Z, ax_input, lim, alpha1, alpha2, colored, tail, lw_net)
            
    ## Debugging Functions ------------------------------------------------------------
    def check_W_L(self, eigenvectors=True, prec=[2, 8, 3]):
        with np.printoptions(precision=prec[0], suppress=True):
            print("W {:s} = \n".format(str(self.W.shape)), self.W)
            print("L {:s} = \n".format(str(self.L.shape)), self.L)

        debug_eig(-np.array(self.L, dtype=complex), eigenvectors, *prec[1:])

    def check_kernel(self):
        print("--- Kernel Eigenvectors")
        with np.printoptions(precision=6, suppress=True):
            print("{:<15} = ".format("L@1_n"), self.L @ np.ones(self.n))
            print("{:<15} = ".format("L@p^*"), self.L @ self.p_star_c)
            print("{:<15} = ".format("L@Re(p^*)"), self.L @ np.real(self.p_star_c))
            print("{:<15} = ".format("L@Im(p^*)"), self.L @ np.imag(self.p_star_c))
            print(" ------------ ")
    
    def check_M(self, eigenvectors=True, prec=[2,8,3]):
        with np.printoptions(precision=prec[0], suppress=True):
            print("M", self.M.shape, "B.T", self.B.T.shape, "K_inv", self.K_inv.shape)
            print("mu_ij matrix:\n", self.mu_matrix)
            print("M:\n", self.M)
            print("M@B^T@p_star_c", self.M @ self.B.T @ self.p_star_c)

        print("\n -> L_mod")
        debug_eig(np.array(-self.h * self.L_mod), eigenvectors, *prec[1:])

    def check_MBT(self, eigenvectors=True, prec=[2,8,3]):
        print("\n -> (M @ B.T).H @ (M @ B.T)")
        A = np.conj(self.M @ self.B.T).T @ (self.M @ self.B.T)
        debug_eig(np.array(A), eigenvectors, *prec[1:])

#######################################################################################
