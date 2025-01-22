"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import numpy.linalg as LA
from tqdm import tqdm

from ssl_pysimutils import debug_eig

from .simulator import Simulator
from ..math.common import gen_edges_set, gen_Ni, gen_inc_matrix
from ..math.complex_lap import gen_weights, gen_laplacian, gen_compnts_matrix
from ..plots import plot_xy


def toComplex(array: np.ndarray):
    """
    2 dimensional numpy array to complex
    """
    if isinstance(array.flatten()[0], complex):
        return array
    else:
        return array[:, 0] + array[:, 1] * 1j


#######################################################################################

__all__ = ["ComplexSimulator"]

class ComplexSimulator(Simulator):
    def __init__(self, Z, p_star, p0, tf, dt=0.001, K=None, kappa=1, h=1, p1=(1+2j)):
        super().__init__(x0=toComplex(np.array(p0)), dt=dt, t0=0)

        self.tf = tf
        self.n = len(self.x0)

        # Set the graph
        self.Z = Z
        self.E = gen_edges_set(Z)

        # Set the desired formation
        self.p_star = np.array(p_star)

        # Set controller settings
        self.h = h
        self.kappa = kappa

        if K is None:
            self.K = np.eye(self.n)
        else:
            self.K = K
        self.K_inv = LA.inv(self.K)

        # Initialise the controller
        self.control_init(p1)

    def control_init(self, p1):
        # Generate all the neighbors sets Ni
        self.Ni_list = []
        for i in range(self.n):
            self.Ni_list.append(gen_Ni(i, self.n, self.E))

        # Generating weights and laplacian matrix
        self.B = gen_inc_matrix(self.n, self.Z)
        self.W = gen_weights(self.p_star, self.Ni_list, p1=p1)
        self.L = gen_laplacian(self.n, self.W, self.Ni_list)

        # Initialise the components matrix and the modified laplacian
        self.L_mod = np.copy(self.L)

    def control_single_integrator(self, p: np.ndarray):
        u = -self.h * self.K @ self.L_mod @ p[:, None]
        p_dot = u[:, 0]
        return p_dot

    def set_velocity(self, vx, vy, a, omega):
        # Stack velocity vector
        self.vf = (
            (vx + 1j *vy) * np.ones(self.n) +
            (a + 1j * omega) * self.p_star
        )

        # Matrix of motion marameters mu_ij
        mu_matrix = np.zeros((self.n, self.n), dtype=complex)
        for i in range(self.n):
            mu_i = np.zeros(self.n, dtype=complex)
            j_neig = self.Ni_list[i][0]
            mu_i[j_neig] = self.vf[i] / (self.p_star[i] - self.p_star[j_neig])
            mu_matrix[i, :] = mu_i

        self.M = gen_compnts_matrix(self.n, self.Z, mu_matrix)
        self.L_mod = self.L - self.kappa / self.h * self.K_inv @ self.M @ self.B.T

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
        xdata = np.real(self.data["x"])
        ydata = np.imag(self.data["x"])
        plot_xy(xdata, ydata, self.Z, ax_input=ax_input, lim=lim)

    ## Debugging Functions ------------------------------------------------------------
    def check_W_L(self, eigenvectors=True, prec=[2, 8, 3]):
        with np.printoptions(precision=prec[0], suppress=True):
            print("B = \n", self.B)
            print("L = \n", self.L)
            print("\nL eigenvalues:")

        debug_eig(-np.array(self.L, dtype=complex), eigenvectors, *prec[1:])

    def check_kernel(self):
        with np.printoptions(precision=6, suppress=True):
            ps = self.p_star
            print("L@1_n = \n", self.L @ (np.ones((self.n))))
            print("L@p^star = \n", self.L @ ps)
            print("L@Re(p^*) = \n", self.L @ np.real(ps))
            print("L@Im(p^*) = \n", self.L @ np.imag(ps))
            print(" ------------ ")

    def check_M(self, eigenvectors=True, prec=[4, 8, 3]):
        with np.printoptions(precision=prec[0], suppress=True):
            print("\nv_f = \n", self.vf)
            print("\nM = \n", self.M)
            print("\nL@1_n = \n", self.L @ (np.ones((self.n, 1))))
            print("\nL@v_f = \n", self.L @ self.vf[:, None])  # TODO: thm. 3
            print("\nL eigenvalues:")
            for i, eigen in enumerate(LA.eig(-self.L_mod)[0]):
                print("lambda_{:d} = {:f}".format(i, eigen))
        
        print("\n -> L_mod")
        debug_eig(np.array(-self.h * self.L_mod), eigenvectors, *prec[1:])


#######################################################################################
