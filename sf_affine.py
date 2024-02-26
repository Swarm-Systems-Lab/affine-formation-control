#!/usr/bin/env python #
"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import numpy.linalg as LA
import matplotlib.pyplot as plt

from tqdm import tqdm

from utils.mt_cmn import gen_edges_set, gen_Ni, gen_inc_matrix
from utils.mt_affine import rot_transf_from_ang, sh_transf_x, sh_transf_y
from utils.mt_affine import gen_weights_rm, gen_weights_r, gen_laplacian
from utils.simulator import simulator

def check_p_dim(p, var_name=""):
    if isinstance(p, list):
        p = np.array(p)
    if len(p.shape) == 2:
        p = p.flatten()
    elif len(p.shape) != 1:
        print("ERR: {:s} wrong dimension".format(var_name), p.shape)
    return p

def build_B(list_edges, n):
    B = np.zeros((n,len(list_edges)))
    for i in range(len(list_edges)):
        B[list_edges[i][0]-1, i] = 1
        B[list_edges[i][1]-1, i] = -1
    return B

"""
"""
class sim_frame_affine:
    def __init__(self, n, Z, p_star, p0, tf, dt = 0.001,
                 h = 1, K = None, kappa = 1,
                 debug = False):
        
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
            self.K = np.eye(n)
        else:
            self.K = K
        self.K_inv = LA.inv(K)
        
        # Generate all the neighbors sets Ni
        self.Ni_list = []
        for i in range(n):
            self.Ni_list.append(gen_Ni(i,n,self.E))

        # Generating weights and laplacian matrix
        self.B = gen_inc_matrix(self.n, self.Z)
        
        # Shiyu Zhao algorithm
        self.W = gen_weights_r(self.p_star, self.B, self.m)
        self.L = np.kron(self.B@self.W@self.B.T, np.eye(self.m))
        
        # As in C^1
        # self.W = gen_weights_rm(self.p_star, self.Ni_list)
        # self.L = gen_laplacian(self.n, self.m, self.W, self.Ni_list)

        # ---- Debugging print -
        if self.debug:
            with np.printoptions(precision=2, suppress=True):
                print("W {:s} = \n".format(str(self.W.shape)), self.W)
                print("L {:s} = \n".format(str(self.L.shape)), self.L)
            print(" --------- Eigen values")
            with np.printoptions(precision=8, suppress=True):
                for i,eigen in enumerate(LA.eig(self.L)[0]):
                    print("lambda_{:d} = {:f}".format(i,eigen))
        # ----
                    
    def check_eigen_vectors(self):
        R45 = rot_transf_from_ang(self.n, np.pi/4)
        Sc = np.kron(np.eye(self.n), 4*np.eye(self.m))
        Shx = sh_transf_x(self.n,1)
        Shy = sh_transf_y(self.n,1)
        A = np.kron(np.eye(self.n), np.eye(self.m))
        A[0:2,0:2] = np.array([[1,0],[0,-1]])
        with np.printoptions(precision=6, suppress=True):
            print("{:<15} = ".format("L@1_n^bar"), self.L @ np.ones(self.n*self.m))
            print("{:<15} = ".format("L@p^*"), self.L @ self.p_star)
            print("{:<15} = ".format("L@R(pi/4)@p^*"), self.L @ R45 @ self.p_star)
            print("{:<15} = ".format("L@(4*I_mn)@p^*"), self.L @ Sc @ self.p_star)
            print("{:<15} = ".format("L@Shx@p^*"), self.L @ Shx @ self.p_star)
            print("{:<15} = ".format("L@Shy@p^*"), self.L @ Shy @ self.p_star)
            print("{:<15} = ".format("L@A@p^*"), self.L @ A @ self.p_star)

        test = self.p_star
        test = test.reshape(self.n, self.m)
        plt.plot(test[:,0], test[:,1], "ok")


        test = Shx @ self.p_star
        test = test.reshape(self.n, self.m)
        plt.plot(test[:,0], test[:,1], "or")

        test = Shy @ self.p_star
        test = test.reshape(self.n, self.m)
        plt.plot(test[:,0], test[:,1], "ob")

        plt.grid(True)
