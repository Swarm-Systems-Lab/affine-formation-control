"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np


def dyn_func(p: np.ndarray, dyn_args):
    p_dot = None
    return p_dot


#######################################################################################

__all__ = ["Simulator"]

class Simulator:
    def __init__(
        self, x0: np.ndarray, dt: float, t0: float = 0, state_dynamics=dyn_func
    ):
        self.x0 = x0
        self.dt = dt
        self.t0 = t0

        # State dynamics
        self.state_dynamics = state_dynamics

        # Simulator variables
        self.sim_variables = {
            "t": self.t0,
            "x": x0,
        }

        # Data dictionary (stores the data from each step of the simulation)
        self.data = {s: [] for s in self.sim_variables.keys()}

    def reset(self):
        """
        Reset the simulation variables and clean the data dictionary
        """
        self.sim_variables["t"] = self.t0
        self.sim_variables["x"] = self.x0
        self.data = {s: [] for s in self.sim_variables.keys()}

    def update_data(self):
        """
        Update the data dictionary with a new entry
        """
        for key in self.sim_variables.keys():
            self.data[key].append(np.copy(self.sim_variables[key]))

    def int_euler(self, dyn_args=None):
        if dyn_args is None:
            x_dot = self.state_dynamics(self.sim_variables["x"])
        else:
            x_dot = self.state_dynamics(self.sim_variables["x"], *dyn_args)

        # Integrate
        self.sim_variables["t"] = self.sim_variables["t"] + self.dt
        self.sim_variables["x"] = self.sim_variables["x"] + x_dot * self.dt

#######################################################################################