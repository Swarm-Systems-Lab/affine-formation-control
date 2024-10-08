"""
# Copyright (C) 2025 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

########################################################################################


class simulator:
    def __init__(self, p0, dt):
        # State variables
        self.p, self.dt = p0, dt
        self.n = len(p0)
        self.t = 0

    def u_control(self, h, K, L):
        u = -h * K @ L @ self.p[:, None]
        return u[:, 0]

    def int_euler(self, h, K, L):
        self.p = self.p + self.u_control(h, K, L) * self.dt
