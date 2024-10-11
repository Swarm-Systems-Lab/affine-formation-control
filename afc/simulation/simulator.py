"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

########################################################################################


class simulator:
    def __init__(self, p0, dt):
        self.t0 = 0
        self.p0 = p0

        # State variables
        self.p, self.dt = p0, dt
        self.n = len(p0)
        self.t = 0

    def reset(self):
        self.t = self.t0
        self.p = self.p0

    def u_control(self, h, K, L):
        u = -h * K @ L @ self.p[:, None]
        return u[:, 0]

    def int_euler(self, h, K, L):
        self.t = self.t + self.dt
        self.p = self.p + self.u_control(h, K, L) * self.dt
