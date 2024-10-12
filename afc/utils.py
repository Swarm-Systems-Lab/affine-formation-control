"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import numpy.linalg as LA

from .simulators import AffineComplexSimulator

## Utils ------------------------------------------------------------------------------

def check_case(sim_fr: AffineComplexSimulator, params: list[float], debug=False):
    vx, vy, a, omega, hx, hy = params

    kappa = sim_fr.kappa

    # Complex conjugate eigen values
    sigma1 = np.sqrt((hx - omega) * (hy + omega) + 0j)
    sigma2 = -np.sqrt((hx - omega) * (hy + omega) + 0j)
    l1 = kappa * (a + sigma1)
    l2 = kappa * (a + sigma2)

    # Calculate (hx - omega) and (hy + omega)
    hxw = hx - omega
    hyw = hy + omega

    # Check the cases
    C1a = l1 != 0 and l2 != 0
    C2a = l1 != 0 or l2 != 0
    C3a = (l1 == 0 and l2 == 0) and (hxw == 0 and hyw == 0)
    C3b = (l1 == 0 and l2 == 0) and (vy == 0 and hyw == 0)
    C3c = (l1 == 0 and l2 == 0) and (vx == 0 and hxw == 0)
    C4a = (l1 == 0 and l2 == 0) and (vx != 0 and hyw != 0)
    C4b = (l1 == 0 and l2 == 0) and (vy != 0 and hxw != 0)

    if C1a:
        id_case = 0  # C1
    elif C2a:
        id_case = 1  # C2
    elif C3a:
        id_case = 2  # C3
    elif C3b:
        id_case = 3  # C3
    elif C3c:
        id_case = 4  # C3
    elif C4a:
        id_case = 5  # C4
    elif C4b:
        id_case = 6  # C4
    else:
        id_case = None

    # Print the information
    labels = [
        "C1",
        "C2",
        "C3 (hxw == 0 and hyw == 0)",
        "C3 (vy == 0, hyw == 0)",
        "C3 (vx == 0, hxw == 0)",
        "C4 (vx != 0, hyw != 0)",
        "C4 (vy != 0, hxw != 0)",
    ]

    if debug:
        print("------------- ->", labels[id_case])
        print("l+ =", l1, " | l- =", l2)
        print("hx - w =", hxw, " | hy + w =", hyw)
        print("vx =", vx, " | vy =", vy)
        print("------------- ")
    else:
        return id_case


def get_pt_parallel(
    sim_fr: AffineComplexSimulator, params: list[float], alphas: list[float]
):
    ## Get data
    vx, vy, a, omega, hx, hy = params

    n = sim_fr.n
    kappa = sim_fr.kappa

    sigma1 = np.sqrt((hx - omega) * (hy + omega) + 0j)
    sigma2 = -np.sqrt((hx - omega) * (hy + omega) + 0j)
    l1 = kappa * (a + sigma1)
    l2 = kappa * (a + sigma2)

    hxw = hx - omega
    hyw = hy + omega

    ps = sim_fr.p_star_c
    ps_x = np.real(ps)
    ps_y = np.imag(ps)

    ## Calculate the eigenvectors (Prop. 2)
    id_case = check_case(sim_fr, params)
    a1, a2, a3 = alphas

    # C1
    if id_case == 0:
        gamma1 = (vx * hxw + vy * sigma1) / (l1 / kappa)
        gamma2 = (vx * hxw + vy * sigma2) / (l2 / kappa)
        xl1 = gamma1 * np.ones(n) + hyw * ps_x + sigma1 * ps_y
        xl2 = gamma2 * np.ones(n) + hyw * ps_x + sigma2 * ps_y
        return (
            lambda t: a1 * np.ones(n)
            + a2 * xl1 * np.exp(l1 * t)
            + a3 * xl2 * np.exp(l2 * t)
        )

    # C2
    elif id_case == 1:
        if l1 != 0:
            l = l1
            gamma1 = (vx * hxw + vy * sigma1) / (l1 / kappa)
            xl = gamma1 * np.ones(n) + hyw * ps_x + sigma1 * ps_y
        else:
            l = l2
            gamma2 = (vx * hxw + vy * sigma2) / (l2 / kappa)
            xl = gamma2 * np.ones(n) + hyw * ps_x + sigma2 * ps_y

        x01 = hyw * ps_x - a * ps_y
        x02 = (vx * hyw - vy * a) * np.ones(n)
        return lambda t: a1 * x01 + a2 * (x02 + x01 * t) + a3 * xl * np.exp(l * t)

    # C3
    elif id_case in [2, 3, 4]:
        if id_case == 2:  # "C3 (hxw == 0 and hyw == 0)"
            y01 = vy * ps_x - vy * ps_y
            y02 = (vx + 1j * vy) * np.ones(n)
        elif id_case == 3:  # "C3 (vy == 0, hyw == 0)"
            y01 = hx * np.ones(n) - vx * ps_y
            y02 = vx * np.ones(n) + hx * ps_y
        elif id_case == 4:  # "C3 (vx == 0, hxw == 0)"
            y01 = 1j * (hy * np.ones(n) - vy * ps_x)
            y02 = 1j * (vy * np.ones(n) + hy * ps_x)
        return lambda t: a1 * y01 * a2 * y02 + a3 * (ps + y02 * t)

    # C4
    elif id_case in [5, 6]:
        if id_case == 5:  # "C4 (vx != 0, hyw != 0)"
            z01 = 1j * vx * hy * np.ones(n)
            z02 = vx * np.ones(n) + 1j * hy * ps_x
        elif id_case == 6:  # "C4 (vy != 0, hxw != 0)"
            z01 = vy * hx * np.ones(n)
            z02 = 1j * vy * np.ones(n) + hx * ps_y
        return (
            lambda t: a1 * z01 + a2 * (z02 + z01 * t) + a3 * (ps + z02 * t + z01 * t**2)
        )

    else:
        return None


# -------------------------------------------------------------------------------------
