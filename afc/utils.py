"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import numpy.linalg as LA

from .simulators import AffineComplexSimulator
from .math import gen_Ni

from typing import List, Set
from numpy.typing import NDArray

## Utils ------------------------------------------------------------------------------

def check_case(
        sim_fr: AffineComplexSimulator, 
        params: List[float], 
        debug: bool =False
    ) -> None:

    vx, vy, ax, ay, omega, hx, hy = params

    kappa = sim_fr.kappa

    # Calculate (hx - omega) and (hy + omega)
    hxw = hx - omega
    hyw = hy + omega

    # Complex conjugate eigen values
    sigma1 =  np.sqrt(((ax - ay)/2)**2 + hxw*hyw + 0j)
    sigma2 = -np.sqrt(((ax - ay)/2)**2 + hxw*hyw + 0j)
    l1 = kappa * ((ax + ay)/2 + sigma1)
    l2 = kappa * ((ax + ay)/2 + sigma2)

    # Check the cases
    C1a = (l1 != 0 and l2 != 0) and (l1 != l2)
    C1b = (l1 != 0 and l2 != 0) and \
          (l1 == l2 and ax == ay and hxw == 0 and hyw == 0)

    C2a = (l1 != 0 and l2 != 0) and ((l1 == l2 and ax == ay) and hxw == 0) 
    C2b = (l1 != 0 and l2 != 0) and ((l1 == l2 and ax == ay) and hyw == 0)

    C3a = ((l1 == 0 and l2 != 0) or (l2 == 0 and l1 != 0)) and \
          (hyw*vx == ax*vy or hxw*vy == ay*vx)

    C4a = ((l1 == 0 and l2 != 0) or (l2 == 0 and l1 != 0)) and \
          (hyw*vx != ax*vy or hxw*vy != ay*vx)

    C5a = (l1 == 0 and l2 == 0) and (hxw == 0 and vx == 0) and hyw != 0
    C5b = (l1 == 0 and l2 == 0) and (hyw == 0 and vy == 0) and hxw != 0
    C5c = (l1 == 0 and l2 == 0) and (hxw == 0 and hyw == 0 and vx != 0)
    C5d = (l1 == 0 and l2 == 0) and (hxw == 0 and hyw == 0 and vy != 0)

    C6a = (l1 == 0 and l2 == 0) and (hxw != 0 and vy != 0)
    C6b = (l1 == 0 and l2 == 0) and (hyw != 0 and vx != 0)
    

    if C1a:
        id_case = 0  # C1a
    elif C1b:
        id_case = 1  # C1b
    elif C2a:
        id_case = 2  # C2a
    elif C2b:
        id_case = 3  # C2b
    elif C3a:
        id_case = 4  # C3
    elif C4a:
        id_case = 5  # C4
    elif C5a:
        id_case = 6  # C5a
    elif C5b:
        id_case = 7  # C5b
    elif C5c:
        id_case = 8  # C5c
    elif C5d:
        id_case = 9  # C5d
    elif C6a:
        id_case = 10 # C6a
    elif C6b:
        id_case = 11 # C6b
    else:
        id_case = None

    # Print the information
    labels = [
        "C1",
        "C1 (l1 == l2 and ax == ay and hxw == 0 and hyw == 0)",
        "C2 (hxw == 0)",
        "C2 (hyw == 0)",
        "C3",
        "C4",
        "C5 (hxw == 0 and vx == 0)",
        "C5 (hyw == 0 and vy == 0)",
        "C5 (hxw == 0 and hyw == 0 and vx != 0)",
        "C5 (hxw == 0 and hyw == 0 and vy != 0)",
        "C6 (hxw != 0 and vy != 0)",
        "C6 (hyw != 0 and vx != 0)",
    ]

    if debug:
        Av_star = np.array([[0, vx, vy], [0, ax, hyw], [0, hxw, ay]])
        print("------------- ->", labels[id_case])
        print("l+ =", l1, " | l- =", l2)
        print("Av* =\n", Av_star)
        print("ax*ay - hxw*hyw =", ax*ay - hxw*hyw)
        print("------------- ")
    else:
        return id_case


def get_pt_parallel(
    sim_fr: AffineComplexSimulator, 
    params: List[float], 
    alphas: List[float]
) -> NDArray:
    
    ## Get data
    vx, vy, ax, ay, omega, hx, hy = params

    n = sim_fr.n
    kappa = sim_fr.kappa

    # Calculate (hx - omega) and (hy + omega)
    hxw = hx - omega
    hyw = hy + omega

    # Complex conjugate eigen values
    sigma1 =  np.sqrt(((ax - ay)/2)**2 + hxw*hyw + 0j)
    sigma2 = -np.sqrt(((ax - ay)/2)**2 + hxw*hyw + 0j)
    l1 = kappa * ((ax + ay)/2 + sigma1)
    l2 = kappa * ((ax + ay)/2 + sigma2)

    ps = sim_fr.p_star_c
    ps_x = np.real(ps)
    ps_y = np.imag(ps)

    ## Calculate the eigenvectors (Prop. 2)
    id_case = check_case(sim_fr, params)
    a1, a2, a3 = alphas
    
    # C1
    if id_case in [0,1]:
        if id_case == 0:
            c3_l1 = l1/kappa - ax
            c3_l2 = l2/kappa - ax
            c2_l1 = hyw
            c2_l2 = hyw
            gamma1 = (vx * c2_l1 + vy*c3_l1) / (l1 / kappa)
            gamma2 = (vx * c2_l2 + vy*c3_l2) / (l2 / kappa)
            xl1 = gamma1 * np.ones(n) + c2_l1 * ps_x + c3_l1*ps_y
            xl2 = gamma2 * np.ones(n) + c2_l2 * ps_x + c3_l2*ps_y
        
        if id_case == 1:
            xl1 = vx * np.ones(n) + (ax + ay)/2 * ps_x
            xl2 = vy * np.ones(n) + (ax + ay)/2 * ps_y

        return (
            lambda t: a1 * np.ones(n)
            + a2 * xl1 * np.exp(l1 * t)
            + a3 * xl2 * np.exp(l2 * t)
        ) 
    
    # C2
    if id_case in [2,3]:
        l = l1
        axy = (ax + ay) / 2

        if id_case == 2:
            xl_1 = kappa * (vx * np.ones(n) + axy * ps_x)
            xl_2 = vy/hyw * np.ones(n) +  ps_x +  axy/hyw * ps_y

        if id_case == 3:
            xl_1 = kappa * (vy * np.ones(n) + axy * ps_y)
            xl_2 = vx/hxw * np.ones(n) + axy/hxw * ps_x +  ps_y

        return (
            lambda t: a1 * np.ones(n)
            + (a2 * xl_1 + a3 * (xl_2 + xl_1*t)) * np.exp(l * t)
        )

    # C3
    if id_case == 4:
        if l1 != 0:
            l = l1
        else:
            l = l2
        
        x0 = vy * ps_x - vx * ps_y

        c3_l = l/kappa - ax
        gamma = (vx * hyw + vy*c3_l) / (l / kappa)
        xl = gamma * np.ones(n) + hyw * ps_x + c3_l*ps_y

        return (
            lambda t: a1 * np.ones(n)
            + a2 * x0 
            + a3 * xl * np.exp(l * t)
        )

    # C4
    if id_case == 5:
        if l1 != 0:
            l = l1
        else:
            l = l2
        
        x0_1 = kappa * (vx + 1j * vy) * np.ones(n)
        x0_1 = - kappa * (ax*vy - vx*(hy+omega)) * np.ones(n)
        x0_2 = hyw * ps_x - ax * ps_y

        c3_l = l/kappa - ax
        gamma = (vx * hyw + vy*c3_l) / (l / kappa)
        xl = gamma * np.ones(n) + hyw * ps_x + c3_l*ps_y

        return (
            lambda t: a1 * x0_1 
            + a2 * (x0_2 + x0_1 * t)
            + a3 * xl * np.exp(l * t)
        )

    # C5
    if id_case in [6,7,8,9]:
        if id_case == 6:
            x0_1 = kappa * (vy * np.ones(n) + (hy + omega) * ps_x)
            x0_2 = ps_y
            y0 = (hy + omega) * np.ones(n) - vy * ps_x
        if id_case == 7:
            x0_1 = kappa * (vx * np.ones(n) + (hx - omega) * ps_y)
            x0_2 = ps_x
            y0 = (hx - omega) * np.ones(n) - vx * ps_y
        if id_case == 8 or id_case == 9:
            x0_1 = kappa * (vx + vy) * np.ones(n)
            x0_2 = vx * ps_x + vy * ps_y
            y0 = vy * ps_x - vx * ps_y

        return (
            lambda t: a1 * x0_1 
            + a2 * (x0_2 + x0_1 * t)
            + a3 * y0
        )

    # C6
    if id_case in [10,11]:
        if id_case == 10:
            x0_1 = kappa**2 * vy**2 * (hx - omega) * np.ones(n)
            x0_2 = kappa * (vx*vy * np.ones(n) + vy * (hx - omega) * ps_y)
            x0_3 = vy * ps_x
        if id_case == 11:
            x0_1 = kappa**2 * vx**2 * (hy + omega) * np.ones(n)
            x0_2 = kappa * (vx*vy * np.ones(n) + vx * (hy + omega) * ps_x)
            x0_3 = vx * ps_y

        return (
            lambda t: a1 * x0_1 
            + a2 * (x0_2 + x0_1 * t)
            + a3 * (x0_3 + x0_2 * t + x0_1 * t**2 / 2)
        )

    else:
        return None


def gen_MBt(
    Z: Set[Set[int]],  # Each set contains 2 int elements (head, tail)
    p_star: NDArray[np.complex_], # 1D array with shape n = to the number of agents
    v_star_coords: List[complex]  # c1, c2 and c3 w.r.t. O_\Delta
) -> NDArray:
    
    # Init
    n = np.shape(p_star)[0]
    MBt = np.zeros((n,n))

    c1,c2,c3 = v_star_coords

    # Calculate every [M @ Bt]_i row (= mu_i) individually (Ax = b problem)
    for i in range(n):
        Ni = gen_Ni(i,n,Z)

        # Generate "A"
        z_star = np.zeros(len(Ni), dtype=complex)
        for idx,j in enumerate(Ni):
            z_star[idx] = p_star[i] - p_star[j]

        # Generate "b"
        vi = c1 + c2 * np.real(p_star)[i] + c3 * np.imag(p_star)[i]

        # Calculate x = \tilde mu_i by solving Ax = b with least squares.
        # The system Ax = b may be under-, well-, or over-determined.
        z_star_real = np.array([np.real(z_star), np.imag(z_star)])
        vi_real = np.array([np.real(vi), np.imag(vi)])
        mu_i_bar, _, _, _ = LA.lstsq(z_star_real, vi_real, rcond=None)
        
        # Generate mu_i
        mu_i = np.zeros(n)
        for idx, jn in enumerate(Ni):
            mu_i[jn] = mu_i_bar[idx]
        MBt[i,:] = mu_i
    
    return MBt
            

# -------------------------------------------------------------------------------------
