"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt

COLORS = ["k", "b", "r", "g"]

## Plots ##############################################################################

__all__ = ["COLORS", "plot_xy"]

def plot_xy(
    xdata: np.ndarray,
    ydata: np.ndarray,
    Z: list,
    ax_input:plt.Axes = None,
    lim: float = 20,
    alpha1: float =1,
    alpha2: float = 1,
    colored: bool = True,
    tail: bool = True,
    lw_net: float = 0.8,
):
    # Create a new figure if an axis is not provided
    if ax_input is None:
        fig = plt.figure()
        ax = fig.subplots()

        ax.set_xlim([-lim, lim])
        ax.set_ylim([-lim, lim])
        ax.set_aspect("equal")
        ax.grid(True)
    else:
        ax = ax_input

    # Plot the XY simulation data

    for i in range(xdata.shape[1]):
        if colored:
            color = COLORS[i % len(COLORS)]
        else:
            color = "royalblue"
        ax.plot(xdata[0, i], ydata[0, i], "x", c=color)
        ax.plot(xdata[-1, i], ydata[-1, i], ".", c=color, label=(i + 1))
        if tail:
            ax.plot(xdata[:, i], ydata[:, i], "-", c=color, lw=0.8, alpha=alpha1)

    # Plot the graph connections
    for edge in Z:
        i, j = np.array(edge) - 1
        ax.plot(
            [xdata[-1, i], xdata[-1, j]],
            [ydata[-1, i], ydata[-1, j]],
            "--",
            c="grey",
            lw=lw_net,
            alpha=alpha2
        )

    if ax_input is None:
        ax.legend(ncols=4, fontsize="xx-small", loc="upper center")


#######################################################################################
