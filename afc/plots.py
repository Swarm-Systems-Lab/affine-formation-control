"""
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import numpy as np
import matplotlib.pyplot as plt

## Plots ------------------------------------------------------------------------------


def plot_xy(xdata: np.ndarray, ydata: np.ndarray, Z: list, ax_input=None, lim=20):
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
    colors = ["k", "b", "r", "g"]
    for i in range(xdata.shape[1]):
        ax.plot(xdata[:, i], ydata[:, i], "-", c=colors[i], lw=0.8)
        ax.plot(xdata[0, i], ydata[0, i], "x", c=colors[i])
        ax.plot(xdata[-1, i], ydata[-1, i], ".", c=colors[i], label=(i + 1))

    # Plot the graph connections
    for edge in Z:
        i, j = np.array(edge) - 1
        ax.plot(
            [xdata[-1, i], xdata[-1, j]],
            [ydata[-1, i], ydata[-1, j]],
            "k--",
            lw=0.8,
        )

    if ax_input is None:
        ax.legend(ncols=4, fontsize="xx-small", loc="upper center")


# -------------------------------------------------------------------------------------
