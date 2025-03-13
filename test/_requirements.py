"""
File: _requirements.py
"""

#######################################################################################
# Standard Libraries
import os
import shutil
import sys

#######################################################################################
# Third-Party Libraries
from tqdm import tqdm

# Algebra (Numerical computation)
import numpy as np

# Graphic Tools (Visualization)
import matplotlib
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation

# Animation Tools (For creating animations)
from IPython.display import HTML
from matplotlib.animation import PillowWriter

#######################################################################################
# Import the Swarm Systems Lab simulator
import ssl_simulator

# Import the Main Module of the Project
file_path = os.path.dirname(__file__)
module_path = os.path.join(file_path, "..")
if module_path not in sys.path:
    sys.path.append(module_path)

# Import the main module of the project
import afc

#######################################################################################

if __name__ == "__main__":
    
    print("\n-----------------------------------------")
    print("All dependencies are correctly installed!")
    print("-----------------------------------------\n")

    print("Testing additional dependencies...")
    
    # Test if LaTex is available for matplotlib
    if (shutil.which('latex')):
        fontsize = 12
        matplotlib.rc("font", **{"size": fontsize, "family": "serif"})
        matplotlib.rc("text", **{"usetex": True, "latex.preamble": r"\usepackage{amsmath}"})
        matplotlib.rc("mathtext", **{"fontset": "cm"})

        fig, ax = plt.subplots()
        ax.text(0.5, 0.5, r"Test: $\int_a^b f(x)dx$", fontsize=12, ha='center')
        ax.set_xticks([])
        ax.set_yticks([])

        print("- LaTeX is correctly installed")

    else:
        print("WARNING: LaTeX is not installed or not found in PATH.")
        print("(Linux) Try installing it with: sudo apt install texlive texlive-latex-extra texlive-fonts-recommended dvipng cm-super")

    # Test FFmpeg
    try:
        def animate(i):
            pass
        fig, ax = plt.subplots()
        anim = FuncAnimation(fig, animate, frames=1)
        anim.to_html5_video()
        print("- FFmpeg is correctly installed")

    except:
        print("WARNING: FFmpeg is missing.")
        print("(Linux) Try installing it with: sudo apt install ffmpeg")

    print("\n-------- DONE \n")