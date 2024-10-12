"""\
# Copyright (C) 2024 Jesús Bautista Villar <jesbauti20@gmail.com>
"""

import os
import sys
from tqdm import tqdm

import numpy as np
from numpy import linalg as LA

# Graphic tools
import matplotlib.pyplot as plt
from matplotlib.legend import Legend

# Animation tools
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.patches as patches

# -------------------------------------------------------------------------------------

# Swarm Systems Lab PySimUtils
from ssl_pysimutils import createDir, set_paper_parameters, config_data_axis
from ssl_pysimutils import unit_vec

# Python project directory to path (to import afc)
file_path = os.path.dirname(__file__)
module_path = os.path.join(file_path, "..")
if module_path not in sys.path:
    sys.path.append(module_path)

# Affine Formation Control simulators
from afc.simulators import ComplexSimulator
from afc.simulators import AffineSimulator
from afc.simulators import AffineComplexSimulator

# Affine Formation Control utils
from afc.utils import check_case, get_pt_parallel

# -------------------------------------------------------------------------------------

# Set Matplotlib parameters for happy submission
set_paper_parameters(fontsize=16)

if __name__ == "__main__":
    print("All dependencies are correctly installed!")
