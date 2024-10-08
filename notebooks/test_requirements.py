import os
import sys
from tqdm import tqdm

import numpy as np
from numpy import linalg as LA

# Graphic tools
import matplotlib.pyplot as plt
from seaborn import color_palette

# Animation tools
from matplotlib.animation import FuncAnimation
from IPython.display import HTML
import matplotlib.patches as patches

# --------------------------------------------------------------------------------------

# Swarm Systems Lab PySimUtils
from ssl_pysimutils import createDir

# Python project directory to path
file_path = os.path.dirname(__file__)
module_path = os.path.join(file_path, "..")
if module_path not in sys.path:
    sys.path.append(module_path)

# Formations control simulation frames
from afc.sf_complex import sim_frame_complex
from afc.sf_affine import sim_frame_affine
from afc.sf_affine_complex import sim_frame_affine_complex
