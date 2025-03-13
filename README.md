# Leaderless Collective Motion in Affine Formation Control over the Complex Plane

We propose a method for the collective maneuvering of affine formations in the plane by modifying the original weights of the Laplacian matrix used to achieve static formations. Specifically, the resulting collective motion is characterized as a time-varying affine transformation of a reference configuration, or shape. Unlike the traditional leader-follower strategy, our leaderless scheme allows agents to maintain distinct and possibly time-varying velocities, enabling a broader range of collective motions, including all the linear combinations of translations, rotations, scaling and shearing of a reference shape. Our analysis provides the analytic solution governing the resulting collective motion, explicitly designing the eigenvectors and eigenvalues that define this motion as a function of the modified weights in the new Laplacian matrix. To facilitate a more tractable analysis and design of affine formations in 2D, we propose the use of complex numbers to represent all relevant information.

```
@misc{jbautista2025collectivemotionafc,
  title={Leaderless Collective Motion in Affine Formation Control over the Complex Plane}, 
  author={Jesus Bautista, Enric Morella, Lili Wang, Hector Garcia de Marina},
  year={2025},
  url={}, 
}
```

## Installation

To install the required dependencies, simply run:

```bash
python install.py
```

### Additional Dependencies
Some additional dependencies, such as LaTeX fonts and FFmpeg, may be required. We recommend following the installation instructions provided in the ```ssl_simulator``` [README](https://github.com/Swarm-Systems-Lab/ssl_simulator/blob/master/README.md). 

To verify that all additional dependencies are correctly installed on Linux, run:
```bash
bash test/test_dep.sh
```

## Usage

For an overview of the project's structure and to see the code in action, we recommend running the Jupyter notebooks located in the `notebooks` directory.

## Credits

If you have any questions, open an issue or reach out to the maintainers:

- **[Jesús Bautista Villar](https://sites.google.com/view/jbautista-research)** (<jesbauti20@gmail.com>) – Main Developer
