![AstroVascPy Logo](docs/source/logo/BPP-AstroVascPy-Github.jpg)
# AstroVascPy

AstroVascPy is a Python library for computing the blood pressure and flow through the vasculature
(whole cortical column). AstroVascPy incorporates the effect of astrocytic endfeet on the blood vessel radii.
In particular, AstroVascPy replicates the dynamics of the radius of a vessel due to vasodilation.

AstroVascPy uses vascpy Point Graph representation to access the vasculature database stored in h5 file (sonata format).

vascpy standardizes the api for the vasculature datasets.
PointVasculature (PointGraph) representation is basically a composition of two pandas data frames,
one for node properties (x, y, z, radius, other...) and one for edge properties (start_node, end_node, other...).

### Inputs
- pointgraph vasculature
- endfeet locations with corresponding ids
- radius of vessels at endfeet locations (possibly depending on simulation time)

### Outputs

- blood pressure at each node of the vasculature (node vector)
- blood flow at each segment (edge vector)

## Installation (Linux & MacOS)

AstroVascPy can be git cloned here:

    https://github.com/BlueBrain/astrovascpy

Either locally or in BB5, one can run:

    source setup.sh

to install the AstroVascPy solver (+ all its dependencies) and set the environment. For the local installation (workstation), please install **conda** before running the command above.
**Remark**: Run this command every time before using the solver in order to set the environment correctly.

Backend Solvers: `export BACKEND_SOLVER_BFS='petsc'` or `export BACKEND_SOLVER_BFS='scipy'`, the user can choose the backend solver for the linear systems.
**Remark**: PETSc is inherently parallel, while SciPy is not. Therefore, running the Blood Flow Solver with MPI makes sense only while using `petsc`!

Blood Flow Solver (BFS) debugging: By typing `export DEBUG_BFS=1`, we run both PETSc & SciPy, and we compare their results. To disable this behavior please type `export DEBUG_BFS=0` (default behavior).

## Usage

The code can be run using

    python3 compute_static_flow_pressure.py

### Sonata reports

Structure of the reports:
This is a particular type of compartment report for the vasculature.
We get a set of 3 reports at each time-step storing the blood flow,
the blood pressure and, the radius at each segment of the vasculature.
Here are the units of these entities:
-flow (µm^3.s^-1)
-pressure (g.µm^-1.s^-2)
-radius (µm)

## Authors

Stéphanie Battini, Nicola Cantarutti, Christos Kotsalos and Tristan Carel

Link to the article on Bio-arxiv:

## Funding and Acknowledgements

The development of this software was supported by funding to the Blue Brain Project, a research center of the
École polytechnique fédérale de Lausanne (EPFL), from the Swiss government’s ETH Board of the Swiss Federal
Institutes of Technology.

We would like to thank Alessandro Cattabiani, Thomas Delemontex and Eleftherios Zisis
for reviewing the code and the engineering support.

Copyright (c) 2023-2023 Blue Brain Project/EPFL
