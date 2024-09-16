# Copyright (c) 2023-2024 Blue Brain Project/EPFL
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#     http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""
:pep:`484`-style type annotations for the public API of this package
"""

import enum
from typing import NotRequired, TypedDict


class VasculatureAxis(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2


class VasculatureParams(TypedDict):
    """
    Required parameters used in the model:

    Args:
        vasc_axis: vasculature axis corresponding to x, y, or z. Should be set to 0, 1, or 2.
        depth_ratio: depth along the vasc_axis. This is the portion of the vasculature where there are inputs.
        max_nb_inputs: maximum number of entry nodes where we inject the flow into the vasculature. Should be >= 1.
        blood_viscosity: plasma viscosity in :math:`g\, \mu m^{-1}\, s^{-1}`.
        base_pressure: reference pressure in :math:`g \, \mu m^{-1}\, s^{-2}`. At resting state equal to the external pressure

    (Optional) Stochastic simulation parameters:

    Args:

        entry_noise: Boolean value to enable or disable the endfeet activity on entry nodes.

        threshold_r: radius (µm) threshold. A radius smaller than the threshold is considered a capillary. A radius bigger than the threshold is considered an artery.

        c_cap: constant used in the ROU parameter calibration for capillaries
        c_art: constant used in the ROU parameter calibration for arteries

        max_r_capill: max radius change factor for capillaries.
        t_2_max_capill: time (in seconds) to reach r_max_capill from 0.
        max_r_artery: max radius change factor for arteries.
        t_2_max_artery: time (in seconds) to reach r_max_artery from 0.

    (Optional) PETSc Linear solver parameters:

    Args:
        solver: iterative linear solver used by PETSc
        max_it: maximum number of solver iterations
        r_tol: relative tolerance
    """

    max_nb_inputs: int
    depth_ratio: float
    vasc_axis: VasculatureAxis
    blood_viscosity: float
    base_pressure: float

    entry_noise: NotRequired[bool]
    c_cap: NotRequired[float]
    c_art: NotRequired[float]
    threshold_r: NotRequired[float]
    max_r_capill: NotRequired[float]
    t_2_max_capill: NotRequired[float]
    max_r_artery: NotRequired[float]
    t_2_max_artery: NotRequired[float]

    solver: NotRequired[str]
    max_it: NotRequired[int]
    r_tol: NotRequired[float]
