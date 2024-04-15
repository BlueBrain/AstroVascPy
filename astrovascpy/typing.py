"""
:pep:`484`-style type annotations for the public API of this package

Copyright (c) 2023-2024 Blue Brain Project/EPFL
Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at
    http://www.apache.org/licenses/LICENSE-2.0
Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
"""

import enum
from typing import TypedDict, NotRequired


class VasculatureAxis(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2


class VasculatureParams(TypedDict):
    max_nb_inputs: int
    depth_ratio: float
    vasc_axis: VasculatureAxis
    blood_viscosity: float
    base_pressure: float

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
