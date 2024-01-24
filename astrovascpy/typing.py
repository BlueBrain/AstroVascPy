"""
:pep:`484`-style type annotations for the public API of this package
"""

import enum
import typing


class VasculatureAxis(enum.IntEnum):
    X = 0
    Y = 1
    Z = 2


class VasculatureParams(typing.TypedDict):
    max_nb_inputs: int
    depth_ratio: float
    vasc_axis: VasculatureAxis
    blood_viscosity: float
    p_base: float
