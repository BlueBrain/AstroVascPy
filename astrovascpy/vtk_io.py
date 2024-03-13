"""
file taken from https://github.com/eleftherioszisis/VasculatureRepair.
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

import logging

import numpy as np
import vtk
from vtk.util import numpy_support as _ns

log = logging.getLogger(__name__)

# pylint: disable-all


def vtk_points(points):  # pragma: no cover
    """Convert an array of numpy points to vtk points.

    Args:
        points (np.array): section' points.

    Returns:
        vtkPoints: 3D points, an array of vx-vy-vz triplets accessible by (point or cell) id.
    """
    vpoints = vtk.vtkPoints()
    vpoints.SetData(_ns.numpy_to_vtk(points.copy(), deep=1))
    return vpoints


def vtk_lines(edges):  # pragma: no cover
    """Convert a list of edges into vtk lines.

    Args:
        edges (np.array): edges of the graph.

    Returns:
        vtkCellArray: vtk lines.
    """
    vlines = vtk.vtkCellArray()

    n_edges = edges.shape[0]

    arr = np.empty((n_edges, 3), order="C", dtype=np.int)

    arr[:, 0] = 2 * np.ones(n_edges, dtype=np.int)
    arr[:, 1:] = edges

    # cell array structure: size of cell followed by edges
    # arr = np.column_stack((2 * np.ones(edges.shape[0], dtype=np.int), edges)).copy()

    # crucial to deep copy the data!!!
    vlines.SetCells(edges.shape[0], _ns.numpy_to_vtkIdTypeArray(arr, deep=1))
    return vlines


def vtk_attribute_array(name, arr):  # pragma: no cover
    """Create a cell array with specified name and assigns the numpy array arr.

    Args:
        name (str): name of the cell array.
        arr (np.array): assigned to the cell array.

    Returns:
        vtkCellArray: cell array with specified name and assigns the numpy array arr.
    """
    val_arr = vtk.util.numpy_support.numpy_to_vtk(arr)
    val_arr.SetName(name)

    return val_arr


def create_polydata_from_data(points, edges, attribute_dict={}):  # pragma: no cover
    """Create a PolyData vtk object from a set of points.

    Points are connected with edges and optionally have a set of attributes.

    Args:
        points (np.array): section's points.
        edges (np.array): edges of the graph.
        attribute_dict (dict): to store attributes.

    Returns:
        vtkPolyData: PolyData vtk object.
    """
    polydata = vtk.vtkPolyData()

    polydata.SetPoints(vtk_points(points))
    polydata.SetLines(vtk_lines(edges))

    cell_data = polydata.GetCellData()

    for key, arr in attribute_dict.items():
        cell_data.AddArray(vtk_attribute_array(key, arr))

    return polydata


def vtk_loader(filename):  # pragma: no cover
    """Extract from a vtk file the points, edges, radii and types.

    Args:
        filename (str): name of the file.

    Returns:
        np.array: points, edges and radii.
    """
    from vtk.util.numpy_support import vtk_to_numpy

    def get_points(polydata):  # pragma: no cover
        vtk_points = polydata.GetPoints()
        return vtk_to_numpy(vtk_points.GetData())

    def get_structure(polydata):  # pragma: no cover
        vtk_lines = polydata.GetLines()

        nmp_lines = vtk_to_numpy(vtk_lines.GetData())

        n_rows = int(len(nmp_lines) / 3)

        return nmp_lines.reshape(n_rows, 3)[:, (1, 2)].astype(np.intp)

    def get_radii(polydata):  # pragma: no cover
        cell_data = polydata.GetCellData()

        N = cell_data.GetNumberOfArrays()

        names = [cell_data.GetArrayName(i) for i in range(N)]

        vtk_floats = cell_data.GetArray(names.index("radius"))

        return vtk_to_numpy(vtk_floats)

    def get_types(polydata):  # pragma: no cover
        cell_data = polydata.GetCellData()

        N = cell_data.GetNumberOfArrays()

        names = [cell_data.GetArrayName(i) for i in range(N)]

        vtk_floats = cell_data.GetArray(names.index("types"))

        return vtk_to_numpy(vtk_floats)

    # create a polydata reader
    reader = vtk.vtkPolyDataReader()

    # add the filename that will be read
    reader.SetFileName(filename)

    # update the output of the reader
    reader.Update()

    polydata = reader.GetOutput()

    points = get_points(polydata)
    edges = get_structure(polydata)
    radii = get_radii(polydata)

    # if no types are provided, it will return an array of zeros.
    try:
        types = get_types(polydata)

    except Exception:
        # warnings.warn("Types were not found. Zeros are used instead." + str(exc))
        types = np.zeros(edges.shape[0], dtype=np.int)

    return points, edges, radii, types


def vtk_writer(
    filename, points, edges, radii, types, mode="ascii", extra_properties=None
):  # pragma: no cover
    """Create a vtk legacy file and populate it with a polydata object.

    Polydata object is generated using the points, edges, radii and types.

    Args:
        filename (str): name of the file.
        points (np.array): shape: (n_nodes, 3).
        edges (np.array): shape: (n_edges, 2).
        radii (np.array): shape: (n_edges, 2).
        types (np.array): shape: (n_edges, 2).
        mode (str): ascii or binary
        extra_properties (iterable of callables): add extra property computation functions
        that are not included.
    """
    # from .vtk_io import vtk_points, vtk_lines, vtk_attribute_array
    points = np.ascontiguousarray(points)
    edges = np.ascontiguousarray(edges)
    radii = np.ascontiguousarray(radii)
    types = np.ascontiguousarray(types)

    attr_dict = {"radius": radii, "type": types}

    if extra_properties is not None:
        assert isinstance(extra_properties, dict)
        attr_dict.update(extra_properties)

    polydata = create_polydata_from_data(points, edges, attribute_dict=attr_dict)

    writer = vtk.vtkPolyDataWriter()

    writer.SetFileName(filename + ".vtk")

    if mode == "binary":
        writer.SetFileTypeToBinary()
    elif mode == "ascii":
        writer.SetFileTypeToASCII()

    writer.SetInputData(polydata)
    writer.Write()
