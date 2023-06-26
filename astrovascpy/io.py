"""
Copyright (c) 2023-2023 Blue Brain Project/EPFL
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
import os
import pickle
from pathlib import Path

import pandas as pd
from mpi4py import MPI as mpi
from vascpy import PointVasculature
from vascpy import SectionVasculature

from astrovascpy.exceptions import BloodFlowError
from astrovascpy.utils import get_main_connected_component
from astrovascpy.utils import set_edge_data

MPI_COMM = mpi.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()


def load_graph(filename):
    """Load a vasculature from file.

    Args:
        filename (str): vasculature dataset.

    Returns:
        vasculatureAPI.PointVasculature: graph containing point vasculature skeleton.

    Raises:
        BloodFlowError: if the file object identified by filename is not in h5 format.
    """
    if Path(filename).suffix == ".h5":
        graph = SectionVasculature.load(filename).as_point_graph()
        graph.edge_properties.index = pd.MultiIndex.from_frame(
            graph.edge_properties.loc[:, ["section_id", "segment_id"]]
        )
        set_edge_data(graph)
        return graph
    raise BloodFlowError("File object type identified by filename is not supported")


def load_graph_from_bin(filename, is_cc=False):
    """
    Loading of a graph from a binary file using pickle.
    Args:
        filename (str): vasculature dataset path.
        save_cc (bool): if True the graph is assumed to be fully connected and
        the computation of the main connected component is skipped
    Returns:
        vasculatureAPI.PointVasculature: graph containing point vasculature skeleton.
    """
    if MPI_RANK == 0:
        if os.path.exists(filename):
            print("Loading graph from binary file using pickle", flush=True)
            filehandler = open(filename, "rb")
            graph = pickle.load(filehandler)
            if not is_cc:
                graph = get_main_connected_component(graph)
        else:
            raise BloodFlowError("Graph file not found")
        return graph
    else:
        return None


def load_graph_from_h5(filename, is_cc=False):
    """
    Loading of a graph from a .h5 using PointVasculature.load_sonata.
    Args:
        filename (str): vasculature dataset path.
        save_cc (bool): if True the graph is assumed to be fully connected and
        the computation of the main connected component is skipped
    Returns:
        vasculatureAPI.PointVasculature: graph containing point vasculature skeleton.
    """
    if MPI_RANK == 0:
        if os.path.exists(filename):
            print("Loading sonata graph using PointVasculature.load_sonata", flush=True)
            graph = PointVasculature.load_sonata(filename)
            set_edge_data(graph)
            if not is_cc:
                graph = get_main_connected_component(graph)
        else:
            raise BloodFlowError("Graph file not found")
        return graph
    else:
        return None


def load_graph_from_csv(node_filename, edge_filename, is_cc=False):
    """
    Loading of node dataset and edge dataset using pandas.
    It creates a PointVasculature graph object.

    Args:
        node_filename (str): node dataset path.
        edge_filename (str): edge dataset path.
        save_cc (bool): if True the graph is assumed to be fully connected and
        the computation of the main connected component is skipped
    Returns:
        vasculatureAPI.PointVasculature: graph containing point vasculature skeleton.
    """
    if MPI_RANK == 0:
        print("Loading csv dataset using pandas", flush=True)
        graph_nodes = pd.read_csv(node_filename)
        graph_edges = pd.read_csv(edge_filename)

        column_entries = ["start_node", "end_node", "type", "section_id", "segment_id"]
        for col in column_entries:
            if col not in graph_edges.columns:
                raise BloodFlowError(f"Missing {col} in columns")

        graph = PointVasculature(graph_nodes, graph_edges)
        if not is_cc:
            graph = get_main_connected_component(graph)

        if "endfeet_id" not in graph_edges.columns:
            set_edge_data(graph)

        return graph
    else:
        return None
