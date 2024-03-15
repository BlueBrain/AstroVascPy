"""
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

import os
import pickle
from pathlib import Path

import pandas as pd
from vascpy import PointVasculature, SectionVasculature

from .exceptions import BloodFlowError
from .utils import Graph, rank0


def load_graph(filename):
    """Load a vasculature from file.

    Args:
        filename (str): vasculature dataset.

    Returns:
        utils.Graph: graph containing point vasculature skeleton.

    Raises:
        BloodFlowError: if the file object identified by filename is not in h5 format.
    """
    if Path(filename).suffix == ".h5":
        pv = SectionVasculature.load(filename).as_point_graph()
        graph = Graph.from_point_vasculature(pv)
        graph.edge_properties.index = pd.MultiIndex.from_frame(
            graph.edge_properties.loc[:, ["section_id", "segment_id"]]
        )
        return graph
    raise BloodFlowError("File object type identified by filename is not supported")


def load_graph_from_bin(filename):
    """
    Loading of a graph from a binary file using pickle.
    Args:
        filename (str): vasculature dataset path.
    Returns:
        utils.Graph: graph containing point vasculature skeleton.
    """
    if rank0():
        if os.path.exists(filename):
            print("Loading graph from binary file using pickle", flush=True)
            filehandler = open(filename, "rb")
            pv = pickle.load(filehandler)
            graph = Graph.from_point_vasculature(pv)
        else:
            raise BloodFlowError("Graph file not found")
        return graph
    else:
        return None


def load_graph_from_h5(filename):
    """
    Loading of a graph from a .h5 using PointVasculature.load_sonata.
    Args:
        filename (str): vasculature dataset path.
    Returns:
        utils.Graph: graph containing point vasculature skeleton.
    """
    if rank0():
        if os.path.exists(filename):
            print("Loading sonata graph using PointVasculature.load_sonata", flush=True)
            pv = PointVasculature.load_sonata(filename)
            graph = Graph.from_point_vasculature(pv)
        else:
            raise BloodFlowError("Graph file not found")
        return graph
    else:
        return None


def load_graph_from_csv(node_filename, edge_filename):
    """
    Loading of node dataset and edge dataset using pandas.
    It creates a PointVasculature graph object.

    Args:
        node_filename (str): node dataset path.
        edge_filename (str): edge dataset path.
    Returns:
        utils.Graph: graph containing point vasculature skeleton.
    """
    if rank0():
        print("Loading csv dataset using pandas", flush=True)
        graph_nodes = pd.read_csv(node_filename)
        graph_edges = pd.read_csv(edge_filename)

        column_entries = ["start_node", "end_node", "type", "section_id", "segment_id"]
        for col in column_entries:
            if col not in graph_edges.columns:
                raise BloodFlowError(f"Missing {col} in columns")

        pv = PointVasculature(graph_nodes, graph_edges)
        graph = Graph.from_point_vasculature(pv)
        return graph
    else:
        return None
