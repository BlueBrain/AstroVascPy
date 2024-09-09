from pathlib import Path

import numpy as np
import numpy.testing as npt

from astrovascpy.io import load_graph, load_graph_from_bin, load_graph_from_csv, load_graph_from_h5
from astrovascpy.PetscBinaryIO import get_conf

TEST_DIR = Path(__file__).resolve().parent
TEST_DATA_DIR = TEST_DIR / "data"


def test_petsc_installation():
    precision, indices, complex = get_conf()
    assert (
        complex is False
    ), "PETSc needs to be compiled with configure option --with-scalar-type=real"


def test_load_graph():
    graph = load_graph(TEST_DATA_DIR / "dataset.h5")
    assert graph.edge_properties.shape == (116, 10)
    assert list(graph.edge_properties) == [
        "start_node",
        "end_node",
        "type",
        "section_id",
        "segment_id",
        "length",
        "radius",
        "radius_origin",
        "endfeet_id",
        "volume",
    ]
    npt.assert_array_equal(graph.edge_properties["endfeet_id"], np.full((116,), -1))
    assert graph.node_properties.shape == (114, 4)
    assert list(graph.node_properties) == ["x", "y", "z", "diameter"]
    assert graph.n_nodes == 114
    assert graph.n_edges == 116


def test_load_graph_from_h5():
    graph = load_graph_from_h5(TEST_DATA_DIR / "toy_graph.h5")
    assert graph.edge_properties.shape == (584, 10)
    assert list(graph.edge_properties) == [
        "start_node",
        "end_node",
        "type",
        "section_id",
        "segment_id",
        "length",
        "radius",
        "radius_origin",
        "endfeet_id",
        "volume",
    ]
    npt.assert_array_equal(graph.edge_properties["endfeet_id"], np.full((584,), -1))
    assert graph.node_properties.shape == (585, 4)
    assert list(graph.node_properties) == ["x", "y", "z", "diameter"]
    assert graph.n_nodes == 585
    assert graph.n_edges == 584


def test_load_graph_from_csv():
    node_dataset = TEST_DATA_DIR / "node_dataset.csv"
    edge_dataset = TEST_DATA_DIR / "edge_dataset.csv"
    graph = load_graph_from_csv(node_filename=node_dataset, edge_filename=edge_dataset)
    assert graph.edge_properties.shape == (584, 10)
    assert list(graph.edge_properties) == [
        "start_node",
        "end_node",
        "type",
        "section_id",
        "segment_id",
        "length",
        "radius",
        "radius_origin",
        "endfeet_id",
        "volume",
    ]
    npt.assert_array_equal(np.count_nonzero(graph.edge_properties["endfeet_id"] != -1), 281)
    assert graph.node_properties.shape == (585, 4)
    assert list(graph.node_properties) == ["x", "y", "z", "diameter"]
    assert graph.n_nodes == 585
    assert graph.n_edges == 584


def test_load_graph_from_bin():
    graph = load_graph_from_bin(TEST_DATA_DIR / "toy_graph.bin")
    assert graph.edge_properties.shape == (584, 10)
    assert list(graph.edge_properties) == [
        "start_node",
        "end_node",
        "type",
        "section_id",
        "segment_id",
        "length",
        "radius",
        "radius_origin",
        "endfeet_id",
        "volume",
    ]
    npt.assert_array_equal(np.count_nonzero(graph.edge_properties["endfeet_id"] != -1), 281)
    assert graph.node_properties.shape == (585, 4)
    assert list(graph.node_properties) == ["x", "y", "z", "diameter"]
    assert graph.n_nodes == 585
    assert graph.n_edges == 584
