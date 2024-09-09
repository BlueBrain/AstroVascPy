import logging
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from vascpy import PointVasculature

import astrovascpy.utils as test_module
from astrovascpy.exceptions import BloodFlowError

L = logging.getLogger(__name__)

TEST_DIR = Path(__file__).resolve().parent


@pytest.fixture
def point_properties():
    return pd.DataFrame({"x": [0, 0, 0], "y": [0, 1, 2], "z": [2, 1, 0], "diameter": [1, 3, 2]})


@pytest.fixture
def edge_properties():
    return pd.DataFrame(
        {"start_node": [0, 1], "end_node": [1, 2], "type": [0, 0]},
        index=pd.MultiIndex.from_arrays(([0, 0], [0, 1]), names=["section_id", "segment_id"]),
    )


def test_find_neighbors(point_properties, edge_properties):
    graph = PointVasculature(point_properties, edge_properties)
    section_id = 0
    segment_id = 0
    npt.assert_array_equal(
        test_module.find_neighbors(graph, section_id, segment_id).to_numpy(),
        np.array([False, True]),
    )


def test_find_degrees_of_neighbors(point_properties, edge_properties):
    graph = PointVasculature(point_properties, edge_properties)
    node_id = 0
    neighbours, connected_id, corresponding_degrees = test_module.find_degrees_of_neighbors(
        graph, node_id
    )
    npt.assert_array_equal(
        neighbours,
        np.array([True, False]),
    )
    npt.assert_array_equal(
        np.array(connected_id),
        np.array({0, 1}),
    )
    npt.assert_array_equal(
        corresponding_degrees,
        np.array([1, 2]),
    )


def test_create_entry_largest_nodes(point_properties, edge_properties, caplog):
    graph = test_module.Graph(point_properties, edge_properties)

    with pytest.raises(BloodFlowError):
        test_module.create_entry_largest_nodes(graph, params={"max_nb_inputs": -1.0})
    with pytest.raises(BloodFlowError):
        test_module.create_entry_largest_nodes(graph, params={"depth_ratio": -1.0})
    with pytest.raises(BloodFlowError):
        test_module.create_entry_largest_nodes(graph, params={"vasc_axis": -1.0})

    with caplog.at_level(logging.WARNING):
        assert (
            test_module.create_entry_largest_nodes(
                graph,
                params={
                    "max_nb_inputs": 1,
                    "depth_ratio": 10,
                    "vasc_axis": 1,
                    "blood_viscosity": 0.1,
                    "base_pressure": 1.33e-3,
                },
            )
            == np.array([2])
        ).all()
    assert "'depth_ratio' parameter must be <= 1. Considering depth_ratio = 1." in caplog.text

    with caplog.at_level(logging.WARNING):
        assert (
            test_module.create_entry_largest_nodes(
                graph,
                params={
                    "max_nb_inputs": 1,
                    "depth_ratio": -1,
                    "vasc_axis": 1,
                    "blood_viscosity": 0.1,
                    "base_pressure": 1.33e-3,
                },
            )
            == np.array([2])
        ).all()
    assert "'depth_ratio' parameter must be >= 0. Considering depth_ratio = 0." in caplog.text


def test_get_largest_nodes(point_properties, edge_properties, caplog):
    graph = PointVasculature(point_properties, edge_properties)

    assert (test_module.get_largest_nodes(graph, n_nodes=1) == np.array([2])).all()
    with pytest.raises(BloodFlowError):
        test_module.get_largest_nodes(graph, n_nodes=-1)
    with caplog.at_level(logging.WARNING):
        assert (test_module.get_largest_nodes(graph, depth_ratio=-1) == np.array([2])).all()
    assert "The depth_ratio must be >= 0. Taking depth_ratio = 0." in caplog.text
    with caplog.at_level(logging.WARNING):
        assert (test_module.get_largest_nodes(graph, depth_ratio=10) == np.array([2])).all()
    assert "The depth_ratio must be <= 1. Taking depth_ratio = 1." in caplog.text
    with pytest.raises(BloodFlowError):
        test_module.get_largest_nodes(graph, vasc_axis=-1)
    with pytest.raises(BloodFlowError):
        test_module.get_largest_nodes(graph, vasc_axis=3)


def test_get_large_nodes(point_properties, edge_properties, caplog):
    graph = PointVasculature(point_properties, edge_properties)

    assert (test_module.get_large_nodes(graph, min_radius=2) == np.array([2])).all()
    with pytest.raises(BloodFlowError):
        assert (test_module.get_large_nodes(graph, min_radius=-1) == np.array([2])).all()
    with caplog.at_level(logging.WARNING):
        assert (test_module.get_large_nodes(graph, depth_ratio=10) == np.array([2])).all()
    assert "The depth_ratio must be <= 1. Taking depth_ratio = 1." in caplog.text
    with caplog.at_level(logging.WARNING):
        assert (test_module.get_large_nodes(graph, depth_ratio=-1) == np.array([2])).all()
    assert "The depth_ratio must be >= 0. Taking depth_ratio = 0." in caplog.text
    with pytest.raises(BloodFlowError):
        test_module.get_large_nodes(graph, vasc_axis=-1)
    with pytest.raises(BloodFlowError):
        test_module.get_large_nodes(graph, vasc_axis=3)


def test_compute_edge_data(point_properties, edge_properties):
    # length = [2, 2], radii  = [1, 1.25]
    graph = test_module.Graph(point_properties, edge_properties)
    edge_lengths, edge_radii, edge_volume = graph._compute_edge_data()
    npt.assert_array_equal(edge_lengths, np.array(np.sqrt([2, 2])))
    npt.assert_array_equal(edge_radii, np.array([1, 1.25]))
    npt.assert_array_equal(
        edge_volume, np.array([np.sqrt(2) * np.pi, 1.25**2 * np.sqrt(2) * np.pi])
    )


def test_set_edge_data(point_properties, edge_properties):
    graph = test_module.Graph(point_properties, edge_properties)
    npt.assert_allclose(np.array(np.sqrt([2, 2])), graph.edge_properties.length)
    npt.assert_allclose(np.array([1, 1.25]), graph.edge_properties.radius)
    npt.assert_allclose(
        np.array([np.sqrt(2) * np.pi, 1.25**2 * np.sqrt(2) * np.pi]), graph.edge_properties.volume
    )


def test_is_iterable():
    assert test_module.is_iterable([12, 13])
    assert test_module.is_iterable(np.asarray([12, 13]))
    assert not test_module.is_iterable(12)
    assert not test_module.is_iterable("abc")


def test_ensure_list():
    assert test_module.ensure_list(1) == [1]
    assert test_module.ensure_list([1]) == [1]
    assert test_module.ensure_list(iter([1])) == [1]
    assert test_module.ensure_list((2, 1)) == [2, 1]
    assert test_module.ensure_list("abc") == ["abc"]


def test_ensure_ids():
    res = test_module.ensure_ids(np.array([1, 2, 3], dtype=np.uint64))
    npt.assert_equal(res, np.array([1, 2, 3], dtype=test_module.IDS_DTYPE))
    npt.assert_equal(res.dtype, test_module.IDS_DTYPE)


def test_create_input_speed():
    T = 1
    step = 0.01
    speed = test_module.create_input_speed(T, step, A=1, f=1, C=0)
    N = T / step

    # There are N steps, corresponding to N+1 time points
    assert len(speed) == N + 1

    TEST_DATA_DIR = TEST_DIR / "data/input_flow"
    speed_from_file_1 = test_module.create_input_speed(
        T, step, read_from_file=TEST_DATA_DIR / "sine.csv"
    )
    # There are N steps, corresponding to N+1 time points
    assert len(speed_from_file_1) == N + 1

    # Read from file. Data has 101 time points
    T = 2  # increase T and therefore the number of time steps
    with pytest.raises(BloodFlowError):
        test_module.create_input_speed(T, step, read_from_file=TEST_DATA_DIR / "sine.csv")


@pytest.mark.parametrize("window", [None, 15, 30])
def test_sine_estimation(window):
    T = 1
    N = 1000
    time = np.linspace(0, T, N)
    A = 40
    f = 15
    C = 100
    signal = A * np.sin(2 * np.pi * f * time) + C

    A_est, f_est, C_est = test_module.fit_sine_model(signal=signal, window=window)

    assert np.abs(A - A_est) < 1e-1
    # Sometimes the end part of the sine is considered as a peak.
    # The estimated frequency can differ by 1.
    assert np.abs(f - f_est) <= 1
    assert np.abs(C - C_est) < 1e-1
