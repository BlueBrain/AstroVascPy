import logging

import numpy.testing as npt
import pandas as pd
import pytest

from astrovascpy import bloodflow as tested
from astrovascpy.utils import Graph

L = logging.getLogger(__name__)


@pytest.fixture
def params():
    return {
        "blood_viscosity": 0.1,
        "base_pressure": 1.33e-3,
        "max_nb_inputs": 3,
        "depth_ratio": 0.05,
        "vasc_axis": 1,
        "threshold_r": 3,
        "max_r_capill": 1.38,
        "t_2_max_capill": 2.7,
        "max_r_artery": 1.23,
        "t_2_max_artery": 3.3,
    }


def test_simple_graph(params):
    point_properties = pd.DataFrame(
        {
            "x": [0, 0, 0, 0, 0],
            "y": [0, 1, 2, 3, 4],
            "z": [0, 0, 0, 0, 0],
            "diameter": [10, 9, 9, 8, 7],
        }
    )
    edge_properties = pd.DataFrame(
        {
            "start_node": [0, 1, 2, 3],
            "end_node": [1, 2, 3, 4],
            "type": [0, 0, 0, 0],
        },
        index=pd.MultiIndex.from_tuples(
            ([0, 0], [0, 1], [0, 2], [0, 3]),
            names=["section_id", "segment_id"],
        ),
    )
    graph = Graph(point_properties, edge_properties)
    entry_nodes = [0]
    input_flow = [1.0]
    boundary_flow = tested.boundary_flows_A_based(graph, entry_nodes, input_flow)
    tested.update_static_flow_pressure(graph, boundary_flow, params)
    normal_flow = graph.edge_properties["flow"].copy()
    normal_pressure = graph.node_properties["pressure"].copy()

    npt.assert_allclose(
        normal_flow.to_list(),
        [1.0, 1.0, 1.0, 1.0],
    )
    npt.assert_allclose(
        normal_pressure.to_list(),
        [
            0.005867,
            0.005104,
            0.004185,
            0.003064,
            0.00133,
        ],
        rtol=1e-6,
        atol=1e-6,
    )

    graph.edge_properties.loc[(0, 2), "radius"] *= 1.2
    tested.update_static_flow_pressure(graph, boundary_flow, params)
    vasodilated_flow = graph.edge_properties["flow"]
    vasodilated_pressure = graph.node_properties["pressure"]

    npt.assert_allclose(
        vasodilated_flow.to_list(),
        [1.0, 1.0, 1.0, 1.0],
    )
    npt.assert_allclose(
        vasodilated_pressure.to_list(),
        [
            0.005342,
            0.00458,
            0.00366,
            0.003064,
            0.00133,
        ],
        rtol=1e-6,
        atol=1e-6,
    )

    assert (vasodilated_flow >= normal_flow - 1e-10).all()  # added 1e-10 for numerical errors
    assert (vasodilated_pressure - 1e-10 <= normal_pressure).all()


def test_bifurcation(params):
    point_properties = pd.DataFrame(
        {
            "x": [0, 0, 0, 0],
            "y": [0, 1, 2, 2],
            "z": [1, 1, 0, 2],
            "diameter": [12, 6, 10, 8],
        }
    )

    edge_properties = pd.DataFrame(
        {
            "start_node": [0, 1, 1],
            "end_node": [1, 2, 3],
            "type": [0, 0, 0],
        },
        index=pd.MultiIndex.from_tuples(
            ([0, 0], [0, 1], [0, 2]),
            names=["section_id", "segment_id"],
        ),
    )
    graph = Graph(point_properties, edge_properties)
    entry_nodes = [0]
    input_flow = [1.0]
    boundary_flow = tested.boundary_flows_A_based(graph, entry_nodes, input_flow)
    tested.update_static_flow_pressure(graph, boundary_flow, params)
    normal_flow = graph.edge_properties["flow"].copy()
    normal_pressure = graph.node_properties["pressure"].copy()
    npt.assert_allclose(normal_flow.to_list(), [1.0, 0.609756, 0.390244], rtol=1e-6, atol=1e-6)
    npt.assert_allclose(
        normal_pressure.to_list(),
        [0.003469, 0.00255, 0.001356, 0.00133],
        rtol=1e-6,
        atol=1e-6,
    )

    graph.edge_properties["radius"] *= 1.2
    tested.update_static_flow_pressure(graph, boundary_flow, params)
    vasodilated_flow = graph.edge_properties["flow"]
    vasodilated_pressure = graph.node_properties["pressure"]

    npt.assert_allclose(vasodilated_flow.to_list(), [1.0, 0.609756, 0.390244], rtol=1e-6, atol=1e-6)
    npt.assert_allclose(
        vasodilated_pressure.to_list(),
        [0.002464, 0.001975, 0.001341, 0.00133],
        rtol=1e-6,
        atol=1e-6,
    )

    assert (vasodilated_flow > normal_flow - 1e-10).all()  # added 1e-10 for numerical errors
    assert (
        vasodilated_pressure - 1e-10 <= normal_pressure
    ).all()  # added 1e-10 for numerical errors


def test_loop(params):
    """This example is from F. Schmidt Phd thesis, figure 9.1A"""
    point_properties = pd.DataFrame(
        {
            "x": [0, 1, 2, 3, 2, 3, 4, 5],
            "y": [0, 0, 1, 1, -1, -1, 0, 0],
            "z": [0, 0, 0, 0, 0, 0, 0, 0],
            "diameter": [1, 1, 1, 1, 1, 1, 1, 1],
        }
    )

    edge_properties = pd.DataFrame(
        {
            "start_node": [0, 1, 2, 3, 1, 4, 5, 6],
            "end_node": [1, 2, 3, 6, 4, 5, 6, 7],
            "type": [0, 0, 0, 0, 0, 0, 0, 0],
        },
        index=pd.MultiIndex.from_tuples(
            ([0, 0], [1, 0], [1, 1], [1, 2], [2, 0], [2, 1], [2, 2], [3, 0]),
            names=["section_id", "segment_id"],
        ),
    )

    graph = Graph(point_properties, edge_properties)
    entry_nodes = [0]
    input_flow = [1.0]
    boundary_flow = tested.boundary_flows_A_based(graph, entry_nodes, input_flow)
    tested.update_static_flow_pressure(graph, boundary_flow, params)
    normal_flow = graph.edge_properties["flow"].copy()
    normal_pressure = graph.node_properties["pressure"].copy()
    npt.assert_allclose(
        normal_flow.to_list(),
        [
            1.0,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            0.5,
            1.0,
        ],
        rtol=1e-6,
        atol=1e-6,
    )
    npt.assert_allclose(
        normal_pressure.to_list(),
        [
            4.327605e02,
            3.221995e02,
            2.440211e02,
            1.887407e02,
            2.440211e02,
            1.887407e02,
            1.105623e02,
            1.330000e-03,
        ],
        rtol=1e-6,
        atol=1e-6,
    )

    graph.edge_properties.loc[(1, 1), "radius"] *= 1.2
    tested.update_static_flow_pressure(graph, boundary_flow, params)
    vasodilated_flow = graph.edge_properties["flow"]
    vasodilated_pressure = graph.node_properties["pressure"]
    npt.assert_allclose(
        vasodilated_flow.to_list(),
        [
            1.0,
            0.545135,
            0.545135,
            0.545135,
            0.454865,
            0.454865,
            0.454865,
            1.0,
        ],
        rtol=1e-6,
        atol=1e-6,
    )
    npt.assert_allclose(
        vasodilated_pressure.to_list(),
        [
            4.136561e02,
            3.030952e02,
            2.178597e02,
            1.957978e02,
            2.319739e02,
            1.816836e02,
            1.105623e02,
            1.330000e-03,
        ],
        rtol=1e-6,
        atol=1e-6,
    )

    # flow increases in active branch (pressure decreases)
    assert vasodilated_flow[(1, 1)] > normal_flow[(1, 1)]
    assert vasodilated_pressure[2] < normal_pressure[2]

    # flow decreases in passive branch (pressure increases)
    assert vasodilated_flow[(2, 1)] < normal_flow[(2, 1)]
    assert vasodilated_pressure[4] < normal_pressure[4]
