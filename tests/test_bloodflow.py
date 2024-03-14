import logging
import pickle
from pathlib import Path

import numpy as np
import numpy.testing as npt
import pandas as pd
import pytest
from scipy import sparse as sp

from astrovascpy import bloodflow as tested
from astrovascpy import utils
from astrovascpy.exceptions import BloodFlowError

L = logging.getLogger(__name__)


@pytest.fixture
def point_properties():
    return pd.DataFrame({"x": [0, 0, 0], "y": [0, 1, 2], "z": [2, 1, 0], "diameter": [1, 3, 2]})


@pytest.fixture
def edge_properties():
    return pd.DataFrame(
        {"start_node": [0, 1], "end_node": [1, 2], "type": [0, 0]},
        index=pd.MultiIndex.from_arrays(([0, 0], [0, 1]), names=["section_id", "segment_id"]),
    )


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


def test_compute_edge_resistances():
    radii = np.array([1, 1.25])
    resistances = tested.compute_edge_resistances(radii, blood_viscosity=1.2e-6)
    npt.assert_allclose(resistances, np.array([2.18499354e-05, 4.95812750e-06]))
    with pytest.raises(BloodFlowError):
        tested.compute_edge_resistances(radii, blood_viscosity=-1)
    with pytest.raises(BloodFlowError):
        tested.compute_edge_resistances(radii, blood_viscosity=1)


def test_set_edge_resistances(point_properties, edge_properties):
    graph = utils.Graph(point_properties, edge_properties)
    radii = np.array([1, 1.25])
    resistances = tested.compute_edge_resistances(radii, blood_viscosity=1.2e-6)
    tested.set_edge_resistances(graph, blood_viscosity=1.2e-6, with_hematocrit=True)
    npt.assert_allclose(resistances, graph.edge_properties["resistances"])
    with pytest.raises(BloodFlowError):
        tested.set_edge_resistances(graph, blood_viscosity=-1)
    with pytest.raises(BloodFlowError):
        tested.set_edge_resistances(graph, blood_viscosity=1)


def test_set_endfeet_ids(point_properties, edge_properties, caplog):
    graph = utils.Graph(point_properties, edge_properties)
    graph.edge_properties["endfeet_id"] = [np.nan, np.nan]
    edge_ids = [(0, 1)]
    endfeet_ids = [1]
    tested.set_endfeet_ids(graph, edge_ids, endfeet_ids)
    npt.assert_array_equal(
        graph.edge_properties["endfeet_id"].to_numpy(),
        np.array([np.nan, 1.0], dtype=np.float64),
    )
    edge_ids = [(0, 4)]
    with caplog.at_level(logging.WARNING):
        tested.set_endfeet_ids(graph, edge_ids, endfeet_ids)
        npt.assert_array_equal(
            graph.edge_properties["endfeet_id"].to_numpy(),
            np.array([np.nan, 1.0], dtype=np.float64),
        )
    assert "edge_ids do not correspond to any graph edge" in caplog.text
    edge_ids = [(0, 0), (0, 4)]
    with pytest.raises(BloodFlowError):
        tested.set_endfeet_ids(graph, edge_ids, endfeet_ids)


def test_get_radii_at_endfeet(point_properties, edge_properties):
    """Verifying that the function returns a list of pairs (endfeet, radii),
    where endfeet_id are different from -1
    """
    graph = utils.Graph(point_properties, edge_properties)
    edge_ids = [(0, 1)]  # (section, segment)
    endfeet_ids = [2]
    tested.set_endfeet_ids(graph, edge_ids, endfeet_ids)

    actual_radii = tested.get_radii_at_endfeet(graph).values[0, 1]
    expected_radii = 1.25  # radius value inside the graph when endfoot is set
    npt.assert_allclose(actual_radii, expected_radii)


def test_get_radius_at_endfoot(point_properties, edge_properties):
    graph = utils.Graph(point_properties, edge_properties)
    endfoot_id = 1
    section_id = 0
    segment_id = 0
    endfeet_length = 1.0

    tested.set_endfoot_id(graph, endfoot_id, section_id, segment_id, endfeet_length)
    npt.assert_array_equal(
        tested.get_radius_at_endfoot(graph, endfoot_id),
        np.array([[1.0, 1.0]], dtype=np.float64),
    )
    endfoot_id = 10
    with pytest.raises(BloodFlowError):
        tested.get_radius_at_endfoot(graph, endfoot_id)


def test_set_radii_at_endfeet(point_properties, edge_properties):
    """Verify setting of radii where there are endfeet."""
    graph = utils.Graph(point_properties, edge_properties)
    edge_ids = [(0, 1)]
    endfeet_ids = [2]
    tested.set_endfeet_ids(graph, edge_ids, endfeet_ids)
    endfeet_radii = tested.get_radii_at_endfeet(graph)
    endfeet_radii.radius *= 2  # we modify the radius at endfoot
    tested.set_radii_at_endfeet(graph, endfeet_radii)
    # below we select element in [1] since [0] has no endfoot
    npt.assert_array_equal(graph.edge_properties["radius"].values[1], 2.5)


def test_set_radius_at_endfoot(point_properties, edge_properties):
    graph = utils.Graph(point_properties, edge_properties)
    section_id = 0
    segment_id = 0
    endfeet_length = 1
    endfoot_id = 1
    endfoot_radius = 3.5
    tested.set_endfoot_id(graph, endfoot_id, section_id, segment_id, endfeet_length)
    tested.set_radius_at_endfoot(graph, endfoot_id, endfoot_radius)
    npt.assert_array_equal(
        tested.get_radius_at_endfoot(graph, endfoot_id),
        np.array([[3.5, 1.0]], dtype=np.float64),
    )
    endfoot_id = 10


def test_set_endfoot_id(point_properties, edge_properties, caplog):
    graph = utils.Graph(point_properties, edge_properties)
    section_id = 0
    segment_id = 0
    endfeet_id = 1
    endfeet_length = -1
    with caplog.at_level(logging.WARNING):
        tested.set_endfoot_id(graph, endfeet_id, section_id, segment_id, endfeet_length)
        npt.assert_array_equal(
            graph.edge_properties["endfeet_id"].to_numpy(),
            np.array([1.0, -1], dtype=np.float64),
        )
    assert "endfeet_length must be > 0." in caplog.text
    segment_id = 15
    endfeet_length = 1
    with pytest.raises(BloodFlowError):
        tested.set_endfoot_id(graph, endfeet_id, section_id, segment_id, endfeet_length)
    section_id = 15
    segment_id = 0
    with pytest.raises(BloodFlowError):
        tested.set_endfoot_id(graph, endfeet_id, section_id, segment_id, endfeet_length)
    section_id = 0
    segment_id = 0
    endfeet_length = 3.0
    endfeet_id = 5
    tested.set_endfoot_id(graph, endfeet_id, section_id, segment_id, endfeet_length)
    npt.assert_array_equal(
        graph.edge_properties["endfeet_id"].to_numpy(),
        np.array([5.0, 5.0], dtype=np.float64),
    )


def test_get_closest_edges(point_properties, edge_properties, caplog):
    graph = utils.Graph(point_properties, edge_properties)

    (section_id, segment_id) = (0, 0)
    endfeet_length = -1
    args = section_id, segment_id, endfeet_length
    with caplog.at_level(logging.WARNING):
        npt.assert_array_equal(
            tested.get_closest_edges(args, graph),
            np.array([[section_id, segment_id]]),
        )
    assert "endfeet_length must be > 0." in caplog.text

    (section_id, segment_id) = (113, 250)
    endfeet_length = 1
    args = section_id, segment_id, endfeet_length
    with pytest.raises(BloodFlowError):
        tested.get_closest_edges(args, graph)


def test_simulate_vasodilation_ou_process(point_properties, edge_properties, params):
    graph = utils.Graph(point_properties, edge_properties)
    dt = 0.01
    nb_iteration = 100
    nb_iteration_noise = 2
    max_radius_ratio = 1.2
    section_id = 0
    segment_id = 1
    endfoot_id = 1
    endfeet_length = 1.0

    tested.set_endfoot_id(graph, endfoot_id, section_id, segment_id, endfeet_length)
    radius_origin = tested.get_radius_at_endfoot(graph, endfoot_id)[:, 0]
    radii = tested.simulate_vasodilation_ou_process(
        graph, dt, nb_iteration, nb_iteration_noise, params
    )
    # probability that the radii process crosses the max radius ratio
    proba_rad = np.count_nonzero(radii[0] >= radius_origin[0] * max_radius_ratio) / radii[0].size
    # we test the shape of the radii to be (number of edges with endfeet, time points)
    # time points = nb_iteration + 1
    assert radii.shape == (1, nb_iteration + 1)
    assert (radius_origin == radii[0, 0]).all()
    # we consider 5% probability as a benchmark
    assert proba_rad < 0.05


def test_simulate_ou_process(point_properties, edge_properties, params):
    graph = utils.Graph(point_properties, edge_properties)

    dt = 0.01
    nb_iteration = 100
    entry_nodes = np.array([0])
    simulation_time = nb_iteration * dt
    relaxation_start = 1.0
    time_step = dt
    section_id = 0
    segment_id = 1
    endfoot_id = 1
    endfeet_length = 1.0
    entry_speed = [1] * nb_iteration

    tested.set_endfoot_id(graph, endfoot_id, section_id, segment_id, endfeet_length)
    flows, pressures, radiii = tested.simulate_ou_process(
        graph, entry_nodes, simulation_time, relaxation_start, time_step, entry_speed, params
    )
    assert radiii.shape == (nb_iteration, 2)
    assert flows.shape == (nb_iteration, 2)
    assert pressures.shape == (nb_iteration, 3)


def test_depth_first_search():
    point_properties = pd.DataFrame(
        {
            "x": [0, 0, 0, 0, 0, 0, 1, 1, 1],
            "y": [0, 0, 0, 0, 0, 0, 0, 0, 0],
            "z": [0, 1, 2, 3, 5, 6, 1, 2, 3],
            "diameter": [10, 11, 9, 11, 4, 10, 5, 2, 2],
        }
    )

    edge_properties = pd.DataFrame(
        {
            "start_node": [0, 1, 2, 3, 4, 6, 7],
            "end_node": [1, 2, 3, 4, 5, 7, 8],
            "type": [0, 0, 0, 0, 0, 0, 0],
        },
        index=pd.MultiIndex.from_tuples(
            ([0, 0], [0, 1], [0, 2], [0, 3], [0, 4], [1, 0], [1, 1]),
            names=["section_id", "segment_id"],
        ),
    )
    graph = utils.Graph(point_properties, edge_properties)
    current_edge = (0, 2)
    current_distance = 0.5
    results = tested._depth_first_search(graph, current_edge, current_distance)
    npt.assert_array_equal(results, np.array([[0, 2]]))
    current_distance = 1.5
    results = tested._depth_first_search(graph, current_edge, current_distance)
    npt.assert_array_equal(results, np.array([[0, 2], [0, 1], [0, 3]]))
    current_distance = 2.5
    results = tested._depth_first_search(graph, current_edge, current_distance)
    npt.assert_array_equal(results, np.array([[0, 2], [0, 1], [0, 0], [0, 3]]))


def test_compute_static_laplacian():
    point_properties = pd.DataFrame(
        {
            "x": [0, 0, 0, 0, 0, 0, 0],
            "y": [0, 0, 0, 1, 1, 1, 0],
            "z": [0, 1, 2, 0, 1, 2, 3],
            "diameter": [10, 11, 9, 11, 4, 10, 5],
        }
    )

    edge_properties = pd.DataFrame(
        {
            "start_node": [0, 1, 3, 4, 6],
            "end_node": [1, 6, 4, 5, 2],
            "type": [0, 0, 0, 0, 0],
        },
        index=pd.MultiIndex.from_tuples(
            ([0, 0], [0, 1], [1, 0], [1, 1], [0, 2]),
            names=["section_id", "segment_id"],
        ),
    )
    graph = utils.Graph(point_properties, edge_properties)

    laplacian = tested.compute_static_laplacian(graph, blood_viscosity=1.2e-6).toarray()

    npt.assert_allclose(
        laplacian,
        np.array(
            [
                [
                    1.54532853e08,
                    -1.54532853e08,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    -1.54532853e08,
                    1.84634661e08,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    -3.01018080e07,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    3.77004143e07,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    -3.77004143e07,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    4.80507980e07,
                    -4.80507980e07,
                    0.00000000e00,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    -4.80507980e07,
                    8.57512122e07,
                    -3.77004143e07,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    -3.77004143e07,
                    3.77004143e07,
                    0.00000000e00,
                ],
                [
                    0.00000000e00,
                    -3.01018080e07,
                    -3.77004143e07,
                    0.00000000e00,
                    0.00000000e00,
                    0.00000000e00,
                    6.78022223e07,
                ],
            ]
        ),
    )


def test_update_static_flow_pressure(params):
    point_properties = pd.DataFrame(
        {
            "x": [0, 0, 0, 0, 0, 0, 0, 0],
            "y": [0, 0, 0, 1, 1, 1, 0, 0],
            "z": [0, 1, 2, 0, 1, 2, 3, 4],
            "diameter": [10, 11, 9, 11, 4, 10, 5, 56],
        }
    )

    edge_properties = pd.DataFrame(
        {
            "start_node": [0, 1, 2, 3, 3, 6],
            "end_node": [1, 2, 3, 4, 5, 7],
            "type": [0, 0, 0, 0, 0, 0],
        },
        index=pd.MultiIndex.from_tuples(
            ([0, 0], [0, 1], [1, 0], [1, 1], [0, 2], [1, 2]),
            names=["section_id", "segment_id"],
        ),
    )
    graph = utils.Graph(point_properties, edge_properties)
    entry_nodes = [0]
    input_flow = len(entry_nodes) * [1.0]
    boundary_flow = tested.boundary_flows_A_based(graph, entry_nodes, input_flow)
    tested.update_static_flow_pressure(graph, boundary_flow, params, with_hematocrit=True)
    npt.assert_allclose(
        graph.edge_properties["flow"],
        np.array([1.0, 1.0, 1.0, 0.137931, 0.862069, 0]),
        rtol=5e-7,
        atol=5e-7,
    )


def test_total_flow_conservation_in_graph(params):
    """Check the total conservation of the flow in a connected graph.
    Here we compute the boundary flows using the area based method.
    """

    TEST_DIR = Path(__file__).resolve().parent.parent
    graph_path_cc = TEST_DIR / "examples/data/graphs_folder/toy_graph.bin"
    filehandler = open(graph_path_cc, "rb")
    pv = pickle.load(filehandler)
    graph = utils.Graph.from_point_vasculature(pv)

    entry_nodes = [123, 144, 499]
    input_flows = [1] * len(entry_nodes)

    dropped_graph = graph.edge_properties.drop(columns=["section_id", "segment_id"]).reset_index()
    df_entry_nodes = dropped_graph[
        (dropped_graph["start_node"].isin(entry_nodes))
        | (dropped_graph["end_node"].isin(entry_nodes))
    ]

    transp_incidence = tested.construct_static_incidence_matrix(graph)
    incidence = transp_incidence.T

    boundary_flows = tested.boundary_flows_A_based(graph, entry_nodes, input_flows)

    npt.assert_allclose(
        np.sum(boundary_flows),
        0.0,
        rtol=1e-2,
        atol=1e-2,
    )

    tested.update_static_flow_pressure(graph, boundary_flows, params, with_hematocrit=True)
    flow = graph.edge_properties["flow"].values

    tot_flow = np.sum(incidence @ flow)

    npt.assert_allclose(
        tot_flow,
        0.0,
        rtol=1e-7,
        atol=1e-7,
    )

    Q0 = flow[df_entry_nodes.index[0]]
    Q1 = flow[df_entry_nodes.index[1]]
    Q2 = flow[df_entry_nodes.index[2]]

    sign = []
    for i in range(len(entry_nodes)):
        if entry_nodes[i] in df_entry_nodes.start_node.values:
            sign.append(1.0)
        elif entry_nodes[i] in df_entry_nodes.end_node.values:
            sign.append(-1.0)
        else:
            BloodFlowError("Entry node is missing")

    tot_flow_on_edges = sign @ np.array([Q0, Q1, Q2])

    npt.assert_allclose(
        np.abs(tot_flow_on_edges),
        np.sum(input_flows),
        rtol=1e-3,
        atol=1e-3,
    )


def test_conservation_flow(params):
    point_properties = pd.DataFrame(
        {
            "x": [0, 0, 0, 0],
            "y": [0, 1, 2, 2],
            "z": [1, 1, 0, 2],
            "diameter": [7.0, 3.0, 10.0, 5.0],
        }
    )

    edge_properties = pd.DataFrame(
        {
            "start_node": [0, 1, 1],
            "end_node": [1, 2, 3],
            "type": [0, 0, 0],
        },
        index=pd.MultiIndex.from_tuples(
            ([0, 0], [1, 0], [2, 0]),
            names=["section_id", "segment_id"],
        ),
    )
    graph = utils.Graph(point_properties, edge_properties)

    entry_nodes = [0]
    input_flow = len(entry_nodes) * [1.0]

    boundary_flow = tested.boundary_flows_A_based(graph, entry_nodes, input_flow)

    tested.update_static_flow_pressure(graph, boundary_flow, params)
    flow = graph.edge_properties["flow"]
    npt.assert_allclose(
        flow[0],
        flow[1] + flow[2],
        rtol=1e-4,
    )


def test_static_construct_incidence_matrix(point_properties, edge_properties):
    graph = utils.Graph(point_properties, edge_properties)
    npt.assert_allclose(
        tested.construct_static_incidence_matrix(graph).toarray(),
        np.array([[1, -1, 0], [0, 1, -1]]),
    )


def test_solve_linear(point_properties, edge_properties):
    graph = utils.Graph(point_properties, edge_properties)
    graph.edge_properties["radius"] = [1, 1.25]
    entry_nodes = [0]
    input_flow = [1.0]
    boundary_flow = tested.boundary_flows_A_based(graph, entry_nodes, input_flow)
    adjacency = sp.csr_matrix(
        (graph.n_edges * [1.0], (graph.edges[:, 0], graph.edges[:, 1])),
        shape=(graph.n_nodes, graph.n_nodes),
        dtype=np.float64,
    )
    adjacency = adjacency + adjacency.T
    laplacian = sp.csgraph.laplacian(adjacency)
    pressure = tested._solve_linear(laplacian.tocsc(), boundary_flow)

    # The solution of the system is composed by a solution + a constant
    target_pressure = np.array([1.0, 0.0, -1.0])  # target solution
    result = target_pressure - pressure  # a constant
    is_const = np.isclose(result, result[0], rtol=1e-6, atol=1e-6)
    assert np.all(is_const)


def test_boundary_flows_A_based(point_properties, edge_properties):
    graph = utils.Graph(point_properties, edge_properties)
    graph.edge_properties["radius"] = [1, 1.25]

    boundary_flow = tested.boundary_flows_A_based(graph, [0], [1])
    npt.assert_allclose(boundary_flow, np.array([1.0, 0.0, -1.0]), rtol=2e-6)
