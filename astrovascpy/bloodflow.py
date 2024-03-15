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

import gc
import logging
import os
import textwrap
import warnings
from functools import partial

import numpy as np
import numpy.typing as npt
import pandas as pd
from petsc4py import PETSc
from scipy import sparse
from scipy.sparse import linalg
from scipy.stats import randint
from tqdm import tqdm

from . import ou
from .exceptions import BloodFlowError
from .scipy_petsc_conversions import (
    PETScVec2array,
    array2PETScVec,
    coomatrix2PETScMat,
    distribute_array,
)
from .typing import VasculatureParams
from .utils import Graph, comm, find_neighbors, mpi_mem, mpi_timer, rank, rank0, size

# PETSc is compiled with complex number support
# -> many warnings from/to PETSc to/from NumPy/SciPy
warnings.filterwarnings(action="ignore", category=np.ComplexWarning)

print = partial(print, flush=True)


# pylint: disable=protected-access


L = logging.getLogger(__name__)


def compute_static_laplacian(graph, blood_viscosity, with_hematocrit=True):
    """Compute the time-independent Laplacian.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.
        blood_viscosity (float): plasma viscosity in g.µm^-1.s^-1
        with_hematocrit (bool): consider hematrocrit for resistance model

    Returns:
        scipy.sparse.csc_matrix: laplacian matrix
    """
    radii = graph.edge_properties.radius.to_numpy()
    lengths = graph.edge_properties.length.to_numpy()
    resistances = lengths * compute_edge_resistances(
        radii, blood_viscosity, with_hematocrit=with_hematocrit
    )
    adjacency = sparse.csr_matrix(
        (1.0 / resistances, (graph.edges[:, 0], graph.edges[:, 1])),
        shape=(graph.n_nodes, graph.n_nodes),
        dtype=np.float64,
    )
    return sparse.csgraph.laplacian(adjacency + adjacency.T)


def update_static_flow_pressure(
    graph: Graph,
    input_flow: npt.NDArray[np.float64],
    params: VasculatureParams,
    with_hematocrit: bool = True,
):
    """Compute the time-independent pressure and flow.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.
        input_flow(numpy.array): input flow for each graph node.
        params (dict): general parameters for vasculature.
        with_hematocrit (bool): consider hematrocrit for resistance model

    Concerns: This function is part of the public API. Any change of signature
    or functional behavior may be done thoroughly.

    """

    if graph is not None:
        if not isinstance(graph, Graph):
            raise BloodFlowError("'graph' parameter must be an instance of Graph")
        for param in VasculatureParams.__annotations__:
            if param not in params:
                raise BloodFlowError(f"Missing parameter '{param}'")
        blood_viscosity = params["blood_viscosity"]
        base_pressure = params["base_pressure"]

    if graph is not None:
        entry_flow = input_flow[input_flow > 0]
        exit_flow = input_flow[input_flow < 0]
        tot_flow_en = np.abs(np.sum(entry_flow))
        tot_flow_ex = np.abs(np.sum(exit_flow))
        if not np.isclose(tot_flow_ex, tot_flow_en, rtol=1e-6, atol=1e-6):
            raise BloodFlowError(
                f"Boundary flows should sum to 0, but we have {tot_flow_ex-tot_flow_en}"
            )

    laplacian = (
        compute_static_laplacian(
            graph, blood_viscosity=blood_viscosity, with_hematocrit=with_hematocrit
        )
        if graph
        else None
    )

    if rank0():
        cc_mask = graph.cc_mask
        degrees = graph.degrees
        laplacian_cc = laplacian.tocsc()[cc_mask, :][:, cc_mask]
        input_flow = input_flow[cc_mask]
    else:
        laplacian_cc = None

    solution = _solve_linear(laplacian_cc if graph else None, input_flow, params)

    if graph is not None:
        pressure = np.zeros(shape=graph.n_nodes)
        pressure[cc_mask] = solution
        pressure += base_pressure - np.min(pressure[(degrees == 1) & cc_mask])

        incidence = construct_static_incidence_matrix(graph)
        radii = graph.edge_properties.radius.to_numpy()
        lengths = graph.edge_properties.length.to_numpy()
        resistances = lengths * compute_edge_resistances(
            radii, blood_viscosity, with_hematocrit=with_hematocrit
        )
        pressure_to_flow = sparse.diags(1.0 / resistances).dot(incidence)
        graph.edge_properties["flow"] = pressure_to_flow.dot(pressure)
        graph.node_properties["pressure"] = pressure


def compute_edge_resistances(radii, blood_viscosity, with_hematocrit=True):
    """Compute the resistances as a function of radii.

    Args:
        radii (numpy.array): (nb_edges, ) radii of each edge (units: µm).
        blood_viscosity (float): 1.2e-6, standard value of the plasma viscosity (g.µm^-1.s^-1).
        Should be between [0,1].
        with_hematocrit (bool): consider hematrocrit for resistance model

    Returns:
        float: resistances' values per edge.

    Raises:
        BloodFlowError: if blood_viscosity < 0 or >= 1.
    """
    if blood_viscosity < 0:
        raise BloodFlowError("The blood_viscosity must be >= 0.")
    if blood_viscosity >= 1:
        raise BloodFlowError("The blood_viscosity must be < 1")

    resistances = 8.0 * blood_viscosity / (np.pi * radii**4)

    if with_hematocrit:
        resistances *= 4 * (1.0 - 0.863 * np.exp(-radii / 14.3) + 27.5 * np.exp(-radii / 0.351))

    return resistances


def set_edge_resistances(graph, blood_viscosity, with_hematocrit=True):
    """Set the edge resistances on graph.edge_properties.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.
        blood_viscosity (float): 1.2e-6 , standard value of the plasma viscosity (g.µm^-1.s^-1).
        with_hematocrit (bool): consider hematrocrit for resistance model.
    """
    graph.edge_properties["resistances"] = compute_edge_resistances(
        graph.edge_properties.radius.to_numpy(),
        blood_viscosity,
        with_hematocrit=with_hematocrit,
    )


def set_endfeet_ids(graph, edge_ids, endfeet_ids):
    """Set endfeet ids to graph.edge_properties.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.
        edge_ids (pandas IndexSlice): is the corresponding id for each edge.
        endfeet_ids (numpy.array): (nb_endfeet_ids,) is the corresponding endfeet ids.

    Raises:
        BloodFlowError: if edge_ids and endfeet_ids don't have the same size.
    """
    if len(edge_ids) != len(endfeet_ids):
        raise BloodFlowError("edge_ids and endfeet_ids should have the same size.")
    try:
        graph.edge_properties.loc[edge_ids, "endfeet_id"] = endfeet_ids
    except KeyError:
        L.warning("edge_ids do not correspond to any graph edge")


def generate_endfeet(graph, endfeet_coverage, seed):
    """Generates endfeet ids on randomly selected edges
    Args:
        graph (utils.Graph).
        endfeet_coverage (float): Percentage of edges connected with endfeet.
        seed (int): random number generator seed.
    """
    if endfeet_coverage < 0 or endfeet_coverage > 1:
        raise BloodFlowError("endfeet_coverage must be between 0 and 1")
    if graph is not None:
        check = np.all(graph.edge_properties.endfeet_id == -1)
        if not check:
            print("There are already endfeet set")
        else:
            print("Generating random endfeet ids", flush=True)
            n_edges = len(graph.edges)
            n_endfeet = int(np.round(endfeet_coverage * n_edges))
            np.random.seed(seed=seed)
            edge_endfeet_id = randint.rvs(0, n_edges, size=n_endfeet)
            set_endfeet_ids(graph, edge_ids=edge_endfeet_id, endfeet_ids=range(n_endfeet))


def get_radii_at_endfeet(graph):
    """Get the radii at endfeet.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.

    Returns:
        DataFrame: (endfeet_id, radius) pandas dataframe with endfeet_id and corresponding radius.
    """
    return graph.edge_properties.loc[
        graph.edge_properties.endfeet_id != -1, ["endfeet_id", "radius"]
    ]


def get_radius_at_endfoot(graph, endfoot_id):
    """Get the radius at endfoot.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.
        endfoot_id (int): is the corresponding endfoot id.

    Returns:
        Float: corresponding radius.

    Raises:
        BloodFlowError: if endfoot_id does not correspond to a real endfoot id in the graph.
    """
    if endfoot_id not in list(graph.edge_properties.endfeet_id.to_numpy()):
        raise BloodFlowError("The endfoot_id must correspond to a real endfoot id in the graph.")
    return graph.edge_properties.loc[
        graph.edge_properties.endfeet_id == endfoot_id, ["radius", "radius_origin"]
    ].to_numpy()


def set_radii_at_endfeet(graph, endfeet_radii):
    """Set radii at endfeet.

    Args:
        graph (utils.Graph): raph containing point vasculature skeleton.
        endfeet_radii (DataFrame): (endfeet_id, radius) pandas dataframe with endfeet_id and
        the corresponding radius.
    """
    graph.edge_properties.loc[endfeet_radii.index, "radius"] = endfeet_radii.radius


def set_radius_at_endfoot(graph, endfoot_id, endfoot_radius):
    """Set radius at endfoot.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.
        endfoot_id (int): is the corresponding endfoot id.
        endfoot_radius (float or numpy.array): corresponding radius.

    Raises:
        BloodFlowError: if endfoot_id does not correspond to a real endfoot id in the graph and
        if endfoot_radius < 0.
    """
    # if endfoot_id not in list(graph.edge_properties.endfeet_id.values):
    #    raise BloodFlowError("The endfoot_id must correspond to a real endfoot id in the graph.")
    # if (np.asarray(endfoot_radius) <= 0).any():
    #    raise BloodFlowError("Please provide endfoot_radius > 0.")

    graph.edge_properties.loc[graph.edge_properties.endfeet_id == endfoot_id, ["radius"]] = (
        endfoot_radius
    )


def set_endfoot_id(graph, endfoot_id, section_id, segment_id, endfeet_length):  # pragma: no cover
    """Set endfoot_id on (section_id, segment_id) edge.

    Explore the graph starting from an edge until endfeet_length is depleted and
    set the explored edges to the endfoot_id.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.
        endfoot_id (int): id of the endfoot.
        segment_id (int): id of the corresponding segment.
        section_id (int): id of the corresponding section.
        endfeet_length (float): is the corresponding endfoot length in µm.
    """
    edge_ids = get_closest_edges((section_id, segment_id, endfeet_length), graph)
    graph.edge_properties.loc[pd.MultiIndex.from_arrays(edge_ids.T), "endfeet_id"] = endfoot_id


def get_closest_edges(args, graph):
    """Get a list of the closest edges ids (section_id, segment_id) from the starting edge.

    Explore the graph starting from an edge until endfeet_length is depleted.

    Args:
        args (tuple): (3,) with
        args[0] being segment_id (int): id of the corresponding segment,
        args[1], section_id (int): id of the corresponding section and
        args[2], endfeet_length (float): is the corresponding endfoot length in µm.
        graph (utils.Graph): graph containing point vasculature skeleton.

    Returns:
        np.ndarray: list of edges close to the original edge id.

    Raises:
        BloodFlowError: if endfoot_id does not correspond to a real endfoot id in the graph.
    """
    section_id, segment_id, endfeet_length = args
    if endfeet_length < 0:
        L.warning("endfeet_length must be > 0.")
    if (section_id, segment_id) not in list(graph.edge_properties.index.values):
        raise BloodFlowError("The endfoot_id must correspond to a real endfoot id in the graph.")

    return _depth_first_search(graph, (section_id, segment_id), endfeet_length)


def _depth_first_search(graph, current_edge, current_distance, visited=None):
    """Traverse graph using Depth-first search and return edge ids close to starting edge.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.
        current_edge (tuple): id of the current edge to process.
        current_distance (float): left distance to be traversed.
        visited (set): list of all edges that have been visited so far.

    Returns:
        np.ndarray: list of edges close to the original edge id.
    """
    if visited is None:
        visited = set()
    sec_id, seg_id = current_edge
    visited.add((sec_id, seg_id))
    edge_length = graph.edge_properties.length[sec_id, seg_id]
    result_ids = np.array([[sec_id, seg_id]])
    if current_distance > edge_length:
        neighbors = graph.edge_properties[find_neighbors(graph, sec_id, seg_id)].index
        for neighbor_sec_id, neighbor_seg_id in neighbors:
            if (neighbor_sec_id, neighbor_seg_id) not in visited:
                ids_list = _depth_first_search(
                    graph,
                    (neighbor_sec_id, neighbor_seg_id),
                    current_distance - edge_length,
                    visited,
                )
                result_ids = np.concatenate((result_ids, ids_list))

    return result_ids


def boundary_flows_A_based(
    graph: Graph, entry_nodes: npt.NDArray[np.float64], input_flows: npt.NDArray[np.float64]
) -> npt.NDArray[np.float64]:
    """Compute the boundary flows on the exit nodes based on their areas.

    Args:
        graph (Graph): graph containing point vasculature skeleton.
        entry_nodes (numpy.array): Ids of the entry nodes.
        input_flows (numpy.array): Flows on the entry nodes.

    Returns:
        np.ndarray: boundary flow vector for every node in the graph.

    Concerns: This function is part of the public API. Any change of signature
    or functional behavior may be done thoroughly.
    """

    if graph is not None:
        if not isinstance(graph, Graph):
            raise BloodFlowError("'graph' parameter must be an instance of Graph")
        degrees = graph.degrees
        cc_mask = graph.cc_mask

        # Compute nodes of degree 1 where blood flows out
        boundary_nodes_mask = (degrees == 1) & cc_mask
        graph_1 = graph.node_properties.loc[boundary_nodes_mask]

        boundary_flows = np.zeros(shape=graph.n_nodes)  # initialize
        boundary_flows[entry_nodes] = input_flows  # set input flow

        areas = np.pi * (graph_1["diameter"] / 2) ** 2
        # set to 0 the area on entry nodes in order to have 0 weight
        areas[entry_nodes] = 0
        tot_area = areas.sum()
        weights = areas / tot_area

        tot_input_flow = np.sum(input_flows)
        boundary_flows[boundary_nodes_mask] -= tot_input_flow * weights
        return boundary_flows
    else:
        return None


def simulate_vasodilation_ou_process(graph, dt, nb_iteration, nb_iteration_noise, params):
    """Simulate vasodilation according to the Ornstein-Uhlenbeck process.

    Args:
        graph (Graph): graph containing point vasculature skeleton.
        dt (float): time-step.
        nb_iteration (int): number of iteration.
        nb_iteration_noise (int): number of time steps with non-zero noise.
        params (dict): general parameters for vasculature.


    Returns:
        np.ndarray: (endfeet_id, time-step, nb_of_edge_per_endfoot) array where each edge
        is linked to the endfoot located at the edge's position.
    """
    radii_at_endfeet = []  # matrix: rows = number of radii, columns = time points

    # constant c for capillaries and arteries
    c_cap = 2.8
    c_art = 2.8

    # Uncomment the following if we want to fit the mean value.
    # Remark:  it is not possible to fit mean value and max value at same time
    # c_cap = np.sqrt(2/np.pi) * (params["max_r_capill"] - 1) / (params["mean_r_capill"] - 1)
    # c_art = np.sqrt(2/np.pi) * (params["max_r_artery"] - 1) / (params["mean_r_artery"] - 1)

    kappa_c, sigma_c = None, None
    kappa_a, sigma_a = None, None

    if graph is not None:
        assert isinstance(graph, Graph)
        ge = graph.edge_properties["radius_origin"]
        # calibrate kappa for capillaries
        # We calibrate only for the first radius.
        try:
            r0_c = ge[ge <= params["threshold_r"]].iloc[0]
            x_max_c = r0_c * (params["max_r_capill"] - 1)
            kappa_c, sigma_c = ou.compute_OU_params(params["t_2_max_capill"], x_max_c, c_cap)
        except IndexError:
            kappa_c = None
        print("kappa for capillaries: ", kappa_c)
        # calibrate kappa for arteries
        try:
            r0_a = ge[ge > params["threshold_r"]].iloc[0]
            x_max_a = r0_a * (params["max_r_artery"] - 1)
            kappa_a, sigma_a = ou.compute_OU_params(params["t_2_max_artery"], x_max_a, c_art)
        except IndexError:
            kappa_a = None
        print("kappa for arteries: ", kappa_a)

    kappa_c = comm().bcast(kappa_c, root=0)
    sigma_c = comm().bcast(sigma_c, root=0)
    if kappa_c is not None:
        sqrt_kappa_c = np.sqrt(2 * kappa_c)
    kappa_a = comm().bcast(kappa_a, root=0)
    sigma_a = comm().bcast(sigma_a, root=0)
    if kappa_a is not None:
        sqrt_kappa_a = np.sqrt(2 * kappa_a)

    if graph is not None:
        radius_origin = graph.edge_properties.loc[:, "radius_origin"].to_numpy()
        endfeet_id = graph.edge_properties.loc[:, "endfeet_id"].to_numpy()

    # Distribute vectors across MPI ranks
    radius_origin = distribute_array(radius_origin if rank0() else None)
    endfeet_id = distribute_array(endfeet_id if rank0() else None)

    seed = 1
    for radius_origin_, endfeet_id_ in zip(radius_origin, endfeet_id):
        if endfeet_id_ == -1:
            continue

        if radius_origin_ <= params["threshold_r"]:
            x_max = radius_origin_ * (params["max_r_capill"] - 1)
            kappa = kappa_c
            sigma = x_max * sqrt_kappa_c / c_cap
        else:
            x_max = radius_origin_ * (params["max_r_artery"] - 1)
            kappa = kappa_a
            sigma = x_max * sqrt_kappa_a / c_art

        radii_process = radius_origin_ + ou.ornstein_uhlenbeck_process(
            kappa, sigma, dt, nb_iteration, nb_iteration_noise, seed
        )
        seed += 1
        radii_at_endfeet.append(radii_process)

    radii_at_endfeet = np.array(radii_at_endfeet)
    # rae : radii at endfeet
    rae_rows = comm().gather(radii_at_endfeet.shape[0], root=0)

    # if there are zero radii affected by endfeet, return None
    zero_radii = False
    if rank0():
        if np.sum(rae_rows) == 0:
            zero_radii = True
    zero_radii = comm().bcast(zero_radii, root=0)
    if zero_radii:
        return None

    rae = np.empty(1)
    if rank0():
        rae = np.empty((np.sum(rae_rows), radii_at_endfeet.shape[1]), dtype=np.float64)
        rae[: rae_rows[0]] = radii_at_endfeet

    for iproc in range(1, size()):
        if rank0():
            i0 = np.sum(rae_rows[:iproc])
            i1 = i0 + rae_rows[iproc]
            comm().Recv(rae[i0:i1], source=iproc)
        elif rank() == iproc:
            comm().Send(radii_at_endfeet, dest=0)

    return rae


def simulate_ou_process(
    graph, entry_nodes, simulation_time, relaxation_start, time_step, entry_speed, params
):
    """Update value according to the reflected Ornstein-Ulenbeck.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.
        params (dict): general parameters for vasculature.
        entry_nodes (numpy.array:): (nb_entry_nodes,) ids of entry_nodes.
        simulation_time (float): total time of the simulation, in seconds.
        relaxation_start (float): time at which the noise is set to zero.
        time_step (float): size of the time-step.
        entry_speed (numpy.array); speed vector on the entry nodes.
        params (dict): general parameters for vasculature.

    Returns:
        tuple of 3 elements:
        - np.ndarray: (nb_iteration, n_edges) flow values at each time-step for each edge,
        - np.ndarray: (nb_iteration, n_nodes) pressure values at each time-step for each edge,
        - np.ndarray: (nb_iteration, n_edges) radius values at each time-step for each edge.
    """

    nb_iteration = round(simulation_time / time_step)
    # nb_iteration_noise = number of time_steps before relaxation starts:
    nb_iteration_noise = round(relaxation_start / time_step)

    # Only rank0() enters here
    if graph is not None:
        # create this df to assign radii fast at each iteration
        end_df = graph.edge_properties[["endfeet_id"]]
        end_df = end_df[end_df.endfeet_id != -1]

    PETSc.Sys.Print("-> simulate_vasodilation_ou_process")
    with (
        mpi_timer.region("simulate_vasodilation_ou_process"),
        mpi_mem.region("simulate_vasodilation_ou_process"),
    ):
        radii = simulate_vasodilation_ou_process(
            graph, time_step, nb_iteration, nb_iteration_noise, params
        )

    if graph is not None:
        flows = np.zeros((nb_iteration, graph.n_edges))
        pressures = np.zeros((nb_iteration, graph.n_nodes))
        radiii = np.zeros((nb_iteration, graph.n_edges))

    time_iterations = range(nb_iteration)
    if rank0():
        time_iterations = tqdm(range(nb_iteration))

    # Compute the edge ids corresponding to input nodes
    if graph is not None:
        input_edge = []
        for node in entry_nodes:
            input_edge.append(np.where(graph.edges == node)[0][0])

    # iteration over time points
    for time_it in time_iterations:
        if graph is not None:
            if radii is not None:
                graph.edge_properties.loc[end_df.index, "radius"] = radii[:, time_it]

            radii_at_entry_edges = graph.edge_properties["radius"].iloc[input_edge].to_numpy()
            input_flows = entry_speed[time_it] * np.pi * radii_at_entry_edges**2
        else:
            input_flows = None

        # Compute nodes of degree 1 where blood flows out
        boundary_flow = boundary_flows_A_based(graph, entry_nodes, input_flows)

        update_static_flow_pressure(graph, boundary_flow, params)

        if graph is not None:
            flows[time_it] = graph.edge_properties["flow"]
            pressures[time_it] = graph.node_properties["pressure"]
            radiii[time_it] = graph.edge_properties["radius"]

    if graph is not None:
        return flows, pressures, radiii
    else:
        return None, None, None


def construct_static_incidence_matrix(graph):
    """Compute the oriented graph opposite and transposed incidence matrix for static computations.

    Args:
        graph (utils.Graph): graph containing point vasculature skeleton.

    Returns:
        scipy.sparse.csc_matrix: returns the opposite and transposed incidence matrix of graph
        (considered as directed graph, -1: incident nodes, 1: out nodes).
    """
    row = np.repeat(np.arange(graph.n_edges), 2)
    col = graph.edges.flatten()
    ones = np.ones(graph.n_edges)
    data = np.dstack([ones, -ones]).flatten()

    return sparse.csc_matrix((data, (row, col)), shape=(graph.n_edges, graph.n_nodes))


def _solve_linear(laplacian, input_flow, params=None):
    """Solve sparse linear problem on the largest connected component only.

    Args:
        laplacian (scipy.sparse.csc_matrix): laplacian matrix associated to the graph.
        input_flow(scipy.sparse.lil_matrix): input flow for each graph node.
        params (dict): general parameters for vasculature.

    Returns:
        scipy.sparse.csc_matrix: frequency dependent laplacian matrix
    """

    WARNING_MSG = textwrap.dedent(
        """\
        Number of nodes = %(n_nodes)s.
        The program can be slow and the result not very accurate.
        It is recommended to use PETSc for a graph with more than 2e6 nodes and
        SciPy for a graph with less than 2e6 nodes.
        The default solver can be selected in setup.sh or by setting the environment
        variable BACKEND_SOLVER_BFS to 'scipy' or 'petsc'.\
    """
    )

    if params is None:
        params = {}

    SOLVER = params.get("solver", "lgmres")  # second argument is default
    MAX_IT = params.get("max_it", 1e3)
    R_TOL = params.get("r_tol", 1e-12)

    if rank0():
        if sparse.issparse(input_flow):
            input_flow = input_flow.toarray()

        n_nodes = np.shape(laplacian)[0]
        result = np.zeros(shape=n_nodes, dtype=laplacian.dtype)
    else:
        result = None

    # in os.getenv() the second argument refers to the default value
    if os.getenv("BACKEND_SOLVER_BFS", "scipy") == "scipy":
        if rank0() and n_nodes > 2e6:
            L.warning(WARNING_MSG, {"n_nodes": n_nodes})

        with (
            mpi_timer.region("Scipy Solver [bloodflow.py]"),
            mpi_mem.region("Scipy Solver [bloodflow.py]"),
        ):
            if rank0():
                with warnings.catch_warnings():
                    warnings.filterwarnings("error")
                    try:
                        result = linalg.spsolve(laplacian, input_flow)
                    except linalg.MatrixRankWarning:
                        # Ad hoc regularization factor
                        factor = laplacian.diagonal().max() * 1e-14
                        if factor < 1e-8:
                            factor = 1e-8
                            L.warning("Diagonal elements are small. Results can be inaccurate.")
                        if factor > 1e3:
                            L.warning("Diagonal elements are big. Results can be inaccurate.")
                        L.warning("We added {:.2e} to diagonal to regularize".format(factor))

                        laplacian += factor * sparse.eye(np.shape(laplacian)[0])
                        result = linalg.spsolve(laplacian, input_flow)

    if os.getenv("BACKEND_SOLVER_BFS", "scipy") == "scipy":
        if bool(int(os.getenv("DEBUG_BFS", "0"))) and (rank0()):
            scipy_res_norm = np.linalg.norm(input_flow - laplacian * result)
            PETSc.Sys.Print(f"-> SciPy residual norm = {scipy_res_norm}")
        return result

    # PETSc-related part!
    if rank0() and n_nodes < 2e6:
        L.warning(WARNING_MSG, {"n_nodes": n_nodes})

    # These containers are distributed across MPI tasks, contrary to the SciPy ones!
    with (
        mpi_timer.region("PETSc containers [bloodflow.py]"),
        mpi_mem.region("PETSc containers [bloodflow.py]"),
    ):
        laplacian_petsc = coomatrix2PETScMat(laplacian if rank0() else [])
        input_flow_petsc = array2PETScVec(input_flow if rank0() else [])
        result_petsc = array2PETScVec(result if rank0() else [])

        # create the nullspace of the laplacian
        one_vec = np.ones(shape=len(input_flow)) if rank0() else None
        null_vec = array2PETScVec(one_vec if rank0() else [])
        null_space = PETSc.NullSpace().create(null_vec)
        laplacian_petsc.setNullSpace(null_space)

    opts = PETSc.Options()
    # solver
    opts["ksp_type"] = SOLVER
    opts["ksp_gmres_restart"] = 100
    # preconditioner
    opts["pc_type"] = "gamg"
    opts["pc_factor_shift_type"] = "NONZERO"
    opts["pc_factor_shift_amount"] = PETSc.DECIDE

    # progress
    if bool(int(os.getenv("VERBOSE_BFS", "0"))):
        opts["ksp_monitor"] = None

    petsc_solver = PETSc.KSP().create(PETSc.COMM_WORLD)
    petsc_solver.setOperators(laplacian_petsc)
    petsc_solver.rtol = R_TOL
    # petsc_solver.atol = 1e-9
    petsc_solver.max_it = MAX_IT
    petsc_solver.setFromOptions()

    PETSc.Sys.Print("-> PETSc Solver [bloodflow.py] : Start")
    with (
        mpi_timer.region("PETSc Solver [bloodflow.py]"),
        mpi_mem.region("PETSc Solver [bloodflow.py]"),
    ):
        petsc_solver.solve(input_flow_petsc, result_petsc)
    PETSc.Sys.Print("-> PETSc Solver [bloodflow.py] : End")

    # convert to numpy array [only in process 0]
    result_petsc_sp = PETScVec2array(result_petsc)
    if rank0():
        if laplacian.dtype != PETSc.ScalarType:
            result_petsc_sp = result_petsc_sp.astype(laplacian.dtype)

    petsc_res_norm = None
    if rank0():
        if petsc_solver.getIterationNumber() == petsc_solver.max_it:
            L.warning(f"Reached maximum number of iteration {petsc_solver.max_it}.")

    if bool(int(os.getenv("DEBUG_BFS", "0"))) and rank0():
        petsc_res_norm = np.linalg.norm(input_flow - laplacian * result_petsc_sp)  # l2 norm
        input_flow_norm = np.linalg.norm(input_flow)
        PETSc.Sys.Print("Laplacian matrix size: ", np.shape(laplacian))
        PETSc.Sys.Print(f"-> PETSc solver = {opts['ksp_type']}, preconditioner = {opts['pc_type']}")
        PETSc.Sys.Print(f"-> PETSc residual l2 norm = {petsc_res_norm}")
        PETSc.Sys.Print(f"-> The norm of input_flow is: {input_flow_norm}")
        PETSc.Sys.Print(
            f"-> Relative norm: ||residuals|| / ||input_flow|| = {petsc_res_norm/input_flow_norm}"
        )
        PETSc.Sys.Print(
            f"-> The KSP preconditioned residual norm is: {petsc_solver.getResidualNorm()}"
        )

    if rank0():
        result = result_petsc_sp

    # Clean-up
    # This clean-up is imperative, otherwise the memory accumulates and
    # after multiple steps the program crashes with an out-of-memory error!
    # The malloc_trim technique that we use in Neurodamus does not work with PETSc,
    # since it leads to SEG FAULT.
    with (
        mpi_timer.region("PETSc Mem Clean-Up [bloodflow.py]"),
        mpi_mem.region("PETSc Mem Clean-Up [bloodflow.py]"),
    ):
        petsc_solver.destroy()
        laplacian_petsc.destroy()
        input_flow_petsc.destroy()
        result_petsc.destroy()
        # Python's garbage collector (explicitly)
        gc.collect()

    return result
