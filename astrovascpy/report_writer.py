"""Function to save flows, pressures and radii report in sonata format.
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

from collections import namedtuple
from shutil import copyfile

import h5py
import numpy as np


def write_simulation_report(
    node_ids, report_folder, start_time, end_time, time_step, flows, pressures, radii, volumes
):
    """Write simulation report in sonata format.

    Args:
        node_ids (numpy.array): id of each edge of the vasculature.
        report_folder (path): folder containing the 3 sonata reports.
        start_time (float): beginning of simulation.
        end_time (float): end of simulation.
        time_step (float): time step of simulation.
        flows (numpy.array): flow values at each time-step for each edge.
        pressures (numpy.array): pressure values at each time-step for each node.
        radii (numpy.array): radius values at each time-step for each edge.
        volumes (numpy.array): volume values at each time-step for each edge.
    """
    Report = namedtuple("Report", ["data", "name", "unit"])

    flows = Report(data=flows, unit="µm^3.s^-1", name=report_folder / "report_flows.h5")
    pressures = Report(
        data=pressures, unit="g.µm^-1.s^-2", name=report_folder / "report_pressures.h5"
    )
    radii = Report(data=radii, unit="µm", name=report_folder / "report_radii.h5")
    volumes = Report(data=volumes, unit="µm^3", name=report_folder / "report_volumes.h5")
    for report in [flows, pressures, radii, volumes]:
        write_report(report, node_ids, start_time, end_time, time_step)


def write_report(report, node_ids, start_time, end_time, time_step):
    """Write simulation report in sonata format.

    Args:
        report (Report): folder containing the 3 sonata reports.
        node_ids (numpy.array): id of each edge of the vasculature.
        start_time (float): beginning of simulation.
        end_time (float): end of simulation.
        time_step (float): time step of simulation.
    """
    index_pointers = np.arange(node_ids.size + 1, dtype=np.uint64)

    element_ids = np.zeros(node_ids.size, dtype=np.uint64)

    string_dtype = h5py.special_dtype(vlen=str)

    with h5py.File(report.name, "w") as fd:
        report_group = fd.create_group("report/vasculature")
        report_data = report_group.create_dataset("data", data=report.data, dtype=np.float32)
        report_data.attrs.create("units", report.unit, dtype=string_dtype)
        gmapping = fd.create_group("/report/vasculature/mapping")
        dnodes = gmapping.create_dataset("node_ids", data=node_ids, dtype=np.uint64)
        dnodes.attrs.create("sorted", data=True, dtype=np.uint8)
        gmapping.create_dataset("index_pointers", data=index_pointers, dtype=np.uint64)
        gmapping.create_dataset("element_ids", data=element_ids, dtype=np.uint32)
        dtimes = gmapping.create_dataset(
            "time", data=(start_time, end_time, time_step), dtype=np.double
        )
        dtimes.attrs.create("units", data="s", dtype=string_dtype)


def write_merged_report(
    input_filename,
    report_folder,
    subgraphs,
    types,
    entry_nodes,
    edges_bifurcations,
    pairs=None,
):  # pragma: no cover
    """Write a combined report in sonata format.

    Args:
        input_filename (path): folder containing the sonata graph.
        report_folder (path): folder containing the sonata report.
        subgraphs (numpy.array): (nb_edges, ) ids of subgraphs group for big vessels.
        types (numpy.array): (vessels_type,) array of vessels' type id.
        entry_nodes (numpy.array): (nb_entry_nodes,) ids of entry nodes for the inflow.
        edges_bifurcations (numpy.array): (nb_edges, ) ids of edges forming bifurcations.
        pairs (numpy.array): (nb_pairs, ) ids of nodes forming pairs of nodes.
    """
    string_dtype = h5py.special_dtype(vlen=str)

    report_name = report_folder / "report_vasculature.h5"
    copyfile(input_filename, report_name)

    with h5py.File(report_name, "a") as fd:
        report_group = fd["nodes/vasculature/0"]
        del report_group["type"]
        report_data = report_group.create_dataset("subgraph_id", data=subgraphs, dtype=np.uint64)
        report_data.attrs.create(
            "description",
            data="ids of subgraphs group for big vessels",
            dtype=string_dtype,
        )
        report_data = report_group.create_dataset("type", data=types, dtype=np.uint64)
        report_data.attrs.create("description", data="types", dtype=string_dtype)
        report_data = report_group.create_dataset("entry_edges", data=entry_nodes, dtype=np.uint64)
        report_data.attrs.create(
            "description", data="ids of entry edges for the inflow", dtype=string_dtype
        )
        report_data = report_group.create_dataset(
            "edges_bifurcations", data=edges_bifurcations, dtype=np.uint64
        )
        report_data.attrs.create(
            "description",
            data="ids of exit edges given a bifurcation",
            dtype=string_dtype,
        )
        if pairs is not None:
            report_data = report_group.create_dataset("pairs", data=pairs, dtype=np.uint64)
            report_data.attrs.create("description", data="ids of pairs", dtype=string_dtype)
