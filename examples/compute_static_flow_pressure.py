#!/usr/bin/env python3
# coding: utf-8
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
import sys
from functools import partial
from pathlib import Path
from pathlib import PurePath

import matplotlib.colors as c
import matplotlib.pyplot as plt
import numpy as np
import petsc4py
import yaml
from mpi4py import MPI
from petsc4py import PETSc

from astrovascpy import bloodflow
from astrovascpy.io import load_graph_from_bin
from astrovascpy.io import load_graph_from_csv
from astrovascpy.io import load_graph_from_h5
from astrovascpy.report_writer import write_simulation_report
from astrovascpy.utils import create_entry_largest_nodes
from astrovascpy.utils import mpi_mem
from astrovascpy.utils import mpi_timer
from astrovascpy.vtk_io import vtk_writer

petsc4py.init(sys.argv)

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()
PETSc.Sys.Print(f"Number of MPI tasks = {MPI_SIZE}")

print = partial(print, flush=True)

RAT = True
plot_yz = True
plot_xz = True
test_vasodilation = False
save_sonata = False
save_vtk = False

params = yaml.full_load(open(str(PurePath("data/params.yaml"))))

output_path = Path(params["output_folder"])
output_join_path = Path(PurePath(params["output_folder"], "figures"))

if MPI_RANK == 0 and not output_path.exists():
    output_path.mkdir()

PETSc.Sys.Print("loading circuit")

node_dataset = "./data/graphs_folder/node_dataset.csv"
edge_dataset = "./data/graphs_folder/edge_dataset.csv"
graph_sonata = "./data/graphs_folder/toy_graph.h5"
graph_bin = "./data/graphs_folder/toy_graph.bin"

with mpi_timer.region("loading circuit"), mpi_mem.region("loading circuit"):
    graph = load_graph_from_bin(graph_bin)
    # Uncomment the following if you want to import with different methods
    #
    # graph = load_graph_from_csv(node_filename=node_dataset, edge_filename=edge_dataset)
    # graph = load_graph_from_h5(filename=graph_sonata)

PETSc.Sys.Print("compute entry nodes")

with mpi_timer.region("compute entry nodes"), mpi_mem.region("compute entry nodes"):
    entry_nodes = create_entry_largest_nodes(graph, params)

PETSc.Sys.Print("entry nodes: ", entry_nodes)

PETSc.Sys.Print("compute input flow")

with mpi_timer.region("compute input flow"), mpi_mem.region("compute input flow"):
    input_flows = len(entry_nodes) * [1.0] if graph is not None else None
    boundary_flow = bloodflow.boundary_flows_A_based(graph, entry_nodes, input_flows)

PETSc.Sys.Print("end of input flow")

PETSc.Sys.Print("compute static flow")

with mpi_timer.region("compute static flow"), mpi_mem.region("compute static flow"):
    bloodflow.update_static_flow_pressure(graph, boundary_flow)

PETSc.Sys.Print("end of static flow pressure")

largest_nodes = entry_nodes

if RAT and graph is not None:
    SMALL_SIZE = 26
    MEDIUM_SIZE = 32
    BIGGER_SIZE = 48
    plt.rc("font", size=SMALL_SIZE)  # controls default text sizes
    plt.rc("axes", titlesize=BIGGER_SIZE)  # fontsize of the axes title
    plt.rc("axes", labelsize=MEDIUM_SIZE)  # fontsize of the x and y labels
    plt.rc("xtick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("ytick", labelsize=SMALL_SIZE)  # fontsize of the tick labels
    plt.rc("legend", fontsize=SMALL_SIZE)  # legend fontsize
    plt.rc("figure", titlesize=BIGGER_SIZE)  # fontsize of the figure title
    plt.rcParams["xtick.major.pad"] = 9.0
    plt.rcParams["axes.linewidth"] = 2.0
    plt.rcParams["xtick.major.size"] = 7.0 * 3.0
    plt.rcParams["xtick.minor.size"] = 4.0 * 3.0
    plt.rcParams["ytick.major.size"] = 7.0 * 3.0
    plt.rcParams["ytick.minor.size"] = 4.0 * 3.0
    plt.rcParams["xtick.major.width"] = 2.4
    plt.rcParams["xtick.minor.width"] = 1.8
    plt.rcParams["ytick.major.width"] = 2.4
    plt.rcParams["ytick.minor.width"] = 1.8

    if plot_yz:
        figure = plt.figure(figsize=(15, 25))
        positions = graph.points
        locate_positions = positions[graph.degrees == 1, 1:]
        plt.scatter(
            locate_positions[:, 1],
            locate_positions[:, 0],
            s=10,
            color="darkgray",
        )

        print("start plotting  y and z axes")
        colors = [[0, 0, 0, 1] for i in range(len(largest_nodes))]
        print("set colors")
        cols = [
            c.to_rgba("palevioletred"),
            c.to_rgba("mediumseagreen"),
            c.to_rgba("mediumturquoise"),
            c.to_rgba("teal"),
            c.to_rgba("mediumpurple"),
            c.to_rgba("darkorange"),
            c.to_rgba("mediumvioletred"),
            c.to_rgba("lightgreen"),
            c.to_rgba("paleturquoise"),
            c.to_rgba("darkslategrey"),
            c.to_rgba("rebeccapurple"),
            c.to_rgba("orangered"),
        ]
        sizes = [150 for i in range(len(largest_nodes))]
        print("set markers")
        markers = ["o", "s", "^", "p", "H", "D", "o", "s", "^", "p", "H", "D"]
        plt.scatter(
            positions[largest_nodes][:, 2],
            positions[largest_nodes][:, 1],
            s=sizes,
            color=colors,
            marker=markers[2],
        )
        plt.hlines(
            np.max(positions[:, 1]) - (np.max(positions[:, 1]) - np.min(positions[:, 1])) * 1 / 10,
            np.min(positions[:, 2]),
            np.max(positions[:, 2]),
            linewidths=6,
            linestyle="dashed",
            colors="cornflowerblue",
        )
        plt.xlabel("z-position (µm)")
        plt.ylabel("y-position (µm)")
        plt.tight_layout()
        plt.savefig(Path(params["output_folder"]) / "input_node_yz.png", dpi=300)
        print("end plotting y and z axes")

    if plot_xz:

        print("start plotting x and z axes")
        figure = plt.figure(figsize=(15, 15))
        positions = graph.points
        filter_positions = (
            positions[:, 1]
            > np.max(positions[:, 1]) - (np.max(positions[:, 1]) - np.min(positions[:, 1])) * 1 / 10
        )
        locate_positions = positions[filter_positions][:, [0, -1]]
        locate_degrees = graph.degrees[filter_positions]

        plt.scatter(
            locate_positions[locate_degrees == 1, 1],
            locate_positions[locate_degrees == 1, 0],
            s=10,
            color="darkgray",
        )

        colors = [[0, 0, 0, 1] for i in range(len(largest_nodes))]
        cols = [
            c.to_rgba("palevioletred"),
            c.to_rgba("mediumseagreen"),
            c.to_rgba("mediumturquoise"),
            c.to_rgba("teal"),
            c.to_rgba("mediumpurple"),
            c.to_rgba("darkorange"),
            c.to_rgba("mediumvioletred"),
            c.to_rgba("lightgreen"),
            c.to_rgba("paleturquoise"),
            c.to_rgba("darkslategrey"),
            c.to_rgba("rebeccapurple"),
            c.to_rgba("orangered"),
        ]
        sizes = [150 for i in range(len(largest_nodes))]
        markers = ["o", "s", "^", "p", "H", "D", "o", "s", "^", "p", "H", "D"]
        plt.scatter(
            positions[largest_nodes][:, 2],
            positions[largest_nodes][:, 0],
            s=sizes,
            color=colors,
            marker=markers[2],
        )
        plt.xlabel("z-position (µm)")
        plt.ylabel("x-position (µm)")
        plt.tight_layout()
        plt.savefig(Path(params["output_folder"]) / "input_node_xz.png", dpi=300)
        print("end xz plotting")

    if test_vasodilation:
        mask = (
            (500 < graph.node_properties.x)
            & (graph.node_properties.x < 550)
            & (1500 < graph.node_properties.y)
            & (graph.node_properties.y < 1550)
            & (500 < graph.node_properties.z)
            & (graph.node_properties.z < 550)
        )
        edge_index = graph.edge_properties[
            graph.edge_properties.start_node.isin(graph.node_properties[mask].index)
            & graph.edge_properties.end_node.isin(graph.node_properties[mask].index)
        ].index
        graph.edge_properties.loc[edge_index, "radius"] *= 1.5

if graph is not None:
    if save_sonata:
        sonata_path = Path(params["output_folder"]) / "sonata_files"
        if not sonata_path.exists():
            Path.mkdir(sonata_path)
        print("start sonata reporting")
        filename = sonata_path / "simulate_ou_process"

        if not filename.exists():
            Path.mkdir(filename)
        write_simulation_report(
            np.arange(graph.n_edges),
            filename,
            start_time=0.0,
            end_time=1.0,
            time_step=1.0,
            flows=graph.edge_properties["flow"],
            pressures=graph.node_properties["pressure"],
            radii=graph.edge_properties["radius"].to_numpy(),
            volumes=graph.edge_properties["volume"].to_numpy(),
        )
        print("end of sonata reporting")

    if save_vtk:
        vtk_path = Path(params["output_folder"]) / "vtk_files"
        if not vtk_path.exists():
            Path.mkdir(vtk_path)

        flow = graph.edge_properties["flow"]
        print("flow:", flow)
        pressure = graph.node_properties["pressure"]
        flow_vtk = np.copy(flow)
        flow_vtk[flow > 0] = np.log10(flow_vtk[flow > 0])
        flow_vtk[flow < 0] = -np.log10(-flow_vtk[flow < 0])

        pressure_edge = []
        for u, v in graph.edges:
            pressure_edge.append(0.5 * (pressure[u] + pressure[v]))

        properties = {"flow": flow_vtk, "pressure": pressure_edge}

        filename = str(vtk_path / "flow_pressure")
        points = graph.node_properties[["x", "y", "z"]].to_numpy()
        edges = graph.edges
        radii = graph.edge_properties.radius
        types = np.zeros(radii.size)

        print("number of edges:", len(edges))
        print("number of connected nodes:", len(np.unique(edges)))

        vtk_writer(filename, points, edges, radii, types, extra_properties=properties)

mpi_timer.print()
mpi_mem.print()
