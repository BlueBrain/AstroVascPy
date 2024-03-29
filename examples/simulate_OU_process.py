#!/usr/bin/env python3
# coding: utf-8

import sys
from functools import partial
from pathlib import Path

import numpy as np
import petsc4py
import yaml
from mpi4py import MPI
from petsc4py import PETSc

from astrovascpy.bloodflow import generate_endfeet, simulate_ou_process

# from astrovascpy.io import load_graph_from_bin
# from astrovascpy.io import load_graph_from_h5
from astrovascpy.io import load_graph_from_csv
from astrovascpy.report_writer import write_simulation_report
from astrovascpy.utils import create_entry_largest_nodes, create_input_speed, mpi_mem, mpi_timer

petsc4py.init(sys.argv)

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()
PETSc.Sys.Print(f"Number of MPI tasks = {MPI_SIZE}")


print = partial(print, flush=True)

save_vtk = False
save_sonata = True

curr_dir = Path(__file__).resolve().parent
params = yaml.full_load(open(str(curr_dir / "data/params.yaml")))

output_path = Path(curr_dir / params["output_folder"])

if MPI_RANK == 0 and not output_path.exists():
    output_path.mkdir()

############################################################################################
PETSc.Sys.Print("loading circuit")

node_dataset = curr_dir / "data/graphs_folder/node_dataset.csv"
edge_dataset = curr_dir / "data/graphs_folder/edge_dataset.csv"
graph_sonata = curr_dir / "data/graphs_folder/toy_graph.h5"
graph_bin = curr_dir / "data/graphs_folder/toy_graph.bin"

with mpi_timer.region("loading circuit"), mpi_mem.region("loading circuit"):
    graph = load_graph_from_csv(node_filename=node_dataset, edge_filename=edge_dataset)
    # Uncomment the following if you want to import with different methods
    #
    # graph = load_graph_from_h5(filename=graph_sonata)
    # graph = load_graph_from_bin(graph_bin)

GEN_ENDFEET = True
COVERAGE = 0.7  # percentage of endfeet coverage
if GEN_ENDFEET:
    generate_endfeet(graph, endfeet_coverage=COVERAGE, seed=42)

PETSc.Sys.Print("compute entry nodes")

entry_nodes = create_entry_largest_nodes(graph, params)
PETSc.Sys.Print("entry nodes: ", entry_nodes)


PETSc.Sys.Print("simulate astrovascpy")

simulation_time = 5  # seconds
time_step = 0.01

# Flag to enable the relaxation phase.
# Set 'RELAXATION = True' if you want to stop the radii perturbation (set noise to zero)
# at the time 'relaxation_start'.
# Set relaxation_start = 0 if you don't want noise at all.
RELAXATION = False  # True
if RELAXATION:
    relaxation_start = 3.0  # relaxation starting time
else:
    relaxation_start = simulation_time  # No relaxation

# Flag to enable sine wave input flow
SINE_INFLOW = True
if SINE_INFLOW:
    A = 6119  # Amplitude of the sine wave
else:
    A = 0

# Blood speed on the entry nodes
if graph is not None:
    entry_speed = create_input_speed(
        T=simulation_time, step=time_step, A=A, f=8, C=35000, read_from_file=None
    )
else:
    entry_speed = None

with mpi_timer.region("simulate astrovascpy"), mpi_mem.region("simulate astrovascpy"):
    flows, pressures, radiii = simulate_ou_process(
        graph, entry_nodes, simulation_time, relaxation_start, time_step, entry_speed, params
    )

if graph is not None:
    points = graph.node_properties[["x", "y", "z"]].to_numpy()
    pressures = np.mean(pressures[:, graph.edges], axis=2)
# The pressure operation above is equivalent to:
# pressure_edge = []
# for u, v in graph.edges:
#     pressure_edge.append(0.5 * (pressure[u] + pressure[v]))

############################################################################################

if graph is not None:
    if save_sonata:
        sonata_path = Path(curr_dir / params["output_folder"]) / "sonata_files"
        if not sonata_path.exists():
            Path.mkdir(sonata_path)
        print("start sonata reporting", flush=True)
        filename = (
            sonata_path
            / "simulate_ou_process_2_8_sigma_10_seconds_january_9_3_entry_nodes_relaxation"
        )
        if not filename.exists():
            Path.mkdir(filename)
        write_simulation_report(
            np.arange(graph.n_edges),
            filename,
            0,
            simulation_time,
            time_step,
            flows,
            pressures,
            radiii,
            np.power(radiii, 2) * np.pi * graph.edge_properties["length"].to_numpy(),
        )
        print("end of sonata reporting", flush=True)

    if save_vtk:
        from astrovascpy.vtk_io import vtk_writer

        vtk_path = Path(params["output_folder"]) / "vtk_files"
        if not vtk_path.exists():
            Path.mkdir(vtk_path)

        print("start vtk saving", flush=True)
        filename = vtk_path / "simulate_ou_process"
        if not filename.exists():
            Path.mkdir(filename)

        for i, (flow, pressure, radii) in enumerate(zip(flows, pressures, radiii)):
            vtk_writer(
                str(filename / ("step_" + str(i))),
                points,
                graph.edges,
                radii,
                np.zeros(graph.n_edges),
                extra_properties={"flow": flow, "pressure": pressure, "radii": radii},
            )
        print("end of vtk saving", flush=True)

############################################################################################

mpi_timer.print()
mpi_mem.print()
