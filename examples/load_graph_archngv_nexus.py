#!/usr/bin/env python
# coding: utf-8

import argparse
import getpass
import glob
import multiprocessing
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
import psutil
from archngv import NGVCircuit
from joblib import Parallel, delayed, parallel_config
from kgforge.core import KnowledgeGraphForge, Resource
from kgforge.specializations.resources import Dataset
from tqdm import tqdm

from astrovascpy import bloodflow
from astrovascpy.exceptions import BloodFlowError
from astrovascpy.utils import Graph


def get_circuit_conf(circuit_name):
    """
    retreive nexus NGV config entry
    param:
    circuit_name(str): the nexus name of the NGV circuit to load"
    """

    TOKEN = getpass.getpass()

    nexus_endpoint = "https://bbp.epfl.ch/nexus/v1"  # production environment

    ORG = "bbp"
    PROJECT = "mmb-neocortical-regions-ngv"

    forge = KnowledgeGraphForge(
        "https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml",
        endpoint=nexus_endpoint,
        bucket=f"{ORG}/{PROJECT}",
        token=TOKEN,
        debug=True,
    )

    p = forge.paths("Dataset")
    resources = forge.search(p.type == "DetailedCircuit", p.name == "NGV O1.v5 (Rat)", limit=30)

    forge.as_dataframe(resources)
    if len(resources) != 1:
        print("There are several NGV circuit with ths name")
        return None
    else:
        circuit = resources[0]
        circuitConfigPath = circuit.circuitConfigPath.url
        circuitConfigPath = circuitConfigPath[len("file://") :]
        print(f"circuitConfigPath: {circuitConfigPath}")

        return circuitConfigPath


def load_graph_archngv_parallel(
    filename, n_workers, n_astro=None, parallelization_backend="multiprocessing"
):
    """Load a vasculature from an NGV circuit.

    Args:
        filename (str): vasculature dataset.
        n_workers (int): number of processes to set endfeet on edges.
        n_astro (int): for testing, if not None, it will reduce the number of astrocytes used
        parallelization_backend (str): Either multiprocessing or joblib

    Returns:
        vasculatureAPI.PointVasculature: graph containing point vasculature skeleton.

    Raises:
        BloodFlowError: if the file object identified by filename is not in h5 format.
    """
    if not Path(filename).exists():
        raise BloodFlowError("File provided does not exist")
    circuit = NGVCircuit(filename)
    pv = circuit.vasculature.point_graph
    graph = Graph.from_point_vasculature(pv)
    graph.edge_properties.index = pd.MultiIndex.from_frame(
        graph.edge_properties.loc[:, ["section_id", "segment_id"]]
    )
    gv_conn = circuit.gliovascular_connectome
    worker = partial(bloodflow.get_closest_edges, graph=graph)

    args = (
        (
            gv_conn.vasculature_sections_segments(endfoot_id).vasculature_section_id.values[0],
            gv_conn.vasculature_sections_segments(endfoot_id).vasculature_segment_id.values[0],
            gv_conn.get(endfoot_id, ["endfoot_compartment_length"]).values[0],
        )
        for astro_id in np.arange(n_astro or circuit.astrocytes.size)
        for endfoot_id in gv_conn.astrocyte_endfeet(astro_id)
    )
    endfoot_ids = [
        endfoot_id
        for astro_id in np.arange(n_astro or circuit.astrocytes.size)
        for endfoot_id in gv_conn.astrocyte_endfeet(astro_id)
    ]

    if parallelization_backend == "multiprocessing":
        with multiprocessing.Pool(n_workers) as pool:
            for result_ids, result_endfeet in zip(
                tqdm(
                    pool.imap(worker, args, chunksize=max(1, int(len(endfoot_ids) / n_workers))),
                    total=len(endfoot_ids),
                ),
                endfoot_ids,
            ):
                # Only the main process executes this part, i.e. as soon as it receives the parallelly generated data
                graph.edge_properties.loc[pd.MultiIndex.from_arrays(result_ids.T), "endfeet_id"] = (
                    result_endfeet
                )

    elif parallelization_backend == "joblib":
        with parallel_config(
            backend="loky", prefer="processes", n_jobs=n_workers, inner_max_num_threads=1
        ):
            parallel = Parallel(return_as="generator", batch_size="auto")
            parallelized_region = parallel(
                delayed(worker)(arg) for arg in tqdm(args, total=len(endfoot_ids))
            )

            for result_ids, result_endfeet in zip(parallelized_region, endfoot_ids):
                # Only the main process executes this part, i.e. as soon as it receives the parallelly generated data
                graph.edge_properties.loc[pd.MultiIndex.from_arrays(result_ids.T), "endfeet_id"] = (
                    result_endfeet
                )

    else:
        raise BloodFlowError(
            f"parallelization_backend={parallelization_backend} invalid option. Use 'joblib' or 'multiprocessing'."
        )

    return graph


def main():
    global print
    print = partial(print, flush=True)

    parser = argparse.ArgumentParser(description="File paths for NGVCircuits and output graph.")
    parser.add_argument("--circuit_name", type=str, required=True, help="NGV circuits nexus name")
    parser.add_argument(
        "--output_graph", type=str, required=True, help="Path to the output graph file"
    )
    args = parser.parse_args()

    circuit_name = args.circuit_name
    # filename_ngv = args.filename_ngv
    filename_ngv = get_circuit_conf(circuit_name)

    output_graph = args.output_graph

    n_cores = psutil.cpu_count(logical=False)
    print(f"number of physical CPU cores = {n_cores}")

    print(f"NGV Circuits file: {filename_ngv}")
    print("loading circuit : start")
    graph = load_graph_archngv_parallel(
        filename_ngv, n_workers=n_cores
    )  # n_astro=50 for debugging (smaller processing needs)
    print("loading circuit : finish")

    print("pickle graph : start")
    filehandler = open(output_graph, "wb")
    pickle.dump(graph, filehandler)
    print("pickle graph : finish")
    print(f"Graph file: {output_graph}")


if __name__ == "__main__":
    main()
