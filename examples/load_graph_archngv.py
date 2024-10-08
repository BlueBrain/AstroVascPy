#!/usr/bin/env python
# coding: utf-8

import argparse
import getpass
from os import environ
import glob
import multiprocessing
import pickle
from functools import partial
from pathlib import Path
import base64
import requests

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


def get_nexus_token(
    client_id="bbp-molsys-sa",
    environ_name="KCS",
    nexus_url="https://bbpauth.epfl.ch/auth/realms/BBP/protocol/openid-connect/token",
):
    """
    retreive a Nexus Token from keycloak
    param:
       client_id(str): the keycloak client id
       environ_name(str): the name of the environement variable that holds the keycloak secret
       nexus_url(str)
    """
    try:
        client_secret = environ[environ_name]

        # bbp keycloack token endpoint
        url = nexus_url

        payload = "grant_type=client_credentials&scope=openid"
        authorization = base64.b64encode(
            f"{client_id}:{client_secret}".encode("utf-8")
        ).decode("ascii")
        headers = {
            "Content-Type": "application/x-www-form-urlencoded",
            "Authorization": f"Basic {authorization}",
        }

        # request the token
        r = requests.request(
            "POST",
            url=url,
            headers=headers,
            data=payload,
        )

        # get access token
        mexus_token = r.json()["access_token"]
        return mexus_token
    except Exception as error:
        print(f'Error: {error}')
        return None


def get_nexus_circuit_conf(
    circuit_name, nexus_org="bbp", nexus_project="mmb-neocortical-regions-ngv"
):
    """
    retrieve nexus NGV config entry
    param:
    circuit_name(str): the Nexus name of the NGV circuit to load
    nexus_org(str): The Nexus organisation
    nexus_projec(str): The Nexus project that holds the circuit
    """

    nexus_token = get_nexus_token()
    if nexus_token is None:
        print("Error: Cannot get a valid Nexus token")
        return None

    nexus_endpoint = "https://bbp.epfl.ch/nexus/v1"  # production environment

    forge = KnowledgeGraphForge(
        "https://raw.githubusercontent.com/BlueBrain/nexus-forge/master/examples/notebooks/use-cases/prod-forge-nexus.yml",
        endpoint=nexus_endpoint,
        bucket=f"{nexus_org}/{nexus_project}",
        token=nexus_token,
        debug=True,
    )

    p = forge.paths("Dataset")
    resources = forge.search(
        p.type == "DetailedCircuit", p.name == circuit_name, limit=30
    )

    forge.as_dataframe(resources)
    if len(resources) != 1:
        print("There are several NGV circuit with this name")
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
            gv_conn.vasculature_sections_segments(
                endfoot_id
            ).vasculature_section_id.values[0],
            gv_conn.vasculature_sections_segments(
                endfoot_id
            ).vasculature_segment_id.values[0],
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
                    pool.imap(
                        worker,
                        args,
                        chunksize=max(1, int(len(endfoot_ids) / n_workers)),
                    ),
                    total=len(endfoot_ids),
                ),
                endfoot_ids,
            ):
                # Only the main process executes this part, i.e. as soon as it receives the parallelly generated data
                graph.edge_properties.loc[
                    pd.MultiIndex.from_arrays(result_ids.T), "endfeet_id"
                ] = result_endfeet

    elif parallelization_backend == "joblib":
        with parallel_config(
            backend="loky",
            prefer="processes",
            n_jobs=n_workers,
            inner_max_num_threads=1,
        ):
            parallel = Parallel(return_as="generator", batch_size="auto")
            parallelized_region = parallel(
                delayed(worker)(arg) for arg in tqdm(args, total=len(endfoot_ids))
            )

            for result_ids, result_endfeet in zip(parallelized_region, endfoot_ids):
                # Only the main process executes this part, i.e. as soon as it receives the parallelly generated data
                graph.edge_properties.loc[
                    pd.MultiIndex.from_arrays(result_ids.T), "endfeet_id"
                ] = result_endfeet

    else:
        raise BloodFlowError(
            f"parallelization_backend={parallelization_backend} invalid option. Use 'joblib' or 'multiprocessing'."
        )

    return graph


def main():
    global print
    print = partial(print, flush=True)

    parser = argparse.ArgumentParser(
        description="File paths for NGVCircuits and output graph."
    )
    parser.add_argument(
        "--circuit-name", type=str, required=False, help="NGV circuits nexus name"
    )
    parser.add_argument(
        "--circuit-path", type=str, required=False, help="Path to the NGV circuits"
    )
    parser.add_argument(
        "--output-graph", type=str, required=True, help="Path to the output graph file"
    )
    args = parser.parse_args()

    if args.circuit_name is not None:
        circuit_name = args.circuit_name
        circtui_path = None
        filename_ngv = get_nexus_circuit_conf(circuit_name)
        if filename_ngv == None:
            print('Error: Could not obtain a valid file path for the NGV circuit')
            return -1

    elif args.circuit_path is not None:
        filename_ngv = args.circuit_path
    # filename_ngv = args.filename_ngv

    else:
        print(f"ERROR: circuit-name or circuit-path must be provided")
        return -1

    print(f"INFO: filename_ngv {filename_ngv} ")

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
    with open(output_graph, "wb") as filehandler:
        pickle.dump(graph, filehandler)
        print("pickle graph : finish")
    print(f"Graph file: {output_graph}")


if __name__ == "__main__":
    print('INFO: Start load_graph_archngv.py')
    main()
