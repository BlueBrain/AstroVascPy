import multiprocessing
from joblib import Parallel, delayed, parallel_config
import pickle
from functools import partial
from pathlib import Path

import numpy as np
import pandas as pd
from archngv import NGVCircuit
from tqdm import tqdm

from astrovascpy import bloodflow
from astrovascpy.exceptions import BloodFlowError
from astrovascpy.utils import set_edge_data

import psutil

print = partial(print, flush=True)

def load_graph_archngv_parallel(filename, n_workers, n_astro=None, parallelization_backend="multiprocessing"):
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
    graph = circuit.vasculature.point_graph
    graph.edge_properties.index = pd.MultiIndex.from_frame(
        graph.edge_properties.loc[:, ["section_id", "segment_id"]]
    )
    set_edge_data(graph)
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
                graph.edge_properties.loc[
                    pd.MultiIndex.from_arrays(result_ids.T), "endfeet_id"
                ] = result_endfeet

    if parallelization_backend == "joblib":
        with parallel_config(backend="loky", prefer="processes", n_jobs=n_workers, inner_max_num_threads=1):
            parallel = Parallel(return_as="generator", batch_size="auto")
            parallelized_region = parallel(delayed(worker)(arg) for arg in tqdm(args, total=len(endfoot_ids)))
            
            for result_ids, result_endfeet in zip(
                parallelized_region,
                endfoot_ids
            ):
                # Only the main process executes this part, i.e. as soon as it receives the parallelly generated data
                graph.edge_properties.loc[
                    pd.MultiIndex.from_arrays(result_ids.T), "endfeet_id"
                ] = result_endfeet

    return graph


if __name__ == '__main__':
    n_cores = psutil.cpu_count(logical=False)
    print(f"number of physical CPU cores = {n_cores}")

    print(f"loading circuit : start")
    filename_ngv = "/gpfs/bbp.cscs.ch/project/proj62/scratch/ngv_circuits/20210325"
    graph = load_graph_archngv_parallel(filename_ngv, n_workers=n_cores) # n_astro=50 for debugging (smaller processing needs)
    print(f"loading circuit : finish")

    print(f"pickle graph : start")
    graph_path = "./data/graphs_folder/dumped_graph.bin"
    filehandler = open(graph_path, "wb")
    pickle.dump(graph, filehandler)
    print(f"pickle graph : finish")
