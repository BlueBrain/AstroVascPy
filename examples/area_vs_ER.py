import pickle
from pathlib import Path
from pathlib import PurePath

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy as scp
import yaml
from mpi4py import MPI

import astrovascpy.bloodflow as bf
from astrovascpy import entry_nodes as en
from astrovascpy import utils

MPI_COMM = MPI.COMM_WORLD
MPI_RANK = MPI_COMM.Get_rank()
MPI_SIZE = MPI_COMM.Get_size()


if MPI_RANK == 0:
    params = yaml.full_load(open(str(PurePath("data/params.yaml"))))

    graph_path = "./data/graphs_folder/toy_graph.bin"
    filehandler = open(graph_path, "rb")
    graph = pickle.load(filehandler)

    entry_nodes = en.create_entry_largest_nodes(graph, params)
    print("Entry nodes: ", entry_nodes)

    # Compute nodes of degree 1 where blood flows out
    boundary_nodes = np.where(graph.degrees == 1)[0]
    entry0 = entry_nodes[0]
    # entry1 = entry_nodes[1]
    # entry2 = entry_nodes[2]
    b_nodes = boundary_nodes

else:
    graph = None
    b_nodes = None
    entry0 = None
#    entry1 = None
#    entry2 = None

b_nodes = MPI_COMM.bcast(b_nodes, root=0)
entry0 = MPI_COMM.bcast(entry0, root=0)
# entry1 = MPI_COMM.bcast(entry1, root=0)
# entry2 = MPI_COMM.bcast(entry2, root=0)

ER0 = en.compute_effective_resistance_matrix(graph=graph, largest_nodes=(entry0, b_nodes))
# ER1 = en.compute_effective_resistance_matrix(graph=graph, largest_nodes=(entry1, b_nodes ) )
# ER2 = en.compute_effective_resistance_matrix(graph=graph, largest_nodes=(entry2, b_nodes ) )

if MPI_RANK == 0:
    np.savetxt("ER0.csv", ER0)
    # np.savetxt("ER1.csv", ER1)
    # np.savetxt("ER2.csv", ER2)
