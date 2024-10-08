#!/bin/bash

#SBATCH --job-name="archngv"
#SBATCH --nodes=1

#SBATCH --account=proj137
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00

#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output="%x-%j.log"

JOB_SCRIPT=$(scontrol show job ${SLURM_JOB_ID} | awk -F= '/Command=/{print $2}')
JOB_SCRIPT_DIR=$(dirname ${JOB_SCRIPT})
JOB_SCRIPT_DIR='/gpfs/bbp.cscs.ch/data/scratch/proj137/jacquemi/Nexus'


SETUP_SCRIPT="${JOB_SCRIPT_DIR}/../setup.sh"
if [[ ! -f ${SETUP_SCRIPT} ]]; then
    >&2 echo "[ERROR] The 'setup.sh' script could not be found!"
    exit 2
fi

source ${SETUP_SCRIPT}

FILENAME_NGV=""

GRAPH_PATH="./data/graphs_folder/dumped_graph.bin"

#CIRCUIT_NAME="tiny_real_spine_morph"
CIRCUIT_NAME="NGV O1.v5 (Rat)"

# It is imperative to use srun and dplace, otherwise the Python processes
# do not work properly (possible deadlocks and/or performance degradation)
if [[ -z $FILENAME_NGV ]]; then
        echo "### Loading graph from name: $CIRCUIT_NAME from Nexus"
        echo "Execute ${JOB_SCRIPT_DIR}/load_graph_archngv.py with srun"
        time srun -n 1 --mpi=none dplace python ${JOB_SCRIPT_DIR}/load_graph_archngv.py --circuit-name "${CIRCUIT_NAME}" --output-graph ${GRAPH_PATH}
else
        echo "### Loading graph from filename: $FILENAME_NGV"
        echo "Execute ${JOB_SCRIPT_DIR}/load_graph_archngv.py with srun"
        time srun -n 1 --mpi=none dplace python ${JOB_SCRIPT_DIR}/load_graph_archngv.py --circuit-path "${FILENAME_NGV}" --output-graph ${GRAPH_PATH}
fi

echo 
