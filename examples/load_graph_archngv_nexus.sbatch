#!/bin/bash

#SBATCH --job-name="archngv"
#SBATCH --nodes=1

#SBATCH --account=proj137
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --time=00:40:00


#JOB_SCRIPT=$(scontrol show job ${SLURM_JOB_ID} | awk -F= '/Command=/{print $2}')
#JOB_SCRIPT_DIR=$(dirname ${JOB_SCRIPT})
#JOB_SCRIPT_DIR=/gpfs/bbp.cscs.ch/project/proj137/scratch/jacquemi/Nexus

SETUP_SCRIPT="${JOB_SCRIPT_DIR}/../setup.sh"
SETUP_SCRIPT="./setup.sh"
if [[ ! -f ${SETUP_SCRIPT} ]]; then
    >&2 echo "[ERROR] The 'setup.sh' script could not be found!"
    exit 2
fi

#source ${SETUP_SCRIPT}

#FILENAME_NGV="/gpfs/bbp.cscs.ch/project/proj137/NGVCircuits/rat_O1"
#FILENAME_NGV="/gpfs/bbp.cscs.ch/project/proj137/NGVCircuits/rat_O1/build/ngv_config.json"
CIRCUIT_NAME="NGV O1.v5 (Rat)"


GRAPH_PATH="./data/graphs_folder/dumped_graph_nexus.bin"

echo
echo "### Loading graph"
echo
# It is imperative to use srun and dplace, otherwise the Python processes
# do not work properly (possible deadlocks and/or performance degradation)
#time srun -n 1 --mpi=none dplace python ${JOB_SCRIPT_DIR}/load_graph_archngv.py --circuit_name ${CIRCUIT_NAME} --output_graph ${GRAPH_PATH}
time srun -n 1 --mpi=none dplace python ${JOB_SCRIPT_DIR}/load_graph_archngv_nexus.py --circuit_name "${CIRCUIT_NAME}" --output_graph ${GRAPH_PATH}
