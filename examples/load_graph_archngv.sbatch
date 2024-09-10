#!/bin/bash

#SBATCH --job-name="archngv"
#SBATCH --nodes=1

#SBATCH --account=proj16
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --time=00:30:00

#SBATCH --cpus-per-task=2
#SBATCH --exclusive
#SBATCH --mem=0
#SBATCH --output="%x-%j.log"

JOB_SCRIPT=$(scontrol show job ${SLURM_JOB_ID} | awk -F= '/Command=/{print $2}')
JOB_SCRIPT_DIR=$(dirname ${JOB_SCRIPT})

SETUP_SCRIPT="${JOB_SCRIPT_DIR}/../setup.sh"
if [[ ! -f ${SETUP_SCRIPT} ]]; then
    >&2 echo "[ERROR] The 'setup.sh' script could not be found!"
    exit -1
fi

source ${SETUP_SCRIPT}

echo
echo "### Loading graph"
echo
# It is imperative to use srun and dplace, otherwise the Python processes
# do not work properly (possible deadlocks and/or performance degradation)
time srun -n 1 --mpi=none dplace python "${JOB_SCRIPT_DIR}/load_graph_archngv.py"
