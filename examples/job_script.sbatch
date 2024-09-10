#!/bin/bash

# for the full mouse brain vasculature, we need at least 12 nodes
#SBATCH --job-name="1_BF"
#SBATCH --nodes=8
#SBATCH --ntasks-per-node=32

#SBATCH --account=proj16
#SBATCH --partition=prod
#SBATCH --constraint=cpu
#SBATCH --time=00:25:00

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
echo "### Simulation Start"
echo
# time srun dplace python "${JOB_SCRIPT_DIR}/compute_static_flow_pressure.py"
time srun dplace python "${JOB_SCRIPT_DIR}/simulate_OU_process.py"
