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

pushd ..
source setup.sh
popd

echo
echo "### Simulation Start"
echo
# time srun dplace python compute_static_flow_pressure.py
time srun dplace python simulate_OU_process.py
