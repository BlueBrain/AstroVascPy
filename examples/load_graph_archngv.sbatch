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

pushd ..
source setup.sh
popd

echo
echo "### Simulation Start"
echo
# It is imperative to use srun and dplace, otherwise the Python processes 
# do not work properly (possible deadlocks and/or performance degradation)
time srun -n 1 --mpi=none dplace python load_graph_archngv.py
