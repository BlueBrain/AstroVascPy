#!/bin/bash -l
#SBATCH --ntasks=1
#SBATCH --mem=0
#SBATCH --time=2:00:00       # Time limit
#SBATCH --partition=prod      # Submit to the production 'partition'
#SBATCH --constraint=cpu,clx      # Constraint the job to run on nodes with/without SSDs. If you want SSD, use only "nvme". If you want KNLs then "knl"
##SBATCH --cpus-per-task=1
#SBATCH --account=proj16      # your project number
#SBATCH --job-name=astrovascpy
#SBATCH --exclusive

set -e

export OMP_NUM_THREADS=1

module purge
module load unstable
module load python
# source /gpfs/bbp.cscs.ch/project/proj34/home/battini/bloodflow_test/vasculature_39/bin/activate
source ~/py_blood

time python simulate_OU_process.py
