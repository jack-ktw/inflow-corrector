#!/bin/bash
#+++++++++++++++++++++++++++++++++
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=0:30:00
#SBATCH --job-name=blockmesh
#SBATCH --output=%j.txt
#SBATCH --mail-type=FAIL
#+++++++++++++++++++++++++++++++++
 
# cd $SLURM_SUBMIT_DIR

bash job-blockMesh.sh # Need to run run_para_export_cell_xyz.sh manually to update cell_xyz.csv, 
# the process is not here due to a weird paraview error that causes the script to hangs.
bash job-inflow_init.sh
bash job-inflow_iterate.sh
bash job-inflow_iterate.sh
bash job-inflow_iterate.sh
bash job-inflow_iterate.sh