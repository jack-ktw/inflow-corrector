#!/bin/bash
#+++++++++++++++++++++++++++++++++
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:15:00
#SBATCH --job-name=paraview
#SBATCH --account=rrg-wind
#SBATCH --output=%j.txt
#SBATCH --mail-type=ALL
#+++++++++++++++++++++++++++++++++

cd $SLURM_SUBMIT_DIR

# module load NiaEnv/2018a
# module load paraview-offscreen/5.6.0
module purge
module load paraview-offscreen/5.9.1
# mpiexec pvserver

echo "Starting Script"
pvbatch --force-offscreen-rendering export_cell_xyz.py
echo "Ending Script"
# pvbatch --mesa-swr-avx2 --force-offscreen-rendering para_script.py