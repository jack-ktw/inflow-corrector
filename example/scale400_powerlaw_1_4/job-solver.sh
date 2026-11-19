#!/bin/bash
#+++++++++++++++++++++++++++++++++
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --time=6:00:00
#SBATCH --job-name=ts1_run2_finer
#SBATCH --output=%j.txt
#SBATCH --mail-type=FAIL
#+++++++++++++++++++++++++++++++++
 
cd $SLURM_SUBMIT_DIR

module load NiaEnv/2019b gcc openmpi openfoam/20.06 boost
export EBROOTBOOST=$BOOST_ROOT
source "/gpfs/fs1/scinet/niagara/software/2019b/opt/gcc-8.3.0-openmpi-4.0.1/openfoam/20.06/OpenFOAM-v2006/etc/bashrc"

mkdir -p $PWD/logs

nproc=$SLURM_NTASKS

### Run solver
mpirun pimpleFoam -parallel 2>&1 | tee logs/log.07.pimpleFoam.parallel.${nproc}
