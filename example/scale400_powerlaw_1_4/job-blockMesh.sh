#!/bin/bash
#+++++++++++++++++++++++++++++++++
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=0:15:00
#SBATCH --job-name=blockmesh
#SBATCH --output=%j.txt
#SBATCH --mail-type=FAIL
#+++++++++++++++++++++++++++++++++
 
# cd $SLURM_SUBMIT_DIR

module load NiaEnv/2019b gcc openmpi openfoam/20.06 boost
export EBROOTBOOST=$BOOST_ROOT
source "/gpfs/fs1/scinet/niagara/software/2019b/opt/gcc-8.3.0-openmpi-4.0.1/openfoam/20.06/OpenFOAM-v2006/etc/bashrc"

mkdir -p $PWD/logs

nproc=$SLURM_NTASKS

### Create volume
blockMesh              2>&1 | tee logs/log.01.blockMesh.txt

### Decompose before snappy
decomposePar -copyZero 2>&1 | tee logs/log.02.decomposePar.txt