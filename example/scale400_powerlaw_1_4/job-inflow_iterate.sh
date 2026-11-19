#!/bin/bash
#+++++++++++++++++++++++++++++++++
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --time=00:30:00
#SBATCH --job-name=job-DFSR
#SBATCH --output=%j.txt
#SBATCH --mail-type=FAIL
#+++++++++++++++++++++++++++++++++
 
# cd $SLURM_SUBMIT_DIR

mkdir -p $PWD/logs
nproc=$SLURM_NTASKS

#####################
module purge
module load CCEnv StdEnv/2020  gcc/9.3.0  openmpi/4.0.3 openfoam/8 imkl/2020.1.217 fftw/3.3.8
export SCINET_FFTW_BASE=$EBROOTFFTW
mpirun -np ${nproc} DFSRTurb -parallel

module purge
module load NiaEnv/2019b python/3.8
source ~/.virtualenvs/py38/bin/activate
python3 inflow_iterate.py 0.5 0.5 0.5