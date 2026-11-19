#!/bin/bash
#+++++++++++++++++++++++++++++++++
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --time=3:00:00
#SBATCH --job-name=square0_SHM
#SBATCH --output=%j.txt
#SBATCH --mail-type=FAIL
#+++++++++++++++++++++++++++++++++
 
cd $SLURM_SUBMIT_DIR

module load NiaEnv/2019b gcc openmpi openfoam/20.06 boost
export EBROOTBOOST=$BOOST_ROOT
source "/gpfs/fs1/scinet/niagara/software/2019b/opt/gcc-8.3.0-openmpi-4.0.1/openfoam/20.06/OpenFOAM-v2006/etc/bashrc"

mkdir -p $PWD/logs

nproc=$SLURM_NTASKS

### Create volume
blockMesh              2>&1 | tee logs/log.00.blockMesh

### Create eMesh files
surfaceFeatureExtract  2>&1 | tee logs/log.01.surfaceFeatureExtract

### Decompose before snappy
cp system/decomposeParDict.${nproc} system/decomposeParDict
decomposePar -copyZero 2>&1 | tee logs/log.02.decomposePar.beforeSnappy

### Parallel snappy
cp system/snappyHexMeshDict system/snappyHexMeshDict
mpirun -np ${nproc} snappyHexMesh -overwrite  -parallel  \
  2>&1 | tee logs/log.03.snappyHexMesh.${snappy}.parallel.${nproc}
mpirun -np ${nproc} checkMesh     -parallel  -allGeometry -allTopology -writeAllFields -writeSets vtk  \
  2>&1 | tee logs/log.04.checkMesh.parallel.${nproc}
mpirun -np ${nproc} renumberMesh  -overwrite  -parallel  \
  2>&1 | tee logs/log.05.renumberMesh.parallel.${nproc}

### Calculate yPlus
mpirun -np ${nproc} simpleFoam -postProcess -func yPlus -parallel  \
  2>&1 | tee logs/log.06.simpleFoam.yPlus.preProcess