#!/bin/bash
#+++++++++++++++++++++++++++++++++
#SBATCH --nodes=4
#SBATCH --ntasks-per-node=40
#SBATCH --time=12:00:00
#SBATCH --job-name=ts1_run2_finer
#SBATCH --output=%j.txt
#SBATCH --mail-type=FAIL
#+++++++++++++++++++++++++++++++++
 
# cd $SLURM_SUBMIT_DIR

test=$(sbatch job-SHM.sh);
if [[ "$test" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then test=$(sbatch -d afterany:${BASH_REMATCH[1]} job-simple.sh); fi
if [[ "$test" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then test=$(sbatch -d afterany:${BASH_REMATCH[1]} prepare_LES.sh); fi
if [[ "$test" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then test=$(sbatch -d afterany:${BASH_REMATCH[1]} changeDict.sh); fi
if [[ "$test" =~ Submitted\ batch\ job\ ([0-9]+) ]]; then test=$(sbatch -d afterany:${BASH_REMATCH[1]} job-solver.sh); fi