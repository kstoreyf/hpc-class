#!/bin/bash

#SBATCH --job-name=mpijob
#SBATCH --output=logs/mpijob_%j.out
#SBATCH --nodes=2
#SBATCH --ntasks=4
#SBATCH --ntasks-per-node=2
#SBATCH --cpus-per-task=1
#SBATCH --mem=2GB
#SBATCH --time=00:10:00
#SBATCH --verbose

module purge
module load amber/openmpi/intel/20.06

mpiexec ./pingpong 0 1
