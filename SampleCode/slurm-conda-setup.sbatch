#!/bin/bash

#SBATCH --job-name=planktonEnvSetup
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=1:00:00
#SBATCH --output=logslurms/slurm-conda-%j.out
#SBATCH --error=logslurms/slurm-conda-%j.err

# Load the conda module
export PATH=/opt/conda/bin:$PATH

# Create conda environment 
conda create --name plankton python=3.9 --force
source activate plankton
pip install -r requirements.txt 
