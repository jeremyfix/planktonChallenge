#!/bin/bash

#SBATCH --job-name=planktonPreproces
#SBATCH --nodes=1
#SBATCH --partition=gpu_prod_long
#SBATCH --time=10:00:00
#SBATCH --output=logslurms/slurm-preprocess-%j.out
#SBATCH --error=logslurms/slurm-preprocess-%j.err

PATH_TO_DATASET=/mounts/Datasets1/ChallengeDeep/train
OUTPUT_DIR=./data
NUM_WORKERS=7
BATCH_SIZE=128
VAL_RATIO=0.1
SEED=42

# Load the conda module
echo "Running on $(hostname)"

# Load the conda environment
export PATH=/opt/conda/bin:$PATH
source activate plankton

# Preprocess the data
echo "Starting the preprocessing"
date

python preprocess.py --datadir $PATH_TO_DATASET --outputdir $OUTPUT_DIR --num_workers $NUM_WORKERS --batch_size $BATCH_SIZE --val_ratio $VAL_RATIO --seed $SEED

echo "Preprocessing completed"
date
