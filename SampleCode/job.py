#!/usr/bin/env python3
"""
    This script belongs to the plankton classification sample code
    Copyright (C) 2022 Jeremy Fix

    This program is free software: you can redistribute it and/or modify
    it under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    This program is distributed in the hope that it will be useful,
    but WITHOUT ANY WARRANTY; without even the implied warranty of
    MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
    GNU General Public License for more details.

    You should have received a copy of the GNU General Public License
    along with this program.  If not, see <https://www.gnu.org/licenses/>.
"""

"""
Script used to run experiments on a slurm cluster using the commit-id to guarantee the executed code version
"""

import os
import subprocess


def makejob(
    commit_id,
    nruns,
    partition,
    walltime,
    normalize,
    class_weights,
    batch_sampler,
    params,
):
    paramsstr = " ".join([f"--{name} {value}" for name, value in params.items()])
    if normalize:
        paramsstr += " --normalize "
    if class_weights:
        paramsstr += " --class_weights "
    if batch_sampler:
        paramsstr += " --batch_sampler "
    return f"""#!/bin/bash

#SBATCH --job-name=challenge-{params['model']}
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --time={walltime}
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=0-{nruns-1}
#SBATCH --exclude=sh[00,10-16]

current_dir=`pwd`

# Fix env variables as bashrc and profile are not loaded
export LOCAL=$HOME/.local
export PATH=$PATH:$LOCAL/bin

echo "Session " {params['model']}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}}
date

echo "Running on $(hostname)"
# env

echo "Copying the source directory and data"
date
mkdir $TMPDIR/challenge
cd ..
rsync -r . $TMPDIR/challenge/ --exclude 'SampleCode/logslurms' --exclude 'DatasetPreparation' --exclude 'Report'

cd $TMPDIR/challenge/SampleCode/

ls 

git checkout {commit_id}

echo ""
echo "Conda virtual env"

export PATH=/opt/conda/bin:$PATH
source activate plankton


echo ""
echo "Training"
date

# To enforce deterministic algorithms, cuda requires the following variable to be defined
# see : https://docs.nvidia.com/cuda/cublas/index.html#cublasApi_reproducibility
export CUBLAS_WORKSPACE_CONFIG=:4096:8
python3 main.py  --datadir ./data/ {paramsstr} --logname {params['model']}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} --commit_id '{commit_id}' --logdir ${{current_dir}}/logs train

if [[ $? != 0 ]]; then
    exit -1
fi
date
"""


def submit_job(job):
    with open("job.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job.sbatch")


# Ensure all the modified files have been staged and commited
result = int(subprocess.check_output("git status -s -uno | wc -l", shell=True).decode())
if result > 0:
    print(f"We found {result} modifications not staged or commited")
    raise RuntimeError(
        "You must stage and commit every modification before submission "
    )

commit_id = subprocess.check_output(
    "git log --pretty=format:'%H' -n 1", shell=True
).decode()

# Ensure the log directory exists
os.system("mkdir -p logslurms")

# Launch the batch jobs
for model in ["resnet18", "regnety_016", "resnet152", "efficientnet_b3"]:
    submit_job(
        makejob(
            commit_id,
            1,
            "gpu_prod_long",
            "48:00:00",
            normalize=True,
            class_weights=False,
            batch_sampler=True,
            params={
                "model": model,
                "batch_size": 64,
                "weight_decay": 0.00,
                "nepochs": 100,
                "base_lr": 0.0003,
                "loss": "BCE",
                "mixup": 0.2,
                "seed": 42,
            },
        )
    )
