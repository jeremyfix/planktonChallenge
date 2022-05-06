#!/usr/bin/env python3

"""
Script used to run experiments on a slurm cluster using the commit-id to guarantee the executed code version
"""

import os
import subprocess


def makejob(commit_id, nruns, partition, walltime, normalize, params):
    paramsstr = " ".join([f"--{name} {value}" for name, value in params.items()])
    if normalize:
        paramsstr += " --normalize "
    return f"""#!/bin/bash

#SBATCH --job-name=challenge-{params['model']}
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --time={walltime}
#SBATCH --output=logslurms/slurm-%A_%a.out
#SBATCH --error=logslurms/slurm-%A_%a.err
#SBATCH --array=0-{nruns-1}

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
rsync -r .. $TMPDIR/ --exclude 'SampleCode/logslurms' --exclude 'DatasetPreparation' --exclude Report 

cd $TMPDIR/SampleCode/
git checkout {commit_id}

echo ""
echo "Virtual env"

python3 -m pip install virtualenv --user
virtualenv -p python3 venv
source venv/bin/activate
python -m pip install -r requirements.txt



echo ""
echo "Training"
date

python3 main.py  --datadir ./ChallengeDeep/training {paramsstr} --logname {params['model']}_${{SLURM_ARRAY_JOB_ID}}_${{SLURM_ARRAY_TASK_ID}} --commit_id '{commit_id}' --logdir ${{current_dir}}/logs train

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
result = int(
    subprocess.check_output("git status | grep 'modifi' | wc -l", shell=True).decode()
)
if result != 1:
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
submit_job(
    makejob(
        commit_id,
        3,
        "gpu_prod_long",
        "48:00:00",
        True,
        {
            "model": "cait_s24_224",
            "batch_size": 32,
            "weight_decay": 0.00,
            "nepochs": 40,
            "base_lr": 0.0003,
            "loss": "BCE",
            "mixup": 0.2,
        },
    )
)