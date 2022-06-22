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
import pathlib
import sys

if len(sys.argv) != 2:
    print(f"Usage : {sys.argv[0]} modelpath")
    sys.exit(-1)


def makejob(partition, datadir, commit_id, normalize, params):
    paramsstr = " ".join([f"--{name} {value}" for name, value in params.items()])
    if normalize:
        paramsstr += " --normalize "
    return f"""#!/bin/bash

#SBATCH --job-name=challenge-infer-{params['model']}
#SBATCH --nodes=1
#SBATCH --partition={partition}
#SBATCH --time=00:10:00
#SBATCH --output=logslurms/slurm-%j.out
#SBATCH --error=logslurms/slurm-%j.err

current_dir=`pwd`

# Fix env variables as bashrc and profile are not loaded
export LOCAL=$HOME/.local
export PATH=$PATH:$LOCAL/bin

echo "Running on $(hostname)"

echo "Copying the source directory and data"
date
mkdir $TMPDIR/challenge
cd ..
rsync -r . $TMPDIR/challenge/ --exclude 'SampleCode/logslurms' --exclude 'DatasetPreparation' --exclude 'Report' --exclude 'SampleCode/logs' --exclude data

cd $TMPDIR/challenge/SampleCode/

echo "copy the model to be evaluated from $current_dir/{modelpath} to {modelpath}"
mkdir -p {modelpath}
cp -r $current_dir/{modelpath}/* {modelpath}

echo "Checkout the version id {commit_id}"
git checkout {commit_id}

echo ""
echo "Conda virtual env"

export PATH=/opt/conda/bin:$PATH
source activate plankton


echo ""
echo "Testing"
date

python3 main.py  --datadir {datadir} {paramsstr} test

if [[ $? != 0 ]]; then
    exit -1
fi

echo "Copying the preditions back to the original directory"
cp {modelpath}/prediction.csv {modelpath}/probs.csv $current_dir/{modelpath}/

date
"""


def submit_job(job):
    with open("job-infer.sbatch", "w") as fp:
        fp.write(job)
    os.system("sbatch job-infer.sbatch")


#
partition = "gpu_prod_long"
datadir = "/mounts/Datasets1/ChallengeDeep/test"

modelpath = sys.argv[1]
# Extract the modelname from the modelpath summary file
summary_file = open(pathlib.Path(modelpath) / "summary.txt", "r").readlines()
token = "Arguments : Namespace("
arguments = []
for l in summary_file:
    if l.strip()[: len(token)] == token:
        arguments = l.strip()[len(token) : -1]  # Skip the stem and the end )
        break
arguments = arguments.split(",")
arguments = [values.strip().split("=") for values in arguments]
arguments = {k: v for k, v in arguments}

commit_id = arguments["commit_id"]
normalize = bool(arguments["normalize"])
params = {"model": arguments["model"], "modelpath": modelpath}


# Ensure the log directory exists
os.system("mkdir -p logslurms")

# Launch the batch jobs
submit_job(makejob(partition, datadir, commit_id, normalize, params))
