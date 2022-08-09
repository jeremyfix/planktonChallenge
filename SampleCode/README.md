# Sample code for a plankton classifier

The code is released under **GPL v3.0 license**.

## Virtual environment setup

If you have [conda](https://anaconda.org/) installed, you can setup a virtual environment with the required dependencies with 

```
conda create --name plankton python=3.9 --force
source activate plankton
pip install -r requirements.txt 
```

This must be done once for all; Then the virtual environment is loaded with 

```
source activate plankton
```

and unloaded with 

```
conda deactivate 
```


## Preprocessing the data

Before training, we first preprocess the images :

```
export PYTHONNOUSERSITE=1
source activate plankton
python preprocess.py --datadir dataset --outputdir preprocessed
```

It loads the raw images, perform some preprocessing (here, resize/pad the images to square images 300 x 300), split the
data into a training and validation fold, by default 0.95/0.05.


## Training

To run a training, you run the `main.py` script. To train an efficientNet B3, using 0.2 of mixup for data augmentation :

```
export PYTHONNOUSERSITE=1
source activate plankton
python main.py --datadir dataset --model efficientnet_b3 --mixup 0.2 train
```

## Inference

For inference, you need to know where the best parameters have been saved and which is the model to evaluate. For example, if your simulation saved its outputs to `logs/regnety_016_8391_0/` and is a `regnety_016` model :

```
export PYTHONNOUSERSITE=1
source activate plankton
python main.py --datadir /path/to/test/data --model regnety_016 --modelpath ./logs/regnety_016_8391_0/ test
```

This will save the predictions as csv files within the model directory.

## On SLURM managed clusters

We provide sbatch scripts for running all the above steps. You may need to adapt some paths (e.g. to the data)

For setting up the conda environment :

```
sbatch slurm-conda-setup.sbatch
```

For preprocessing the dataset :

```
sbatch slurm-preprocess.sbatch
```

For training, it is advised to use the `job.py` script which creates a sbatch script and run it. The created batch script takes care of the commit id of the code to be execute and ensures everything has been git added and git commited. You should edit the end of the `job.py` script and run it as :
```
python job.py
```

For inference, we provide a `job-infer.py` scripts which takes care of extracting the correct commit id, model name, etc.. from a log directory; for example, for evaluating the simulation that saved its output to `logs/regnety_016_8391_0/`, you simply need to run:

```
python job-infer.py logs/regnety_016_8391_0/
```
