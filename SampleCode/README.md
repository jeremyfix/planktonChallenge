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
conda activate plankton
python preprocess.py --datadir dataset --outputdir preprocessed
```

It loads the raw images, perform some preprocessing (here, resize/pad the images to square images 300 x 300), split the
data into a training and validation fold, by default 0.95/0.05.


## Training

## Inference


## On SLURM managed clusters

We provide sbatch scripts for running all the above steps. You may need to adapt some paths (e.g. to the data)

For setting up the conda environment :

```
sbatch slurm-conda-setup.sbatch
```

For preprocessing the dataset :

```

```
