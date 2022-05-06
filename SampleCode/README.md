# Sample code for a plankton classifier

The code is released under **GPL v3.0 license**.

Before training, we first preprocess the images :

```
python3 preprocess.py --datadir dataset --outputdir preprocessed
```

It loads the raw images, perform some preprocessing (here, resize/pad the images to square images 300 x 300), split the
data into a training and validation fold, by default 0.95/0.05.

