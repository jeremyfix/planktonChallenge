# 3MD4040 Dataset preparation

The 3MD4040 challenge dataset is based on the [ZooScan database downloable from seanoe](https://www.seanoe.org/data/00446/55741/). 

The script requires the imagemagick `convert` command and runs as :

```
python3 prepare_data.py --srcdir ./ZooScanNet --tgtdir ./dataset
```

For preparing the train/test folds, you simply run the `prepare_data.py` script. By default, the script will : 

- split in 95% for training, 5% for testing, 
- keep a maximum of 200000 samples per class, 
- apply random rotations to the images (to obfuscate the data hopefully preventing a cheater overfitting the original full dataset to have good scores), 
- resize every images to 300x300,
- skip the classes badfocus artefact, badfocus copepoda, bubble, multiple copepoda, multiple other,
- fuze fiber detritus, seaweed and detritus

At the end, the dataset contains :

- 855.394 images for training
- 47.473 images for the public test set
- 47.472 images for the private test set
