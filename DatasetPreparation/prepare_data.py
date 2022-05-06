#!/usr/bin/env python3
# coding: utf-8

# Standard modules
import argparse
import sys
import os
import glob
import logging
import shutil
import pathlib
import random
import joblib
import contextlib

# External modules
import numpy as np
from numpy.random import MT19937
from numpy.random import RandomState, SeedSequence
import pandas as pd
from tqdm import tqdm

"""
We prepare the data which means : 
    - fusing some classes
    - discarding some others

We also relabel the images (to avoid tricking) and split into a train set, a public test set and a private test set

Finally the classes are numbered to avoid issues in the way the students may output their results

We do not use the csv files. If we were to use them, given we rename the objids (to limit the possibility to trick) 
and fuse some classes, we should do that carefully !!! Also, if you want to do that, consider using pandas as

    import pandas as pd
    taxa = pd.read_csv(sourcedir / 'taxa.csv', index_col='objid')
    idx_to_drop = [int(p.name[:-4]) for p in dirname.glob('*.jpg')]
    taxa = taxa.drop(idx_to_drop)
    taxa.to_csv(targetdir / 'taxa.csv')

"""
skiped_classes = [
    "badfocus__artefact",
    "badfocus__Copepoda",
    "bubble",
    "multiple__Copepoda",
    "multiple__other",
]
fused_classes = {"fiber__detritus": "detritus", "seaweed": "detritus"}
class_stats = {}


@contextlib.contextmanager
def tqdm_joblib(tqdm_object):
    """Context manager to patch joblib to report into tqdm progress bar given as argument"""

    class TqdmBatchCompletionCallback(joblib.parallel.BatchCompletionCallBack):
        def __init__(self, *args, **kwargs):
            super().__init__(*args, **kwargs)

        def __call__(self, *args, **kwargs):
            tqdm_object.update(n=self.batch_size)
            return super().__call__(*args, **kwargs)

    old_batch_callback = joblib.parallel.BatchCompletionCallBack
    joblib.parallel.BatchCompletionCallBack = TqdmBatchCompletionCallback
    try:
        yield tqdm_object
    finally:
        joblib.parallel.BatchCompletionCallBack = old_batch_callback
        tqdm_object.close()


def create_image(source_path, target_path, maxsize, rotation):
    rotatecmd = f" -rotate {rotation} "
    cmd = f"convert '{source_path}' {rotatecmd} -resize {maxsize}x{maxsize}\> {target_path}"
    # print(cmd)
    os.system(cmd)
    # if testmode:
    #     target_path.symlink_to(source_path)
    # else:
    #     shutil.copy(source_path, target_path)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()
    parser.add_argument("--srcdir", type=pathlib.Path, required=True)
    parser.add_argument("--tgtdir", type=pathlib.Path, required=True)
    parser.add_argument(
        "--pubtest",
        type=float,
        default=0.05,
        help="fraction of the data used for the public test",
    )
    parser.add_argument(
        "--privtest",
        type=float,
        default=0.05,
        help="fraction of the data used for the public test",
    )
    parser.add_argument(
        "--testmode",
        action="store_true",
        help="In test mode, we do not copy the files but make symllinks",
    )
    parser.add_argument(
        "--maxsamples",
        type=int,
        default=200000,
        help="The max number of samples per class",
    )
    parser.add_argument(
        "--randrotate",
        action="store_true",
        help="Whether or not to random rotate the images",
    )
    parser.add_argument(
        "--maxsize",
        type=int,
        default=300,
        help="The maximum side size of an image. We keep the aspect ratio",
    )
    parser.add_argument(
        "--numjobs", type=int, default=7, help="The number of concurrent jobs to use"
    )
    parser.add_argument(
        "--seed", type=int, default=0, help="The seed to generate the data"
    )
    args = parser.parse_args()

    sourcedir = args.srcdir
    targetdir = args.tgtdir
    testmode = args.testmode
    maxsamples = args.maxsamples
    randrotate = args.randrotate
    maxsize = args.maxsize
    if maxsamples == -1:
        maxsamples = np.inf
    numjobs = args.numjobs

    random.seed(args.seed)
    rng = RandomState(MT19937(SeedSequence(args.seed)))

    targetdir_train = targetdir / "train"
    targetdir_pubtest = targetdir / "pubtest"
    targetdir_privtest = targetdir / "privtest"
    pubtest_dict = {}  # Dict : pathlib.Path : classid
    privtest_dict = {}  # Dict : pathlib.Path : classid

    logging.info(f"Removing {targetdir}")
    shutil.rmtree(targetdir, ignore_errors=True)
    for p in [targetdir, targetdir_train, targetdir_pubtest, targetdir_privtest]:
        logging.info(f"Creating {p}")
        os.makedirs(p)

    pool_task = []
    for dirname in (sourcedir / "imgs").glob("*"):
        cls = dirname.name
        if cls in skiped_classes:
            logging.info(f"{cls} skipped")
            continue

        if cls in fused_classes:
            # The old class is changed to the new one
            targetcls = fused_classes[cls]
        else:
            targetcls = cls
        targetcls = targetcls.replace(" ", "_")

        if targetcls not in class_stats:
            class_stats[targetcls] = {"id": len(class_stats), "num": 0}
        targetcls_id = class_stats[targetcls]["id"]

        targetclsdir_train = targetdir_train / f"{targetcls_id:03d}_{targetcls}"

        # And then make the symlinks
        if not targetclsdir_train.exists():
            os.makedirs(targetclsdir_train)
        # A fraction is used for the train set, for the public test set
        # and for the private test set
        filelist = list(dirname.glob("*.jpg"))
        rng.shuffle(filelist)
        num_files = min(len(filelist), maxsamples)
        filelist = filelist[:num_files]
        num_privtest = int(args.privtest * num_files)
        num_pubtest = int(args.pubtest * num_files)
        num_train = num_files - num_privtest - num_pubtest
        logging.info(
            f"For class {targetcls_id:4d} named {cls:35s}, train : {num_train:6d}, pubtest : {num_pubtest:6d}, privtest : {num_privtest: 6d}"
        )
        if num_pubtest == 0 or num_privtest == 0:
            logging.error("Either pubtest or privtest is empty !!")
            sys.exit(-1)

        # Fill in the private and public test sets
        for p in filelist[:num_privtest]:
            privtest_dict[p] = targetcls_id
        for p in filelist[num_privtest : (num_privtest + num_pubtest)]:
            pubtest_dict[p] = targetcls_id

        # Copy the training set images and rename the files on the fly
        for p in filelist[(num_privtest + num_pubtest) :]:
            pi = class_stats[targetcls]["num"]
            target_path = targetclsdir_train / f"{pi}.jpg"
            # create_image(p, target_path, maxsize, randrotate)
            rotation = 0
            if randrotate:
                rotation = random.randint(0, 3) * 90
            pool_task.append(
                {
                    "source_path": p,
                    "target_path": target_path,
                    "maxsize": maxsize,
                    "rotation": rotation,
                }
            )
            class_stats[targetcls]["num"] += 1

    ntask = len(pool_task)
    logging.info(f"We ended up creating {len(class_stats)} classes")
    logging.info(f"Got {ntask} jobs to execute")

    # Run the tasks in parallel
    with tqdm_joblib(tqdm(desc="Training set", total=ntask)) as pbar:
        joblib.Parallel(n_jobs=numjobs)(
            joblib.delayed(create_image)(**p) for p in pool_task
        )

    # We now create the public and private test sets
    # For Kaggle, we need to provide a single set of images
    # it seems we cannot, say, release the private test set latter on
    # InClass competition
    targetdir_test = targetdir / "test"
    logging.info(f"Removing {targetdir_test}")
    shutil.rmtree(targetdir_test, ignore_errors=True)
    logging.info(f"Creating {targetdir_test} and {targetdir_test}/imgs")
    os.makedirs(targetdir_test)
    os.makedirs(targetdir_test / "imgs")

    pool_task = []

    def make_testset(start_idx, test_dict, test_path, rng, usage):
        # Shuffle the set
        test_list = list(test_dict.items())
        rng.shuffle(test_list)
        if not np.isinf(maxsamples):
            test_list = test_list[:maxsamples]
        imgidlabels = [
            [f"{start_idx + pi}.jpg", clid, usage]
            for pi, (_, clid) in enumerate(test_list)
        ]

        logging.info(f"Copying the {len(test_list)} images")
        # Copy the image files renaming them on the fly
        for pi, (p, _) in enumerate(test_list):
            target_path = test_path / "imgs" / f"{start_idx + pi}.jpg"
            rotation = 0
            if randrotate:
                rotation = random.randint(0, 3) * 90
            pool_task.append(
                {
                    "source_path": p,
                    "target_path": target_path,
                    "maxsize": maxsize,
                    "rotation": rotation,
                }
            )
        return imgidlabels

    priv_imgidlabels = make_testset(0, privtest_dict, targetdir_test, rng, "Private")
    pub_imageidlabels = make_testset(
        len(pool_task), pubtest_dict, targetdir_test, rng, "Public"
    )

    test_labels = pd.DataFrame(
        priv_imgidlabels + pub_imageidlabels, columns=["imgname", "label", "Usage"]
    )
    test_labels.to_csv(targetdir_test / "labels.csv", index=False)

    # Run the tasks in parallel
    ntask = len(pool_task)
    with tqdm_joblib(tqdm(desc="Test sets", total=ntask)) as pbar:
        joblib.Parallel(n_jobs=numjobs)(
            joblib.delayed(create_image)(**p) for p in pool_task
        )
