#!/usr/bin/env python3
# coding: utf-8
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
Preprocess the data and save the resized, cropped, etc.. images

Takes a directory as dir/000_xxx/i.jpg , ...
loads the images, resize them , pad, crop , etc..

And saves the resulting images into dir/train/000_xxx/j.jpg
 and                                dir/valid/000_xxx/k.jpg
"""

# Standard modules
import logging
import argparse
import pathlib
import os

# External modules
import tqdm
import torch

# Local modules
import data

IMAGE_IDX_FILEWIDTH = 6


def process_dataset(dataset: torch.data.Dataset, outputdir: pathlib.Path):
    """Loads and apply the transform on every image of a dataset

    Args:
        dataset (torch.data.Dataset) : a mappable dataset
        outputdir (Pathlib.path) : an output dir to save the data in
    """
    img_idx = {}
    logging.info(f"Processing {len(dataset)} objects in {outputdir}")
    for X, y in tqdm.tqdm(dataset):
        if y not in img_idx:
            img_idx[y] = 0
        output_filedir = outputdir / f"{y:03d}"

        if not output_filedir.exists():
            os.makedirs(output_filedir)
        if len(str(img_idx[y])) >= IMAGE_IDX_FILEWIDTH:
            raise ValueError(
                "Our image filename format expects a number of images <= 1"
                + "0" * IMAGE_IDX_FILEWIDTH
            )
        # If the following assert is not true
        # we need to adapt the f-string format below
        # do not know how to that with a variable
        assert IMAGE_IDX_FILEWIDTH == 6
        output_filepath = output_filedir / f"{img_idx[y]:06d}.jpg"
        img_idx[y] += 1
        X.save(output_filepath)


def main(args):
    """Precompte the training and validation images (resize them all)

    Args:
        args (dict): the parameters of the transform, data dir and output dir
    """
    logging.info("Loading the images")
    train_dataset, valid_dataset = data.load_dataset(
        args.datadir,
        transform=data.Resize(data.__default_size[0]),
        val_ratio=args.val_ratio,
    )
    process_dataset(train_dataset, args.outputdir / "train")
    logging.info("Done")
    process_dataset(valid_dataset, args.outputdir / "valid")
    logging.info("Done")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--datadir", type=pathlib.Path, required=True)
    parser.add_argument("--outputdir", type=pathlib.Path, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=64)
    parser.add_argument("--val_ratio", type=float, default=0.05)

    args = parser.parse_args()
    if args.outputdir.exists():
        logging.info("The datadir exists, nothing to be done")
    else:
        main(args)
