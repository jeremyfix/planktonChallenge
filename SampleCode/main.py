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

# Standard modules
import logging
import argparse
import pathlib
import os
import sys
import math

# External modules
import tqdm
import deepcs
import deepcs.training
import deepcs.testing
import deepcs.metrics
import deepcs.display
import deepcs.fileutils
import torch
import torch.nn as nn
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import pandas as pd
import cv2
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local modules
import data
import models
import utils

# Image mean and std can be computed by calling the "stats" function
IMAGE_MEAN = 0.92
IMAGE_STD = 0.16


def stats(args):
    """Computes the mean/std over all the pixels of the training data
    The statistics are displayed in the console
    Args:
        args (dict): the parameters for loading the data
    """
    logging.info("Computing the statistics over the training images")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Get the data
    loaders = data.load_preprocessed_data(
        args.datadir,
        train_transform=A.Compose([A.ToGray(), data.ScaleData(), ToTensorV2()]),
        valid_transform=A.Compose([A.ToGray(), data.ScaleData(), ToTensorV2()]),
        # val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    train_loader, valid_loader, n_samples_per_class = loaders

    X, _ = next(iter(train_loader))
    print(X)
    print(X.dtype)

    mean_pix, std_pix, num_imgs = 0, 0, 0
    print("go")
    for X, _ in train_loader:
        print("go 1 ")
        X = X.to(device)
        mean_pix += X.shape[0] * X.mean().item()  # Mean over num_pixels
        num_imgs += X.shape[0]
        print("go 2 ")
    mean_pix /= num_imgs  # Mean over num_pixels x batch_size
    for X, _ in train_loader:
        X = X.to(device)
        std_pix += (
            X.shape[0] * ((X - mean_pix) ** 2).mean().item()
        )  # Mean over num_pixels
    std_pix /= num_imgs
    std_pix = math.sqrt(std_pix)
    logging.info(f"Mean pixel value : {mean_pix} (std {std_pix})")


def train(args):
    """Train a neural network on the plankton classification task

    Args:
        args (dict): parameters for the training

    Examples::

        python3 main.py train --normalize --model efficientnet_b3
    """
    logging.info("Training")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Set up the train and valid transforms
    train_transforms = [
        data.KeepChannel(0, always_apply=True),
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.Rotate(180, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=255),
        data.ScaleBrightness(scale_range=(0.8, 1.0)),
        data.ScaleData(always_apply=True),
    ]
    if args.normalize:
        train_transforms.append(
            A.Normalize((IMAGE_MEAN,), (IMAGE_STD,), always_apply=True)
        )
    train_transforms.append(ToTensorV2())
    train_transforms = A.Compose(train_transforms)

    valid_transforms = [
        data.KeepChannel(0, always_apply=True),
        data.ScaleData(always_apply=True),
    ]
    if args.normalize:
        valid_transforms.append(
            A.Normalize((IMAGE_MEAN,), (IMAGE_STD,), always_apply=True)
        )
    valid_transforms.append(ToTensorV2())
    valid_transforms = A.Compose(valid_transforms)

    loaders = data.load_preprocessed_data(
        args.datadir,
        train_transform=train_transforms,
        valid_transform=valid_transforms,
        # val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
        label_smooth=args.labelsmooth,
        mixup=args.mixup,
        batch_sampler=args.batch_sampler,
    )
    train_loader, valid_loader, n_samples_per_class = loaders

    # train_in, train_out = next(iter(train_loader))
    # valid_in, valid_out = next(iter(valid_loader))

    num_classes = len(n_samples_per_class["train"])
    logging.info(f"Considering {num_classes} classes")

    # Make the model
    model = models.build_model(args.model, data.__default_size, num_classes)
    model.to(device)

    if args.class_weights:
        class_weights = utils.compute_class_weights(n_samples_per_class["train"])
    else:
        class_weights = torch.ones((num_classes,))
    class_weights = class_weights.to(device)
    logging.info(f"Min weight : {min(class_weights)}, max : {max(class_weights)}")

    # Make the loss
    bce_loss = nn.CrossEntropyLoss(weight=class_weights)
    if args.loss == "BCE":
        loss = bce_loss
    else:
        loss = utils.FocalLoss(gamma=0.5)
        logging.warn("Class weights are not used with the Focal loss")

    # Make the optimizer
    optimizer = torch.optim.Adam(
        model.parameters(), lr=args.base_lr, weight_decay=args.weight_decay
    )
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, 5, 0.5)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer)

    # Metrics
    metrics = {"CE": bce_loss, "accuracy": utils.accuracy}
    val_metrics = {"CE": bce_loss, "accuracy": utils.accuracy}

    # Callbacks
    if args.logname is None:
        logdir = deepcs.fileutils.generate_unique_logpath(args.logdir, args.model)
    else:
        logdir = args.logdir / args.logname
    logdir = pathlib.Path(logdir)

    if not logdir.exists():
        logdir.mkdir()
    input_size = next(iter(train_loader))[0].shape
    summary_text = (
        f"Logdir : {logdir}\n"
        + f"Commit id : {args.commit_id}\n"
        + f"Class weights : {class_weights}\n"
        + f"Num samples per class : {n_samples_per_class}\n\n"
        + "## Command \n"
        + " ".join(sys.argv)
        + "\n\n"
        + " Arguments : {}".format(args)
        + "\n\n"
        + "## Summary of the model architecture\n"
        + f"{deepcs.display.torch_summarize(model, input_size)}\n\n"
        + "## Datasets : \n"
        + f"Train : {train_loader.dataset}\n"
        + f"Validation : {valid_loader.dataset}"
    )

    with open(logdir / "summary.txt", "w") as f:
        f.write(summary_text)
    logging.info(summary_text)

    # Callbacks
    tensorboard_writer = SummaryWriter(log_dir=logdir)
    tensorboard_writer.add_text(
        "Experiment summary", deepcs.display.htmlize(summary_text)
    )

    model_checkpoint = deepcs.training.ModelCheckpoint(
        model, os.path.join(logdir, "best_model.pt"), min_is_best=False
    )

    for e in range(args.nepochs):
        deepcs.training.train(
            model,
            train_loader,
            loss,
            optimizer,
            device,
            metrics,
            num_epoch=e,
            tensorboard_writer=tensorboard_writer,
            dynamic_display=False,
        )

        test_metrics = deepcs.testing.test(model, valid_loader, device, val_metrics)
        macro_F1, class_F1 = utils.f1_metric(model, valid_loader, device)
        updated = model_checkpoint.update(macro_F1)
        logging.info(
            "[%d/%d] Test:   Loss : %.3f | F1 : %.3f | Acc : %.3f%% %s"
            % (
                e,
                args.nepochs,
                test_metrics["CE"],
                macro_F1,
                100.0 * test_metrics["accuracy"],
                "[>> BETTER <<]" if updated else "",
            )
        )

        logging.info(f"Class F1 : {class_F1}")
        # Confusion matrix
        cm = utils.make_confusion_matrix(model, valid_loader, device, num_classes)
        fig = utils.plot_confusion_matrix(cm)
        tensorboard_writer.add_figure("confusion", fig, e)

        # Metrics recording
        for m_name, m_value in test_metrics.items():
            tensorboard_writer.add_scalar(f"metrics/test_{m_name}", m_value, e)
        tensorboard_writer.add_scalar("metrics/fix_test_F1", macro_F1, e)
        scheduler.step(test_metrics["CE"])


def test(args):
    """Test a neural network on the plankton classification task test data

    The parameters are loaded from args['modelpath']/best_model.pt

    Args:
        args (dict): parameters for the testing

    Examples::

        python3 main.py test --normalize --model efficientnet_b3 --modelpath
    """
    if not args.modelpath:
        raise ValueError("The modelpath must be provided")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    logging.info("Loading the data")
    test_transforms = [
        data.SquareResize(data.__default_size[0], always_apply=True),
        data.KeepChannel(0, always_apply=True),
        data.ScaleData(always_apply=True),
    ]
    if args.normalize:
        test_transforms.append(A.Normalize((IMAGE_MEAN,), (IMAGE_STD,)))
    test_transforms.append(ToTensorV2())
    test_transforms = A.Compose(test_transforms)

    loader = data.load_test_data(
        args.datadir,
        transform=test_transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    # batch, _ = next(iter(loader))
    logging.info(loader.dataset)

    # Create the model
    logging.info("Loading the model")
    num_classes = 86
    model = models.build_model(args.model, data.__default_size, num_classes)
    model.to(device)

    logging.info(f"{deepcs.display.torch_summarize(model)}")

    model.load_state_dict(
        torch.load(args.modelpath / "best_model.pt", map_location=device)
    )

    # Put the model in eval mode
    model.eval()

    # Compute the prediction for the test set
    logging.info("Computing the predictions")
    predictions = []
    all_probs = []
    for (X, _) in tqdm.tqdm(loader):
        X = X.to(device)
        probs = nn.functional.softmax(model(X), dim=1).detach().to("cpu")
        all_probs += probs.tolist()
        pred = probs.argmax(axis=1).tolist()
        predictions += pred

    # Fuse the predictions and the filenames
    output = [
        (pathlib.Path(pathtargeti[0]).name, predi)
        for (pathtargeti, predi) in zip(loader.dataset.samples, predictions)
    ]
    output = pd.DataFrame(output, columns=["imgname", "label"])
    # And order by filename
    # output = output.sort_values(by=['imgname'])
    # And output the csv file
    output.to_csv(args.modelpath / "prediction.csv", index=False)
    logging.info(f"Predictions saved to {args.modelpath} / prediction.csv")

    all_probs_filename = [
        (pathlib.Path(pathtargeti[0]).name, *probs)
        for (pathtargeti, probs) in zip(loader.dataset.samples, all_probs)
    ]
    all_probs_filename = pd.DataFrame(
        all_probs_filename,
        columns=["imgname"] + [f"cls_{i}" for i in range(num_classes)],
    )
    all_probs_filename.to_csv(args.modelpath / "probs.csv", index=False)
    logging.info(f"Probs saved to {args.modelpath} / probs.csv")


if __name__ == "__main__":
    logging.basicConfig(stream=sys.stdout, level=logging.INFO, format="%(message)s")
    license = """
    prepare_data.py  Copyright (C) 2022  Jeremy Fix
    This program comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions;
    """
    logging.info(license)
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test", "stats"])

    parser.add_argument("--logdir", type=pathlib.Path, default="./logs")
    parser.add_argument("--commit_id", type=str, default=None)
    parser.add_argument("--datadir", type=pathlib.Path, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", choices=models.available_models, required=True)

    parser.add_argument("--loss", choices=["BCE", "Focal"], default="BCE")

    parser.add_argument("--normalize", action="store_true")
    parser.add_argument("--labelsmooth", type=float, default=None)
    parser.add_argument("--mixup", type=float, default=None)

    # Training parameters
    parser.add_argument("--logname", type=str, default=None)
    parser.add_argument("--class_weights", action="store_true", default=False)
    parser.add_argument("--val_ratio", type=float, default=0.1)
    parser.add_argument("--nepochs", type=int, default=50)
    parser.add_argument("--base_lr", type=float, default=3e-4)
    parser.add_argument("--weight_decay", type=float, default=1e-2)
    parser.add_argument("--batch_sampler", action="store_true", default=False)

    # Testing parameters
    parser.add_argument("--modelpath", type=pathlib.Path, default=None)

    args = parser.parse_args()
    exec(f"{args.command}(args)")
