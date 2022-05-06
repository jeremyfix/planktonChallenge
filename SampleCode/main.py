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
import torch.nn.functional as F
import torch.optim
from torch.utils.tensorboard import SummaryWriter
import torchvision.transforms as transforms
import pandas as pd
import cv2
from PIL import ImageOps
import albumentations as A
from albumentations.pytorch import ToTensorV2

# Local modules
import data
import models
import utils


class FocalLoss(nn.Module):
    def __init__(self, weights, gamma):
        super(FocalLoss, self).__init__()
        self.weights = weights
        self.gamma = gamma
        self.eps = 1e-10

    def forward(self, predictions, target):
        probs = F.softmax(predictions, dim=1) + self.eps
        target_one_hot = utils.one_hot(
            target,
            num_classes=predictions.shape[1],
            device=predictions.device,
            dtype=predictions.dtype,
        )

        weight = torch.pow(1.0 - probs, self.gamma)
        focal = -weight * torch.log(probs)
        loss_tmp = torch.sum(target_one_hot * focal, dim=1)
        loss = loss_tmp.mean()

        return loss


def stats(args):
    print("Stats")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Get the data
    loaders = data.load_preprocessed_data(
        args.datadir,
        train_transform=transforms.Compose([ImageOps.grayscale, transforms.ToTensor()]),
        valid_transform=data.__default_preprocessed_transform,
        # val_ratio=args.val_ratio,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
    train_loader, valid_loader, n_samples_per_class = loaders
    mean_pix, std_pix, num_imgs = 0, 0, 0
    for X, _ in train_loader:
        X = X.to(device)
        mean_pix += X.shape[0] * X.mean().item()  # Mean over num_pixels
        num_imgs += X.shape[0]
    mean_pix /= num_imgs  # Mean over num_pixels x batch_size
    for X, _ in train_loader:
        X = X.to(device)
        std_pix += (
            X.shape[0] * ((X - mean_pix) ** 2).mean().item()
        )  # Mean over num_pixels
    std_pix /= num_imgs
    std_pix = math.sqrt(std_pix)
    print(f"Mean pixel value : {mean_pix} (std {std_pix})")


def train(args):
    print("Training")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    # Get the data
    # train_transforms = [
    #     ImageOps.grayscale,
    #     transforms.ToTensor()
    # ]
    # if args.normalize:
    #     train_transforms.append(transforms.Normalize((0.92, ), (0.16, )))
    # train_transforms.append(transforms.RandomRotation(180, fill=1.0))
    # train_transforms = transforms.Compose(train_transforms)

    train_transforms = [
        A.HorizontalFlip(),
        A.VerticalFlip(),
        A.MotionBlur(),
        A.CoarseDropout(fill_value=(255, 255, 255), max_height=20, max_width=20),
        A.Rotate(180, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)),
        data.ScaleBrightness(scale_range=(0.8, 1.0)),
        data.KeepChannel(0, always_apply=True),
    ]
    if args.normalize:
        train_transforms.append(A.Normalize((0.92,), (0.16,)))
    train_transforms.append(ToTensorV2())
    train_transforms = A.Compose(train_transforms)

    valid_transforms = [ImageOps.grayscale, transforms.ToTensor()]
    if args.normalize:
        valid_transforms.append(transforms.Normalize((0.92,), (0.16,)))
    valid_transforms = transforms.Compose(valid_transforms)

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
    )
    train_loader, valid_loader, n_samples_per_class = loaders

    num_classes = len(train_loader.dataset.classes)
    print(num_classes)

    # Check the number of samples per class
    # logging.info('Samples counting')
    # n_samples_per_class = {
    #     'train': [0]*num_classes,
    #     'valid': [0]*num_classes
    # }
    # for _, y in tqdm.tqdm(train_loader):
    #     for i in y:
    #         n_samples_per_class['train'][i] += 1
    # for _, y in tqdm.tqdm(valid_loader):
    #     for i in y:
    #         n_samples_per_class['valid'][i] += 1
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
        loss = utils.FocalLoss(class_weights, gamma=0.5)

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

    if not os.path.isdir(logdir):
        os.makedirs(logdir)
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
    print(summary_text)

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
        print(
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

        print(f"Class F1 : {class_F1}")
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
    if not args.modelpath:
        raise ValueError("The modelpath must be provided")

    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda") if use_cuda else torch.device("cpu")

    logging.info("Loading the data")
    test_transforms = [
        data.Resize(data.__default_size[0]),
        ImageOps.grayscale,
        transforms.ToTensor(),  # ,
    ]
    if args.normalize:
        test_transforms.append(transforms.Normalize((0.92,), (0.16,)))
    test_transforms = transforms.Compose(test_transforms)

    loader = data.load_test_data(
        args.datadir,
        transform=test_transforms,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        pin_memory=use_cuda,
    )
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
        probs = model(X).detach().to("cpu")
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
    license = """
    prepare_data.py  Copyright (C) 2022  Jeremy Fix
    This program comes with ABSOLUTELY NO WARRANTY;
    This is free software, and you are welcome to redistribute it
    under certain conditions;
    """
    print(license)

    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=["train", "test", "stats"])

    parser.add_argument("--logdir", type=pathlib.Path, default="./logs")
    parser.add_argument("--commit_id", type=str, default=None)
    parser.add_argument("--datadir", type=pathlib.Path, required=True)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--model", choices=models.available_models)

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

    # Testing parameters
    parser.add_argument("--modelpath", type=pathlib.Path, default=None)

    args = parser.parse_args()
    exec(f"{args.command}(args)")
