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

# Standard imports
from typing import Optional
import random
import logging
import os

# External imports
import torch
import torch.utils.data
import torch.nn as nn
import torch.nn.functional as F
from sklearn.metrics import f1_score
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tqdm


def accuracy(probabilities: torch.Tensor, targets: torch.Tensor):
    """Computes the accuracy. Works with either PackedSequence or Tensor

    Args:
        probabiliities (torch.Tensor (B, C)) : The predicted probabilities
        targets (torch.Tensor (B,) ): The targets

    Returns:
        double: the accuracy
    """
    with torch.no_grad():
        if len(targets.shape) == 2:
            # For the LabelSmooth case
            targets = targets.argmax(axis=-1)
        return (probabilities.argmax(axis=-1) == targets).double().mean()


def f1_metric(
    model: torch.nn.Module, loader: torch.utils.data.DataLoader, device: torch.device
):
    """Computes the F1 score of a model over datas provided by a dataloader

    Args:
        model (nn.Module): The module through which to forward propagate
                           the inputs
        loader (torch.utils.data.DataLoader): a mappable dataloader
        device (torch.device): The device on which to compute

    Returns:
        macro_F1 (double) : the macro average F1 score
        class_F1 (double): the F1 score per class
    """
    with torch.no_grad():
        pred_labels, true_labels = [], []
        for (inputs, targets) in loader:
            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)

            predicted_labels = outputs.argmax(axis=-1).detach().tolist()
            pred_labels.extend(predicted_labels)
            if len(targets.shape) == 2:
                # For the LabelSmooth case
                targets = targets.argmax(axis=-1)
            true_labels.extend(targets.detach().tolist())
        macro_F1 = f1_score(true_labels, pred_labels, average="macro")
        class_F1 = f1_score(true_labels, pred_labels, average=None)
    return macro_F1, class_F1


def compute_class_weights(n_samples_per_class):
    """Compute class weights as min(counts)/counts_i

    Args:
        n_samples_per_class (dict[int]->int): the number of samples per class

    Returns:
        a torch.Tensor of the weights per class
    """
    # example count bounds for the training set
    # min 20, max : 305508
    logging.info("Computing class weights")
    num_classes = len(n_samples_per_class)
    # Note: be carefull, n_samples_for_class is a dictionnary with possibly
    # unordered int keys
    counts = torch.tensor([n_samples_per_class[i] for i in range(num_classes)])
    logging.info(f"Min class count : {min(counts)}, max : {max(counts)}")
    return 1.0 / (counts / min(counts))


def one_hot(
    labels: torch.Tensor,
    num_classes: int,
    device: Optional[torch.device] = None,
    dtype: Optional[torch.dtype] = None,
):
    """Returns the one-hot tensor of the provided labels

    Args:
        labels (torch.Tensor (B,) ): the labels
        num_classes (int) : the total number of classes
        device (torch.device): the device on which to create the tensor
        dtype (torch.dtype) : the datatype of the one hot tensor


    Returns:
        torch.Tensor (B, num_classes): one hot tensor

    """
    batch_size = labels.shape
    one_hot = torch.zeros(batch_size, num_classes, device=device, dtype=dtype)
    return one_hot.scatter_(1, labels.unsqueeze(1), 1.0)


def make_confusion_matrix(
    model: torch.Tensor,
    loader: torch.utils.data.DataLoader,
    device: torch.device,
    num_classes: int,
):
    """Builds the confusion matrix of the module over the data from the loader

    Args:
        model (torch.nn.Module) : the module to evaluate
        loader (torch.utils.data.DataLoader): the iterable over the data
        device (torch.device): the device on which to compute
        num_classes (int) : the total number of classes

    Returns:
        confusion_matrix (np.array): A prediction x predicted confusion matrix
    """
    with torch.no_grad():
        logging.info("Computing the confusion matrix")
        # pred x true
        confusion_matrix = np.zeros((num_classes, num_classes))
        for (X, y) in tqdm.tqdm(loader):
            X, y = X.to(device), y.to(device)
            pred = model(X).argmax(axis=1).to("cpu").tolist()
            for predi, truei in zip(pred, y):
                confusion_matrix[predi, truei] += 1
        # We normalize the confusion matrix along the "true" direction
        for truei in range(confusion_matrix.shape[1]):
            ntruei = confusion_matrix[:, truei].sum()
            if ntruei != 0:
                confusion_matrix[:, truei] /= ntruei
        return confusion_matrix


def plot_confusion_matrix(confusion_matrix: np.array):
    """Create a matplotlib figure and display a confusion matrix

    Args:
        confusion_matrix(np.array): The matrix to display

    Returns:
        plt.figure : the figure with the plotted confusion matrix
    """
    fig = plt.figure(figsize=(8, 6))
    plt.imshow(confusion_matrix, cmap=plt.get_cmap("Blues"))
    plt.colorbar()
    plt.ylabel("Predicted")
    plt.xlabel("True")
    return fig


class FocalLoss(nn.Module):
    """
    Implementation of the Focal Loss :math:`\\frac{1}{N} \\sum_i -(1-p_{y_i} + \\epsilon)^\\gamma \\log(p_{y_i})`

    Args:
        gamma: :math:`\\gamma > 0` puts more focus on hard misclassified samples

    Shape:
        - predictions :math:`(B, C)` : the logits
        - target :math`(B, )` : the target ids to predict
    """

    def __init__(self, gamma):
        super(FocalLoss, self).__init__()
        self.gamma = gamma
        self.eps = 1e-10

    def forward(self, predictions, target):
        probs = F.softmax(predictions, dim=1) + self.eps
        target_one_hot = one_hot(
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


def seed_everything(seed):
    """Set the seed on pipeline, so the results are the same at every time we run it
    This function is for the reproducibility
    """
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.benchmark = False
    torch.use_deterministic_algorithms(True)
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed)
    print(" > Seeding done ")
