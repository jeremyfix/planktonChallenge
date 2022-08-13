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
import logging
import pathlib
import os
import argparse
import tqdm
import random
from collections import defaultdict

# External imports
import torchvision
import torch
from torch.distributions import Beta
import numpy as np
import matplotlib as mpl

mpl.use("Agg")
import matplotlib.pyplot as plt
from typing import Any, Callable, Optional
from PIL import ImageOps
import cv2
import albumentations as A
from albumentations.core.transforms_interface import ImageOnlyTransform
from albumentations.pytorch import ToTensorV2

__TRAIN_URL = "http://infomob.metz.supelec.fr/fix/ChallengeDeep/train.tar.gz"
__TEST_URL = "http://infomob.metz.supelec.fr/fix/ChallengeDeep/test.tar.gz"


class SquareResize(ImageOnlyTransform):
    def __init__(self, target_size: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.target_size = target_size

    def get_transform_init_args_names(self):
        return ["target_size"]

    def apply(self, img, **params):
        longest_size = max(*img.shape[:2])  # img is (H, W, C)
        img = img.astype(np.float32)
        transforms = [
            A.PadIfNeeded(
                min_height=longest_size,
                min_width=longest_size,
                value=255,
                always_apply=True,
            ),
            A.Resize(
                height=self.target_size, width=self.target_size, always_apply=True
            ),
        ]
        transform = A.Compose(transforms)
        return transform(image=img)["image"]


class MyDataset(torch.utils.data.Dataset):
    """
    Wrapper class around the ImageFolder dataset to add the static features as
    well
    """

    def __init__(
        self,
        root: str,
        transform: Optional[Callable] = None,
    ) -> None:
        root_path = pathlib.Path(root)
        if not root_path.exists():
            # TODO: download the data
            # and extract the archive
            print(f"Path {root_path} does not exist")
        imgdataset = torchvision.datasets.ImageFolder(root)
        self.imgdataset = AugmentedDataset(imgdataset, transform)

    def __getitem__(self, index: int) -> Any:
        return self.imgdataset[index]

    @property
    def samples(self):
        return self.imgdataset.samples

    @property
    def classes(self):
        return self.imgdataset.classes

    @property
    def class_to_idx(self):
        return self.imgdataset.class_to_idx

    def __len__(self) -> int:
        return len(self.imgdataset)

    def __repr__(self) -> str:
        return self.imgdataset.__repr__()


class LabelSmoothedDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, eps, num_classes):
        super().__init__()
        self.eps = eps
        self.base_dataset = base_dataset
        self.num_classes = num_classes

    def __getitem__(self, index: int) -> Any:
        x, y = self.base_dataset[index]
        # Label smooth the label
        one_hot = self.eps / (self.num_classes - 1) * torch.ones(self.num_classes)
        one_hot[y] = 1.0 - self.eps
        return x, one_hot

    @property
    def samples(self):
        return self.base_dataset.samples

    @property
    def classes(self):
        return self.base_dataset.classes

    @property
    def class_to_idx(self):
        return self.base_dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __repr__(self) -> str:
        return self.base_dataset.__repr__()


class MixUpDataset(torch.utils.data.Dataset):
    def __init__(self, base_dataset, alpha, num_classes):
        super().__init__()
        self.base_dataset = base_dataset
        self.num_classes = num_classes
        self.distrib = Beta(alpha, alpha)

    def __getitem__(self, index: int) -> Any:
        x1, y1 = self.base_dataset[index]
        x2, y2 = self.base_dataset[random.randint(0, len(self.base_dataset) - 1)]
        lbd = self.distrib.sample()
        # Mix the samples
        one_hot1 = torch.zeros(self.num_classes)
        one_hot1[y1] = 1.0
        one_hot2 = torch.zeros(self.num_classes)
        one_hot2[y2] = 1.0

        x = lbd * x1 + (1 - lbd) * x2
        y = lbd * one_hot1 + (1 - lbd) * one_hot2
        return x, y

    @property
    def samples(self):
        return self.base_dataset.samples

    @property
    def classes(self):
        return self.base_dataset.classes

    @property
    def class_to_idx(self):
        return self.base_dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.base_dataset)

    def __repr__(self) -> str:
        return self.base_dataset.__repr__()


def load_raw_data(datadir):
    return MyDataset(root=datadir)


def load_dataset(
    datadir: pathlib.Path, transform: Callable, val_ratio: float, stratified: bool
):
    dataset = MyDataset(root=datadir, transform=transform)
    logging.info(dataset)

    if not stratified:
        indices = list(range(len(dataset)))
        num_data = len(dataset)
        num_valid = int(val_ratio * num_data)
        num_train = num_data - num_valid

        np.random.shuffle(indices)
        train_indices = indices[:num_train]
        valid_indices = indices[num_train:]
    else:
        # Browse all the dataset for giving the right proportion of samples
        # per class
        # As we do not know the number of classes, we use a dictionnary
        class_indices = defaultdict(list)
        logging.info("Collecting the samples")
        for i, (_, yi) in tqdm.tqdm(enumerate(dataset)):
            class_indices[yi].append(i)

        # Random assign the indices for the train and valid folds
        train_indices, valid_indices = [], []
        logging.info("Splitting the samples between the folds")
        for ci, idxi in tqdm.tqdm(class_indices.items()):
            num_data = len(idxi)
            num_valid = int(val_ratio * num_data)
            num_train = num_data - num_valid

            np.random.shuffle(idxi)
            train_indices.extend(idxi[:num_train])
            valid_indices.extend(idxi[num_train:])

    train_dataset = torch.utils.data.Subset(dataset, train_indices)
    valid_dataset = torch.utils.data.Subset(dataset, valid_indices)
    return train_dataset, valid_dataset


def load_preprocessed_data(
    datadir,
    train_transform,
    valid_transform,
    batch_size,
    num_workers,
    pin_memory,
    label_smooth=None,
    mixup=None,
    batch_sampler=False,
):
    """
    This function loads the data, supposed split in datadir/train datadir/valid
    outputs the dataloaders and the counts per class in the training set
    """
    # The data are already split in train/val
    # and already cropped/padded/resized
    train_dir = datadir / "train"
    valid_dir = datadir / "valid"

    train_dataset = torchvision.datasets.ImageFolder(train_dir)
    train_dataset = AugmentedDataset(train_dataset, train_transform)

    if label_smooth is not None:
        train_dataset = LabelSmoothedDataset(
            train_dataset, label_smooth, len(train_dataset.classes)
        )
    if mixup is not None:
        train_dataset = MixUpDataset(train_dataset, mixup, len(train_dataset.classes))

    # Count the numer of samples per class
    num_classes = len(train_dataset.classes)
    n_samples_per_class = {"train": [0] * num_classes, "valid": [0] * num_classes}
    Ntrain = 0
    for classdir in train_dir.glob("*"):
        cls = int(classdir.name)
        n_samples_per_class["train"][cls] = sum(1 for _ in classdir.glob("*.jpg"))
        Ntrain += n_samples_per_class["train"][cls]
    for classdir in valid_dir.glob("*"):
        cls = int(classdir.name)
        n_samples_per_class["valid"][cls] = sum(1 for _ in classdir.glob("*.jpg"))

    # Use these counts to bias batch sampling
    class_weights = torch.zeros((len(n_samples_per_class["train"])))
    for cls, num in enumerate(n_samples_per_class["train"]):
        class_weights[cls] = 1.0 / num if num != 0 else 0.0
    if batch_sampler:
        sample_weights = torch.zeros((sum(n_samples_per_class["train"]),))
        for i, (_, y) in enumerate(iter(train_dataset)):
            if isinstance(y, int):
                sample_weights[i] = class_weights[y]
            else:
                # in case of label smooth
                sample_weights[i] = class_weights[y.argmax()]
    else:
        sample_weights = torch.ones((sum(n_samples_per_class["train"]),))

    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights)
    )

    logging.info(f"Using a sampler with sample_weights : {sample_weights}")

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        # shuffle=True,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_dataset = torchvision.datasets.ImageFolder(valid_dir)
    valid_dataset = AugmentedDataset(valid_dataset, valid_transform)
    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader, n_samples_per_class


def load_data(datadir, transform, val_ratio, batch_size, num_workers, pin_memory):

    train_dataset, valid_dataset = load_dataset(datadir, transform, val_ratio)

    train_loader = torch.utils.data.DataLoader(
        train_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    valid_loader = torch.utils.data.DataLoader(
        valid_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )

    return train_loader, valid_loader


def load_test_data(datadir, transform, batch_size, num_workers, pin_memory):
    test_dataset = torchvision.datasets.ImageFolder(datadir)
    test_dataset = AugmentedDataset(test_dataset, transform)
    loader = torch.utils.data.DataLoader(
        test_dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=pin_memory,
    )
    return loader


def get_stats(args):
    dataset = MyDataset(args.datadir)
    widths = []
    heights = []

    logging.info(dataset)
    for (ki, (img, label)) in tqdm.tqdm(enumerate(dataset)):

        widths.append(img.width)
        heights.append(img.height)
    widths, heights = np.array(widths), np.array(heights)

    logging.info(
        f"Size stats : \n [{widths.min()}<width<{widths.max()}] x [{heights.min()}<height<{heights.max()}]"
    )

    plt.figure()
    bins = np.arange(
        min(widths.min(), heights.min()), max(widths.max(), heights.max()), 50
    )
    plt.hist([widths, heights], bins, alpha=0.5, label=["width", "height"])
    plt.legend(loc="upper right")
    plt.savefig("sizes.pdf", bbox_inches="tight")
    plt.show()


def test_dataset(args):

    imgidx = 20
    dataset = MyDataset(args.datadir, transform=Resize(__default_size[0]))
    img, label = dataset[imgidx]
    print(dataset.samples[imgidx])
    print(len(dataset))
    print(type(img), label)
    img.show()
    # img.write('sample.png')
    img.save("sample.png")


def test_transforms(args):
    train_transform = A.Compose([A.ToGray(always_apply=True), ToTensorV2()])
    train_dir = args.datadir / "train"
    train_dataset1 = torchvision.datasets.ImageFolder(
        train_dir, transform=ImageOps.grayscale
    )
    train_dataset2 = AugmentedDataset(train_dataset1, train_transform)
    data_in, data_out = train_dataset2[0]
    print(data_in.shape)
    tf = A.ToGray(always_apply=True)
    data_in, data_out = train_dataset1[0]
    print(type(data_in))
    data_tf = tf(image=np.array(data_in))["image"]
    print(data_tf.shape)


def test_dataloader(args):
    train_loader, valid_loader = load_data(
        args.datadir,
        transform=__default_transform,
        val_ratio=0.1,
        batch_size=64,
        num_workers=4,
        pin_memory=False,
    )
    _ = load_test_data(
        pathlib.Path(args.datadir) / ".." / "pubtest",
        transform=__default_transform,
        batch_size=64,
        num_workers=4,
        pin_memory=False,
    )
    X, y = next(iter(train_loader))
    fig, axes = plt.subplots(4, 4, figsize=(10, 4))
    axes = axes.ravel()
    for i, ax in enumerate(axes):
        ax.imshow(X[i, 0], cmap="gray")
        ax.set_title(train_loader.dataset.classes[y[i]])
        ax.axis("off")
    plt.tight_layout()
    plt.savefig("samples.pdf")
    plt.show()


class AugmentedDataset(torch.utils.data.Dataset):
    def __init__(self, dataset, transform):
        self.dataset = dataset
        self.transform = transform

    def __getitem__(self, idx):
        img, label = self.dataset[idx]
        img = np.array(img)
        img = self.transform(image=img)["image"]
        return img, label

    @property
    def samples(self):
        return self.dataset.samples

    @property
    def classes(self):
        return self.dataset.classes

    @property
    def class_to_idx(self):
        return self.dataset.class_to_idx

    def __len__(self) -> int:
        return len(self.dataset)

    def __repr__(self) -> str:
        return self.dataset.__repr__() + str(f"\n Transform : {self.transform}")


class ScaleBrightness(ImageOnlyTransform):
    def __init__(self, scale_range, max_pixel_value, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.scale_range = scale_range
        self.max_pixel_value = max_pixel_value

    def get_transform_init_args_names(self):
        return ["scale_range"]

    def apply(self, img, **params):
        f = self.scale_range[0] + np.random.random() * (
            self.scale_range[1] - self.scale_range[0]
        )
        return self.max_pixel_value - np.uint8((self.max_pixel_value - img) * f)


class ScaleData(ImageOnlyTransform):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def get_transform_init_args_names(self):
        return []

    def apply(self, img, **params):
        res = img / 255.0
        return res


class KeepChannel(ImageOnlyTransform):
    def __init__(self, chan_idx, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.chan_idx = chan_idx

    def get_transform_init_args_names(self):
        return ["chan_idx"]

    def apply(self, img, **params):
        img2 = img[:, :, self.chan_idx]
        return img2


def test_augmentations(args):
    num_img = 36
    transform = A.Compose(
        [
            __default_transform,
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.MotionBlur(),
            A.CoarseDropout(fill_value=1.0, max_height=20, max_width=20),
            A.Rotate(180, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=1.0),
            ScaleBrightness(scale_range=(0.8, 1.0)),
            # A.Normalize((0.92,), (0.16,)),
            # ToTensorV2(),
        ]
    )
    dataset = MyDataset(root=args.datadir, transform=transform)

    fig = plt.figure()
    for i in range(6):
        img, _ = dataset[num_img]
        print(img.min(), img.max())
        ax = fig.add_subplot(6, 6, i + 1)
        ax.imshow(img, cmap="gray")
        ax.axis("off")
    plt.show()


def test_augmented_dataloader(args):
    transforms = A.Compose(
        [
            A.HorizontalFlip(),
            A.VerticalFlip(),
            A.MotionBlur(),
            A.CoarseDropout(fill_value=(255, 255, 255), max_height=20, max_width=20),
            A.Rotate(
                180, p=0.5, border_mode=cv2.BORDER_CONSTANT, value=(255, 255, 255)
            ),
            ScaleBrightness(scale_range=(0.8, 1.0)),
            KeepChannel(0, always_apply=True),
            A.Normalize((0.92,), (0.16,)),
            ToTensorV2(),
        ]
    )
    dataset = torchvision.datasets.ImageFolder(args.datadir)
    dataset = AugmentedDataset(dataset, transforms)
    for i in range(10):
        print(dataset[i][0].shape)
    loader = torch.utils.data.DataLoader(dataset, batch_size=32)

    X, y = next(iter(loader))
    print(X.shape)


def test_preprocessed(args):

    batch_size = 16
    train_transforms = [A.ToGray(), A.HorizontalFlip()]
    train_transforms = A.Compose(train_transforms)
    valid_transforms = [A.ToGray()]
    valid_transforms = A.Compose(valid_transforms)
    num_workers = 4
    train_loader, valid_loader, n_samples_per_class = load_preprocessed_data(
        args.datadir,
        train_transforms,
        valid_transforms,
        batch_size,
        num_workers,
        pin_memory=True,
    )


def test_sampler(args):
    transforms = A.Compose([ToTensorV2()])

    dataset = torchvision.datasets.ImageFolder(args.datadir)
    dataset = AugmentedDataset(dataset, transforms)
    batch_size = 32

    num_classes = len(dataset.classes)
    print(f"I found {num_classes} classes")
    n_samples_per_class = [0] * num_classes
    # Compute the number of samplers per class
    for classdir in args.datadir.glob("*"):
        cls = int(classdir.name)
        n_samples_per_class[cls] = sum(1 for _ in classdir.glob("*.jpg"))
    print(f"The num samples per class is: {n_samples_per_class}")

    class_weights = torch.zeros((len(n_samples_per_class)))
    for cls, num in enumerate(n_samples_per_class):
        class_weights[cls] = 1.0 / num if num != 0 else 0.0
    # class_weights /= class_weights.sum()
    sample_weights = torch.zeros((sum(n_samples_per_class),))
    for i, (_, y) in enumerate(iter(dataset)):
        sample_weights[i] = class_weights[y]

    # logging.info(f'Using a sampler with weights : {weights}')
    sampler = torch.utils.data.sampler.WeightedRandomSampler(
        sample_weights, len(sample_weights)
    )
    loader = torch.utils.data.DataLoader(
        dataset, batch_size=batch_size, sampler=sampler
    )

    # Let us compute statistics on the sampled minibatches
    n_samples_drawn = torch.zeros((num_classes,))
    num_samples = 0
    for X, y in loader:
        for yi in y:
            n_samples_drawn[yi] += 1
        num_samples += X.shape[0]

    print(num_samples)
    print(f"I had originally {len(dataset)} samples")
    class_prob = np.array(n_samples_per_class)
    class_prob = class_prob / class_prob.sum()
    print(f"where the class probabilities are {class_prob}")
    print(f"I sampled a total of {n_samples_drawn.sum().item()} samples")
    n_samples_drawn /= n_samples_drawn.sum()
    print(f"where the class probabilities are {n_samples_drawn}")


__default_size = (300, 300)
__default_preprocess_transform = SquareResize(__default_size[0], always_apply=True)

__default_transform = A.Compose(
    [
        __default_preprocess_transform,
        KeepChannel(0, always_apply=True),
        ScaleData(always_apply=True),
    ]
)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--datadir", type=pathlib.Path, default="/mounts/Datasets1/ChallengeDeep/test"
    )
    parser.add_argument("--batch_size", default=64)
    parser.add_argument("--classidx", type=int, default=None)

    args = parser.parse_args()

    # get_stats(args)
    # test_dataset(args)
    # test_transforms(args)
    # test_dataloader(args)
    # print(len(dataloader))

    test_augmentations(args)
    # test_augmented_dataloader(args)
    # test_sampler(args)
    # test_preprocessed(args)
