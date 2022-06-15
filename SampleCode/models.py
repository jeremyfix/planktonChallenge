#!/usr/bin/env python3
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

import torch
import torch.nn as nn
import torchvision.models.resnet as resnet
import timm
import timm.models.layers


class LinearNet(nn.Module):
    def __init__(self, img_size, num_classes):
        super(LinearNet, self).__init__()
        dim = img_size[0] * img_size[1]
        self.model = nn.Linear(dim, num_classes)

    def forward(self, x):
        x = x.view(x.size()[0], -1)
        return self.model(x)


def conv_bn_relu(in_channels, out_channels, ks):
    return [
        nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size=ks,
            stride=1,
            padding=int((ks - 1) / 2),
            bias=True,
        ),
        nn.BatchNorm2d(out_channels),
        nn.ReLU(inplace=True),
    ]


def conv_block(fin, fout, ks):
    return [
        *conv_bn_relu(fin, fout, ks),
        *conv_bn_relu(fout, fout, ks),
        *conv_bn_relu(fout, fout, ks),
    ]


class SimpleCNNRF(nn.Module):
    def __init__(self, img_size, num_classes):
        super().__init__()
        self.features = nn.Sequential(
            *conv_block(1, 32, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(32, 64, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(64, 128, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(128, 128, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(128, 128, 5),
        )
        # Probe the output shape
        probe_tensor = torch.zeros((1, 1, img_size[0], img_size[1]))
        out = self.features(probe_tensor)
        dim = out.view(-1).shape[0]

        # Make the classifier head
        self.classifier = nn.Sequential(nn.Dropout2d(0.5), nn.Linear(dim, num_classes))

    def forward(self, x):
        self.feature_maps = []
        for layer in self.features:
            x = layer(x)
            self.feature_maps.append(x)
        x = x.view(x.size()[0], -1)
        return self.classifier(x)


class SimpleCNN(nn.Module):
    def __init__(self, img_size, num_classes):
        super(SimpleCNN, self).__init__()
        self.features = nn.Sequential(
            *conv_block(1, 32, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(32, 64, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(64, 128, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(128, 128, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(128, 128, 5),
        )
        # Probe the output shape
        probe_tensor = torch.zeros((1, 1, img_size[0], img_size[1]))
        out = self.features(probe_tensor)
        dim = out.view(-1).shape[0]

        # Make the classifier head
        self.classifier = nn.Sequential(nn.Dropout2d(0.5), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        return self.classifier(x)


class SimpleCNN2(nn.Module):
    def __init__(self, img_size, num_classes):
        super(SimpleCNN2, self).__init__()
        self.features = nn.Sequential(
            *conv_bn_relu(1, 64, 7),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(64, 64, 3),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(64, 128, 3),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(128, 512, 3),
            nn.AdaptiveAvgPool2d((1, 1)),
        )
        # Probe the output shape
        probe_tensor = torch.zeros((1, 1, img_size[0], img_size[1]))
        out = self.features(probe_tensor)
        print(f"Output shape of the convolutional part : {out.shape}")
        dim = out.view(-1).shape[0]

        # Make the classifier head
        self.classifier = nn.Sequential(nn.Dropout2d(0.5), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        return self.classifier(x)


class SimpleCNN3(nn.Module):
    def __init__(self, img_size, num_classes):
        super(SimpleCNN3, self).__init__()
        self.features = nn.Sequential(
            *conv_bn_relu(1, 64, 7),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(64, 64, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(64, 128, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(128, 256, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(256, 512, 5),
            nn.MaxPool2d(kernel_size=2),
            *conv_block(512, 1024, 5)
            # nn.AdaptiveAvgPool2d((1, 1))
        )
        # Probe the output shape
        probe_tensor = torch.zeros((1, 1, img_size[0], img_size[1]))
        out = self.features(probe_tensor)
        print(f"Output shape of the convolutional part : {out.shape}")
        dim = out.view(-1).shape[0]

        # Make the classifier head
        self.classifier = nn.Sequential(nn.Dropout2d(0.5), nn.Linear(dim, num_classes))

    def forward(self, x):
        x = self.features(x)
        x = x.view(x.size()[0], -1)
        return self.classifier(x)


def resnet18(img_size, num_classes):
    m = resnet.resnet18(pretrained=True, num_classes=1000)
    # Change the first convolutional layer because we have only 1 channel for the input
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Change the number of output classes
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def resnet50(img_size, num_classes):
    m = resnet.resnet50(pretrained=True, num_classes=1000)
    # Change the first convolutional layer because we have only 1 channel for the input
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Change the number of output classes
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def resnet152(img_size, num_classes):
    m = resnet.resnet152(pretrained=True, num_classes=1000)
    # Change the first convolutional layer because we have only 1 channel for the input
    m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
    # Change the number of output classes
    m.fc = nn.Linear(m.fc.in_features, num_classes)
    return m


def efficientnet_b3a(img_size, num_classes):
    m = timm.create_model("efficientnet_b3a", pretrained=True)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    m.conv_stem = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
    return m


def efficientnet_b3(img_size, num_classes):
    m = timm.create_model("efficientnet_b3", pretrained=True)
    m.classifier = nn.Linear(m.classifier.in_features, num_classes)
    m.conv_stem = nn.Conv2d(1, 40, kernel_size=3, stride=2, padding=1, bias=False)
    return m


# def efficientnet_b7(model_name, img_size, num_classes):
#     m = torchvision.models.efficientnet_b7(pretrained=True, num_classes=1000)
#     print(m.features[0])
#     # Change the first convolutional layer because we have only 1 channel for the input
#     # m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     # Change the number of output classes
#     # m.fc = nn.Linear(m.fc.in_features, num_classes)
#     return m

# def regnet_y_8gf(model_name, img_size, num_classes):
#     m = torchvision.models.regnet_y_8gf(pretrained=True, num_classes=1000)
#     # Change the first convolutional layer because we have only 1 channel for the input
#     # m.conv1 = nn.Conv2d(1, 64, kernel_size=7, stride=2, padding=3, bias=False)
#     print(m.stem)
#     # m.stem
#     # Change the number of output classes
#     m.fc = nn.Linear(m.fc.in_features, num_classes)
#     return m


def regnety_016(img_size, num_classes):
    m = timm.create_model("regnety_016", pretrained=True)
    m.head.fc = nn.Linear(m.head.fc.in_features, num_classes)
    m.stem.conv = nn.Conv2d(1, 32, kernel_size=3, stride=2, padding=1, bias=False)
    return m


def cait_s24_224(img_size, num_classes):
    m = timm.create_model("cait_s24_224", pretrained=True)
    assert img_size[0] == img_size[1]
    m.head = nn.Linear(m.head.in_features, num_classes)
    ope = m.patch_embed

    m.patch_embed = timm.models.layers.PatchEmbed(
        img_size[0], ope.patch_size, in_chans=1, embed_dim=ope.proj.out_channels
    )
    return m


available_models = [
    "LinearNet",
    "SimpleCNN",
    "SimpleCNN2",
    "SimpleCNN3",
    "resnet18",
    "resnet50",
    "resnet152",
    "regnety_016",
    "efficientnet_b3a",
    "efficientnet_b3",
    "cait_s24_224",
]


def build_model(model_name, img_size, num_classes):
    if model_name not in available_models:
        raise RuntimeError(f"Unavailable model {model_name}")
    exec(f"m = {model_name}(img_size, num_classes)")
    return locals()["m"]
    # elif model_name == 'regnet_y_8gf':
    #     return regnet_y_8gf('regnet_y_8gf', img_size, num_classes)
    # elif model_name == 'efficientnet_b7':
    #     return efficientnet_b7('efficientnet_b7', img_size, num_classes)


if __name__ == "__main__":
    # from receptivefield.pytorch import PytorchReceptiveField
    # img_size = (300, 300)
    # num_classes = 86
    # def model_fn():
    #     model = SimpleCNNRF(img_size, num_classes)
    #     model.eval()
    #     return model
    # rf = PytorchReceptiveField(model_fn)
    # rf_params = rf.compute(input_shape = img_size + (1, ))

    mname = ["cait_s24_224", "efficientnet_b3"]

    for n in mname:
        m = build_model(n, (224, 224), 85)
        out = m(torch.zeros(2, 1, 224, 224))
        print(out.shape)
