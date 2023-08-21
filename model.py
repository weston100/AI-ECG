"""Objects for defining AI-ECG models."""

import torch
from torch import nn
import torchvision
import numpy as np
import sklearn.metrics
import scipy.special
from torchinfo import summary

from model_specs import specs, two_d_specs


class ECGModel(torch.nn.Module):
    """
    A convolutional neural network for ingesting ECG data.

    Arguments:
        cfg: The config dict, like config.py's cfg["model"].
        num_input_channels: The number of channels (ECG leads).
        num_outputs: The number of model outputs.
        binary: Whether the output of the model is one or more binary predictions.
        score: Optionally, a score function to rank models by, taking (y, yh, loss)
            as input.

    This object builds 1- and 2-d backbones based on model_specs.py, where other 
    architectures can be easily implemented.
    """

    def __init__(self, cfg, num_input_channels, num_outputs, binary, score=None):
        super(ECGModel, self).__init__()
        self.cfg = cfg
        self.out_channels = num_outputs
        self.binary = binary
        self.num_channels_for_adaptive_2d = 12

        if self.cfg["is_2d"]:
            self.num_input_channels = 1
            self.features, in_channels = self.make_2d_conv()
        else:
            self.num_input_channels = num_input_channels
            self.features, in_channels = self.make_conv()

        self.classifier = self.make_fc(in_channels)

        if self.binary:
            self.score = score or (lambda y, yh, loss: sklearn.metrics.roc_auc_score(y, yh))
            w = torch.from_numpy(np.array([cfg["pos_weight"]])).float()
            self.loss = torch.nn.BCEWithLogitsLoss(pos_weight=w)
        else:
            self.score = score or (lambda y, yh, loss: sklearn.metrics.r2_score(y, yh))
            self.loss = torch.nn.MSELoss()

        self.float()

        num_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        print(self, flush=True)
        print(f"num params: {num_params}", flush=True)

    def forward(self, x):
        if self.cfg["is_2d"]:
            x = x.unsqueeze(1)
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x

    def train_step(self, x, y):
        y_hat = self.forward(x)
        loss = self.loss(y_hat, y)
        return (y, y_hat, loss)

    def make_conv(self):
        # ("C", out_channels, kernel_size=cfg["conv_width"], padding=1) # Conv block
        # ("B", out_channels, downsample=False) # ResNet block
        # ("L", out_channels, num_blocks=1) # ResNet Layer (num_blocks blocks, first downsamples)
        # ("R", nbs) # Whole ResNet core
        # ("m", kernel_size=2) # Max pool
        # ("c", out_channels, kernel_size=cfg["conv_width"]) # Conv
        # ("b") # Batch norm
        # ("r") # ReLU
        # ("a", bins=1) # Adaptive average pooling

        spec = specs[self.cfg["model_type"]][0]
        in_channels = self.num_input_channels

        layers = []
        for d in spec:
            if type(d) == int:
                d = ("C", d, self.cfg["conv_width"], 1)
            elif type(d) == str:
                d = (d)

            if d[0] == "C":
                # Conv block
                self.check_length(d, [2, 3, 4])
                if len(d) == 2:
                    d = (d[0], d[1], self.cfg["conv_width"], 1)
                if len(d) == 3:
                    d = (d[0], d[1], d[2], 1)
                conv = nn.Conv1d(in_channels, d[1], kernel_size=d[2], padding=d[3])
                if self.cfg["batch_norm"]:
                    layers += [conv, nn.BatchNorm1d(d[1]), nn.ReLU(inplace=False)]
                else:
                    layers += [conv, nn.ReLU(inplace=False)]
                in_channels = d[1]

            elif d[0] == "B":
                # ResNet block
                self.check_length(d, [2, 3])
                if len(d) == 2:
                    d = (d[0], d[1], False)
                layers += [Block(in_channels, d[1], d[2], batch_norm=self.cfg["batch_norm"])]
                in_channels = d[1]

            elif d[0] == "L":
                # ResNet layer
                self.check_length(d, [2, 3])
                if len(d) == 2:
                    d = (d[0], d[1], 1)
                layers += self.make_layer(in_channels, d[1], d[2], self.cfg["conv_width"], batch_norm=self.cfg["batch_norm"])
                in_channels = d[1]

            elif d[0] == "R":
                # Whole ResNet core
                self.check_length(d, [2])
                nbs = d[1]
                layers += self.make_layer(in_channels, 64, nbs[0], self.cfg["conv_width"], batch_norm=self.cfg["batch_norm"])
                layers += self.make_layer(64, 128, nbs[1], self.cfg["conv_width"], batch_norm=self.cfg["batch_norm"])
                layers += self.make_layer(128, 256, nbs[2], self.cfg["conv_width"], batch_norm=self.cfg["batch_norm"])
                layers += self.make_layer(256, 512, nbs[3], self.cfg["conv_width"], batch_norm=self.cfg["batch_norm"])
                in_channels = 512

            elif d[0] == "N":
                # Whole ResNet Bottleneck core
                self.check_length(d, [2])
                nbs = d[1]
                layers += self.make_layer(in_channels, 256, nbs[0], self.cfg["conv_width"], block=BottleneckBlock, batch_norm=self.cfg["batch_norm"])
                layers += self.make_layer(256, 512, nbs[1], self.cfg["conv_width"], block=BottleneckBlock, batch_norm=self.cfg["batch_norm"])
                layers += self.make_layer(512, 1024, nbs[2], self.cfg["conv_width"], block=BottleneckBlock, batch_norm=self.cfg["batch_norm"])
                layers += self.make_layer(1024, 2048, nbs[3], self.cfg["conv_width"], block=BottleneckBlock, batch_norm=self.cfg["batch_norm"])
                in_channels = 2048

            elif d[0] == "m":
                # max pool
                self.check_length(d, [1, 2])
                if len(d) == 1:
                    d = (d[0], 2)
                layers += [nn.MaxPool1d(kernel_size=d[1], stride=d[1])] 

            elif d[0] == "c":
                # single conv
                self.check_length(d, [2, 3])
                if len(d) == 2:
                    d = (d[0], d[1], self.cfg["conv_width"])
                layers += [nn.Conv1d(in_channels, d[1], kernel_size=d[2], padding=1)]
                in_channels = d[1]

            elif d[0] == "b":
                # batch norm
                if self.cfg["batch_norm"]:
                    self.check_length(d, [1])
                    layers += [nn.BatchNorm1d(in_channels)]

            elif d[0] == "r":
                # relu
                self.check_length(d, [1])
                layers += [nn.ReLU(inplace=False)]

            elif d[0] == "a":
                # Adaptive average pooling
                self.check_length(d, [1, 2])
                if len(d) == 1:
                    d = (d[0], 1)
                layers += [nn.AdaptiveAvgPool1d(output_size=d[1])]          
                in_channels = d[1] * in_channels

            elif d[0] == "d":
                self.check_length(d, [1])
                layers += [nn.Dropout(self.drop_prob)]
            else:
                raise NotImplementedError(d)


        return nn.Sequential(*layers), in_channels

    def make_2d_conv(self):

        spec = two_d_specs[self.cfg["model_type"]][0]
        in_channels = self.num_input_channels

        layers = []
        for d in spec:
            if type(d) == str:
                d = (d,)
            elif type(d) == int:
                d = ("C", int(d), (int(cfg["conv_width"]), int(cfg["conv_width"]),), (1, 1),)
            if d[0] == "l":
                if len(d) == 1:
                    d = (d[0], 64,)
                d = ("C", d[1], (self.num_channels_for_adaptive_2d, 1), (0, 0))
            if d[0] == "C":
                if len(d) == 3:
                    d = (d[0], d[1], d[2], (1, 1),)
                if type(d[2]) == int:
                    d = (d[0], d[1], (d[2], d[2],), d[3])
                conv = nn.Conv2d(in_channels, d[1], kernel_size=d[2], padding=d[3])
                if self.cfg["batch_norm"]:
                    layers += [conv, nn.BatchNorm2d(d[1]), nn.ReLU(inplace=True)]
                else:
                    layers += [conv, nn.ReLU(inplace=True)]
                in_channels = d[1]
            elif d[0] == "d":
                layers += [nn.Dropout(self.drop_prob)]
            elif d[0] == "m":
                if len(d) == 1:
                    d = (d[0], (2, 2))
                layers += [nn.MaxPool2d(kernel_size=d[1], stride=d[1])]
            elif d[0] == "a":
                # Adaptive average pooling
                if len(d) == 1:
                    d = (d[0], (3,3,))
                layers += [nn.AdaptiveAvgPool2d(output_size=d[1])]          
                in_channels = np.prod(d[1]) * in_channels
            else:
                raise NotImplementedError(d)

        return nn.Sequential(*layers), in_channels

    def make_fc(self, in_channels):

        if self.cfg["is_2d"]:
            spec = two_d_specs[self.cfg["model_type"]][1]
        else:
            spec = specs[self.cfg["model_type"]][1]

        spec = spec.copy()
        spec.append(self.out_channels)

        layers = []
        for d in spec:
            if type(d) == int:
                layers.append(nn.Linear(in_channels, d))
                in_channels = d
            elif d == "r":
                layers.append(nn.ReLU(inplace=True))
            elif d == "d":
                layers.append(nn.Dropout(self.cfg["drop_prob"]))
            elif d == "b":
                layers.append(nn.BatchNorm1d(in_channels))
            elif type(d) == tuple and d[0] == "d":
                layers.append(nn.Dropout(d[1]))
            else:
                print(d, type(d))
                raise NotImplementedError(d)
        return nn.Sequential(*layers)

    def check_length(self, obj, lengths):
        assert len(obj) in lengths, "{} is malformed".format(d)

    def make_layer(self, in_channels, out_channels, n_blocks, kernel_size=3, block=None, batch_norm=False):
        if block is None:
            block = Block
        blocks = [block(in_channels, out_channels, average_pool=True, kernel_size=kernel_size, batch_norm=batch_norm)]
        blocks += [block(out_channels, out_channels, kernel_size=kernel_size, batch_norm=batch_norm) for _ in range(1, n_blocks)]
        return nn.Sequential(*blocks)


class Block(nn.Module):
    """An ECG ResNet Block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, average_pool=False, batch_norm=True):
        super(Block, self).__init__()
        if batch_norm:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels,
                          out_channels,
                          padding=1,
                          kernel_size=kernel_size,
                          stride=(2 if average_pool else 1)),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(out_channels,
                          out_channels,
                          padding=1,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=False)
            )
        else:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels,
                          out_channels,
                          padding=1,
                          kernel_size=kernel_size,
                          stride=(2 if average_pool else 1)),
                nn.ReLU(inplace=False),
                nn.Conv1d(out_channels,
                          out_channels,
                          padding=1,
                          kernel_size=kernel_size),
                nn.ReLU(inplace=False)
            )

        self.downsample = None

        if average_pool:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=2),
                #nn.MaxPool1d(kernel_size=2, stride=2)
            )

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(identity)
            x = identity.clone() + self.backbone(x)
        else:
            x = identity.clone() + self.backbone(x)

        return x


class BottleneckBlock(nn.Module):
    """An ECG ResNet Bottleneck Block."""
    def __init__(self, in_channels, out_channels, kernel_size=3, average_pool=False, batch_norm=False):
        super(BottleneckBlock, self).__init__()
        middle_channels = int(out_channels / 4)
        if batch_norm:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels,
                          middle_channels,
                          padding=0,
                          kernel_size=1,
                          stride=(2 if average_pool else 1)),
                nn.BatchNorm1d(middle_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(middle_channels,
                          middle_channels,
                          padding=1,
                          kernel_size=kernel_size),
                nn.BatchNorm1d(middle_channels),
                nn.ReLU(inplace=False),
                nn.Conv1d(middle_channels,
                          out_channels,
                          padding=0,
                          kernel_size=1),
                nn.BatchNorm1d(out_channels),
                nn.ReLU(inplace=False)
            )

        else:
            self.backbone = nn.Sequential(
                nn.Conv1d(in_channels,
                          middle_channels,
                          padding=0,
                          kernel_size=1,
                          stride=(2 if average_pool else 1)),
                nn.ReLU(inplace=False),
                nn.Conv1d(middle_channels,
                          middle_channels,
                          padding=1,
                          kernel_size=kernel_size),
                nn.ReLU(inplace=False),
                nn.Conv1d(middle_channels,
                          out_channels,
                          padding=0,
                          kernel_size=1),
                nn.ReLU(inplace=False)
            )

        self.downsample = None

        if average_pool:
            self.downsample = nn.Sequential(
                nn.Conv1d(in_channels,
                      out_channels,
                      kernel_size=1,
                      stride=2),
                #nn.MaxPool1d(kernel_size=2, stride=2)
            )

    def forward(self, x):
        identity = x
        if self.downsample:
            identity = self.downsample(identity)
            x = identity.clone() + self.backbone(x)
        else:
            x = identity.clone() + self.backbone(x)

        return x
