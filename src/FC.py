import math
import os

import numpy as np
from comet_ml import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm

TRAIN_MINI_BATCH = 0
TEST_MINI_BATCH = 0


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1, 240, 240
        self.l1 = nn.Sequential(
            nn.Linear(64, 32), nn.SELU(),
            nn.Linear(32, 16), nn.SELU(),
            nn.Linear(16, 8), nn.SELU(),
            nn.Linear(8, 4), nn.SELU(),
            nn.Linear(4, 2), nn.SELU(),
            nn.Linear(2, 1), nn.Sigmoid(),
        )

        self._initialize_submodules()

    def forward(self, x):
        x = self.l1(x)
        return x.view(-1)

    def _initialize_submodules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))


def train(train_data, model, optimizer, experiment: Experiment):
    global TRAIN_MINI_BATCH

    model.train()
    model.cuda()

    batches = len(train_data)
    total_loss = 0
    predictions = []
    ground_truth = []

    with experiment.train():
        for x, y in tqdm(train_data):
            x = x.cuda()
            y = y.cuda()

            optimizer.zero_grad()
            prediction = model(x)
            loss = F.binary_cross_entropy(prediction, y)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

            prediction = prediction >= 0.5
            predictions.append(prediction.detach().cpu().numpy())
            ground_truth.append(y.detach().cpu().numpy())

            experiment.log_metric(
                "Mini batch loss", loss.item(), step=TRAIN_MINI_BATCH)
            TRAIN_MINI_BATCH += 1

    average_loss = total_loss / batches
    predictions = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)

    accuracy = accuracy_score(ground_truth, predictions)
    f1score = f1_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)

    return average_loss, accuracy, f1score, precision, recall


def test(test_data, model, experiment: Experiment):
    global TEST_MINI_BATCH

    model.eval()
    model.cuda()

    batches = len(test_data)
    total_loss = 0
    predictions = []
    ground_truth = []

    with experiment.test():
        for x, y in tqdm(test_data):
            x = x.cuda()
            y = y.cuda()

            prediction = model(x)
            loss = F.binary_cross_entropy(prediction, y)
            total_loss += loss.item()

            prediction = prediction >= 0.5
            predictions.append(prediction.detach().cpu().numpy())
            ground_truth.append(y.detach().cpu().numpy())

            experiment.log_metric(
                "Mini batch loss", loss.item(), step=TEST_MINI_BATCH)
            TEST_MINI_BATCH += 1

    average_loss = total_loss / batches
    predictions = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)

    accuracy = accuracy_score(ground_truth, predictions)
    f1score = f1_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)

    return average_loss, accuracy, f1score, precision, recall


class ModelSaver:
    def __init__(self, directory):
        os.makedirs(directory, exist_ok=True)
        self.directory = directory

    def Save(self, filename, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler):
        model_dict = model.state_dict()
        optimizer_dict = optimizer.state_dict()
        scheduler_dict = scheduler.state_dict()

        checkpoint = {
            "model_dict": model_dict,
            "optimizer_dict": optimizer_dict,
            "scheduler_dict": scheduler_dict
        }

        th.save(checkpoint, os.path.join(self.directory, filename))

    def Load(self, filename, model: nn.Module, optimizer: optim.Optimizer, scheduler: optim.lr_scheduler):
        checkpoint = th.load(os.path.join(self.directory, filename))
        model.load_state_dict(checkpoint["model_dict"])
        optimizer.load_state_dict(checkpoint["optimizer_dict"])
        scheduler.load_state_dict(checkpoint["scheduler_dict"])

        return model, optimizer, scheduler
