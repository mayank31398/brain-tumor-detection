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
            nn.Conv2d(1, 2, 7), nn.MaxPool2d(2), nn.SELU(),  # 2, 117, 117
            nn.Conv2d(2, 4, 5, stride=2), nn.SELU(),  # 4, 57, 57
            nn.Conv2d(4, 8, 5, stride=2), nn.SELU(),  # 8, 27, 27
            nn.Conv2d(8, 16, 5, stride=2), nn.SELU(),  # 16, 12, 12
            nn.Conv2d(16, 32, 5), nn.MaxPool2d(2), nn.SELU(),  # 32, 4, 4
            nn.Conv2d(32, 64, 3), nn.MaxPool2d(2), nn.SELU(),  # 64, 1, 1
        )

        self.l2 = nn.Sequential(
            nn.Conv1d(1, 2, 5), nn.MaxPool1d(2), nn.SELU(), # 2, 30
            nn.Conv1d(2, 4, 3), nn.MaxPool1d(2), nn.SELU(), # 4, 14
            nn.Conv1d(4, 4, 3), nn.MaxPool1d(2), nn.SELU(), # 4, 6
            nn.Conv1d(4, 2, 3), nn.MaxPool1d(2), nn.SELU(), # 2, 2
            nn.Conv1d(2, 1, 2), nn.Sigmoid(), # 1, 1
        )

        self._initialize_submodules()

    def forward(self, x):
        cnn_output = self.l1(x)
        cnn_output = cnn_output.view(cnn_output.shape[0], 1, -1)
        prediction = self.l2(cnn_output)
        return prediction.view(-1)

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
