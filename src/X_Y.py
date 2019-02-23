import math
import os

# import matplotlib.pyplot as plt
import numpy as np
from comet_ml import Experiment
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm
from UtilsEncoder import GetData, GetDataloader
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)

BATCH_SIZE = 128
TRAIN_MINI_BATCH = 0
TEST_MINI_BATCH = 0
EPOCHS = 101
EXPERIMENT = Experiment(
    project_name="tumor",
    workspace="mayank31398",
    auto_output_logging=None,
    auto_metric_logging=False,
    auto_param_logging=False
)


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


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.l1 = nn.Sequential(nn.Linear(16, 8), nn.SELU())
        self.l2 = nn.Sequential(nn.Linear(8, 4), nn.SELU())
        self.l3 = nn.Sequential(nn.Linear(4, 2), nn.SELU())
        self.l4 = nn.Sequential(nn.Linear(2, 1), nn.SELU())

        self._initialize_submodules()

    def forward(self, x):
        x = self.l1(x)
        x = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)

        return x

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
            y = y.cuda().float().view

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

            print(total_loss)

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
            y = y.cuda().float()

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


def main():
    EXPERIMENT.add_tag("Embeddings")

    dataset_x, dataset_y = GetData(
        is_train=True, normalize=True)
    train_data = GetDataloader(dataset_x, dataset_y, BATCH_SIZE)

    dataset_x, dataset_y = GetData(
        is_train=False, normalize=True)
    test_data = GetDataloader(dataset_x, dataset_y, BATCH_SIZE)

    model = MyNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 0.5)
    model_saver = ModelSaver(os.path.join("Models", "Embeddings"))

    for epoch in range(EPOCHS):
        loss, accuracy, f1score, precision, recall = train(
            train_data, model, optimizer, EXPERIMENT)
        print("Training loss @ epoch", epoch, "=", loss)
        print("Training accuracy @ epoch", epoch, "=", accuracy)
        print("Training F1 score @ epoch", epoch, "=", f1score)
        print("Training precision @ epoch", epoch, "=", precision)
        print("Training recall @ epoch", epoch, "=", recall)
        with EXPERIMENT.train():
            EXPERIMENT.log_metric("Batch loss", loss, step=epoch)
            EXPERIMENT.log_metric("Batch accuracy", accuracy, step=epoch)
            EXPERIMENT.log_metric("Batch F1 score", f1score, step=epoch)
            EXPERIMENT.log_metric("Batch precision", precision, step=epoch)
            EXPERIMENT.log_metric("Batch recall", recall, step=epoch)

        loss, accuracy, f1score, precision, recall = test(
            test_data, model, EXPERIMENT)
        print("Validation loss @ epoch", epoch, "=", loss)
        print("Validation accuracy @ epoch", epoch, "=", accuracy)
        print("Validation F1 score @ epoch", epoch, "=", f1score)
        print("Validation precision @ epoch", epoch, "=", precision)
        print("Validation recall @ epoch", epoch, "=", recall)
        with EXPERIMENT.test():
            EXPERIMENT.log_metric("Batch loss", loss, step=epoch)
            EXPERIMENT.log_metric("Batch accuracy", accuracy, step=epoch)
            EXPERIMENT.log_metric("Batch F1 score", f1score, step=epoch)
            EXPERIMENT.log_metric("Batch precision", precision, step=epoch)
            EXPERIMENT.log_metric("Batch recall", recall, step=epoch)
        print("###############################################\n")
        model_saver.Save("epoch" + str(epoch) + ".pt",
                         model, optimizer, scheduler)

        if(epoch % 10 == 0):
            scheduler.step()


if(__name__ == "__main__"):
    main()