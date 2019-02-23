import math
import os

import matplotlib.pyplot as plt
from comet_ml import Experiment
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tqdm import tqdm


EPOCH = 0
TRAIN_MINI_BATCH = 0
TEST_MINI_BATCH = 0


class MyNet(nn.Module):
    def __init__(self):
        super().__init__()

        # 1, 240, 240
        self.l1 = nn.Conv2d(1, 2, 9)
        self.l2 = nn.MaxPool2d(2, return_indices=True)
        self.l3 = nn.SELU()
        # 2, 116, 116
        self.l4 = nn.Conv2d(2, 4, 7)
        self.l5 = nn.MaxPool2d(2, return_indices=True)
        self.l6 = nn.SELU()
        # 4, 55, 55
        self.l7 = nn.Conv2d(4, 8, 5, stride=2)
        self.l8 = nn.SELU()
        # 8, 26, 26
        self.l9 = nn.Conv2d(8, 16, 5)
        self.l10 = nn.MaxPool2d(2, return_indices=True)
        self.l11 = nn.SELU()
        # 16, 11, 11
        self.l12 = nn.Conv2d(16, 32, 3, stride=2)
        self.l13 = nn.SELU()
        # 32, 5, 5
        self.l14 = nn.ConvTranspose2d(32, 16, 3, stride=2)
        self.l15 = nn.SELU()
        # 16, 11, 11
        self.l16 = nn.MaxUnpool2d(2)
        self.l17 = nn.ConvTranspose2d(16, 8, 5)
        self.l18 = nn.SELU()
        # 8, 26, 26
        self.l19 = nn.ConvTranspose2d(8, 4, 5, stride=2)
        self.l20 = nn.SELU()
        # 4, 55, 55
        self.l21 = nn.MaxUnpool2d(2)
        self.l22 = nn.ConvTranspose2d(4, 2, 7)
        self.l23 = nn.SELU()
        # 2, 116, 116
        self.l24 = nn.MaxUnpool2d(2)
        self.l25 = nn.ConvTranspose2d(2, 1, 9)
        self.l26 = nn.SELU()
        # 1, 240, 240

        self._initialize_submodules()

    def forward(self, x):
        x = self.l1(x)
        x, ind1 = self.l2(x)
        x = self.l3(x)
        x = self.l4(x)
        x, ind2 = self.l5(x)
        x = self.l6(x)
        x = self.l7(x)
        x = self.l8(x)
        x = self.l9(x)
        x, ind3 = self.l10(x)
        x = self.l11(x)
        x = self.l12(x)
        embeddings = self.l13(x)
        x = self.l14(embeddings)
        x = self.l15(x)
        x = self.l16(x, ind3)
        x = self.l17(x)
        x = self.l18(x)
        x = self.l19(x)
        x = self.l20(x)
        x = self.l21(x, ind2)
        x = self.l22(x)
        x = self.l23(x)
        x = self.l24(x, ind1)
        x = self.l25(x)
        x = self.l26(x)

        return x, embeddings.view(embeddings.shape[0], -1)

    def _initialize_submodules(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, math.sqrt(1. / n))
            elif isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.in_channels
                m.weight.data.normal_(0, math.sqrt(1. / n))


def MyLoss(y, prediction):
    y = y.view(y.shape[0], -1)
    prediction = prediction.view(prediction.shape[0], -1)
    loss = (y * prediction).sum(dim=1) / (y.norm(dim=1) * prediction.norm(dim=1))
    loss = th.acos(loss) * 180 / np.pi
    return loss.mean()


def train(train_data, model, optimizer, experiment: Experiment):
    global TRAIN_MINI_BATCH
    global EPOCH

    model.train()
    model.cuda()

    batches = len(train_data)
    total_loss = 0
    loss_func = nn.L1Loss()
    with experiment.train():
        for x, in tqdm(train_data):
            x = x.cuda()

            optimizer.zero_grad()
            prediction, _ = model(x)
            loss = loss_func(prediction, x)
            # loss = MyLoss(x, prediction)

            experiment.log_metric(
                "mini-batch loss", loss.item(), step=TRAIN_MINI_BATCH)
            TRAIN_MINI_BATCH += 1
            loss.backward()
            optimizer.step()
            total_loss += loss.item()

        average_loss = total_loss / batches
        experiment.log_metric("batch loss", average_loss, step=EPOCH)

    return average_loss


def test(test_data, model, experiment: Experiment):
    global TEST_MINI_BATCH
    global EPOCH

    with experiment.test():
        with th.no_grad():
            model.eval()
            model.cuda()

            batches = len(test_data)
            total_loss = 0
            loss_func = nn.L1Loss()
            for x, in tqdm(test_data):
                x = x.cuda()

                prediction, _ = model(x)
                loss = loss_func(prediction, x)
                # loss = MyLoss(x, prediction)
                experiment.log_metric(
                    "mini-batch loss", loss.item(), step=TEST_MINI_BATCH)
                TEST_MINI_BATCH += 1
                total_loss += loss.item()

            average_loss = total_loss / batches
            experiment.log_metric("batch loss", average_loss, step=EPOCH)
            EPOCH += 1

    return average_loss


def Visualize(test_data, model):
    with th.no_grad():
        model.eval()
        model.cuda()

        dataset_embeddings = []
        dataset_class = []
        loss_func = nn.MSELoss()
        for x, y in test_data:
            x = x.cuda()
            y = y.cuda()

            prediction, embed = model(x)
            loss = loss_func(prediction, x)
            print("Loss =", loss.item())

            # prediction = prediction.reshape(prediction.shape[0], prediction.shape[2], prediction.shape[3])

            dataset_embeddings.append(embed.detach().cpu().numpy())
            dataset_class.append(y.detach().cpu().numpy())

            for dimension in range(x.shape[0]):
                plt.subplot(121)
                plt.imshow(np.squeeze(x.cpu().numpy()[dimension, ...], axis=0))

                plt.subplot(122)
                plt.imshow(np.squeeze(prediction.cpu().numpy()
                                      [dimension, ...], axis=0))

                plt.show()
        
        np.save("Data/Test_embeddings.npy", np.concatenate(dataset_embeddings, axis=0))
        np.save("Data/Test_labels.npy", np.concatenate(dataset_class, axis=0))


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
