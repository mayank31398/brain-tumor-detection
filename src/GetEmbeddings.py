import math
import os

import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm
from UtilsEncoder import GetData, GetDataloader
import torch.optim as optim
import torch as th

from AutoEncoder import ModelSaver, MyNet

BATCH_SIZE = 128


def Generate(test_data, model):
    with th.no_grad():
        model.eval()
        model.cuda()

        dataset_s = []
        dataset_embeddings = []
        for x, y in tqdm(test_data):
            x = x.cuda()
            y = y.cuda()

            s = (y.sum(dim=1).sum(dim=1).sum(dim=1) != 0).detach().cpu().numpy()
            _, embeddings = model(x)
            embeddings = embeddings.detach().cpu().numpy()

            dataset_s.append(s)
            dataset_embeddings.append(embeddings)

    return np.concatenate(dataset_embeddings), np.concatenate(dataset_s)


def main():
    dataset_x, dataset_y = GetData(
        is_train=True, discard_blanks=True, normalize=True)
    train_data = GetDataloader(dataset_x, dataset_y, BATCH_SIZE)

    dataset_x, dataset_y = GetData(
        is_train=False, discard_blanks=True, normalize=True)
    test_data = GetDataloader(dataset_x, dataset_y, BATCH_SIZE)

    model = MyNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-2)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 0.5)
    model_saver = ModelSaver(os.path.join("Models", "AutoEncoder"))
    model_saver.Load("epoch30.pt", model, optimizer, scheduler)

    embeddings, s = Generate(train_data, model)
    np.save(os.path.join("Data", "Train_embeddings.npy"), embeddings)
    np.save(os.path.join("Data", "Train_s.npy"), s)

    embeddings, s = Generate(test_data, model)
    np.save(os.path.join("Data", "Validation_embeddings.npy"), embeddings)
    np.save(os.path.join("Data", "Validation_s.npy"), s)


if(__name__ == "__main__"):
    main()