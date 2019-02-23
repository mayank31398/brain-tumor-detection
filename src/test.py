import os

import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
from sklearn.metrics import (accuracy_score, f1_score, precision_score,
                             recall_score)
from tqdm import tqdm

from ConvNet import ModelSaver, MyNet
from utils import CodeLogger, GetData, GetDataloader, MinMaxNormalize, IsBlank

BATCH_SIZE = 128
EPOCHS = 101


def test(test_data, model):
    model.eval()
    model.cuda()

    batches = len(test_data)
    total_loss = 0
    predictions = []
    ground_truth = []

    for x, y in tqdm(test_data):
        x = x.cuda()
        y = y.cuda()

        prediction = model(x)
        loss = F.binary_cross_entropy(prediction, y)
        total_loss += loss.item()

        prediction = prediction >= 0.5
        predictions.append(prediction.detach().cpu().numpy())
        ground_truth.append(y.detach().cpu().numpy())

    average_loss = total_loss / batches
    predictions = np.concatenate(predictions)
    ground_truth = np.concatenate(ground_truth)

    accuracy = accuracy_score(ground_truth, predictions)
    f1score = f1_score(ground_truth, predictions)
    precision = precision_score(ground_truth, predictions)
    recall = recall_score(ground_truth, predictions)

    return average_loss, accuracy, f1score, precision, recall


def main():
    dataset_x = np.load(os.path.join("Data", "Test_embeddings.npy"))
    # dataset_x = MinMaxNormalize(dataset_x).astype(np.float32)
    dataset_y = np.load(os.path.join("Data", "Test_labels.npy")).astype(np.float32)
    # dataset_y = IsBlank(dataset_y)
    test_data = GetDataloader(dataset_x, dataset_y, BATCH_SIZE)

    model = MyNet()
    optimizer = optim.Adam(model.parameters(), lr=1e-3)
    scheduler = optim.lr_scheduler.StepLR(optimizer, 0.5)
    model_saver = ModelSaver(os.path.join("Models", "ConvNet1"))

    model, _, _ = model_saver.Load("epoch40.pt", model, optimizer, scheduler)

    loss, accuracy, f1score, precision, recall = test(test_data, model)
    print("Loss =", loss)
    print("Accuracy =", accuracy)
    print("F1 score =", f1score)
    print("Precision =", precision)
    print("Recall =", recall)


if(__name__ == "__main__"):
    main()
