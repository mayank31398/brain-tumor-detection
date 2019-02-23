import os

import numpy as np
from sklearn.model_selection import train_test_split
from tqdm import tqdm
from sklearn.utils import shuffle


def IsBlank(images):
    result = np.zeros(images.shape[0])
    for i in tqdm(range(images.shape[0])):
        image = images[i, ...]
        max_pixel = np.max(image)
        if(max_pixel != 0):
            result[i] = 1
    return result.astype(np.float32)


print("Loading data")

dataset_x = np.load(os.path.join("Data", "Train_x.npy"))
dataset_y = np.load(os.path.join("Data", "Train_y.npy"))

dataset_x = np.concatenate(
    [dataset_x, np.load(os.path.join("Data", "Validation_x.npy"))])
dataset_y = np.concatenate(
    [dataset_y, np.load(os.path.join("Data", "Validation_y.npy"))])
dataset_y = IsBlank(dataset_y)

dataset_x_ = np.load(os.path.join("Data", "Test_x.npy"))
dataset_y_ = np.load(os.path.join("Data", "Test_y.npy"))
dataset_x_, dataset_y_ = shuffle(dataset_x_, dataset_y_)

dataset_x = np.concatenate([dataset_x, dataset_x_[:, ...]])
dataset_y = np.concatenate([dataset_y, dataset_y_[:]])

# dataset_x_test = dataset_x_[400:, ...]
# dataset_y_test = dataset_y_[400:]

print("Data loaded")

dataset_x_train, dataset_x_val, dataset_y_train, dataset_y_val = train_test_split(
    dataset_x, dataset_y, test_size=0.1)

# dataset_x_val, dataset_x_test, dataset_y_val, dataset_y_test = train_test_split(
#     dataset_x_val_test, dataset_y_val_test, random_state=42, test_size=0.5)

np.save("Data/Train_x_mixed.npy", dataset_x_train)
np.save("Data/Train_y_mixed.npy", dataset_y_train)
np.save("Data/Validation_x_mixed.npy", dataset_x_val)
np.save("Data/Validation_y_mixed.npy", dataset_y_val)
# np.save("Data/Test_x_mixed.npy", dataset_x_test)
# np.save("Data/Test_y_mixed.npy", dataset_y_test)
