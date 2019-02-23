import os

import numpy as np
import SimpleITK as sitk
from sklearn.model_selection import train_test_split
from tqdm import tqdm
import matplotlib.pyplot as plt

DATA_PATH = os.path.join("Data", "BRATS2015_Training")
TRAIN_TEST_SPLIT = 0.9


def Make(dataset_x, dataset_y):
    x_train, x_test, y_train, y_test = train_test_split(
        dataset_x, dataset_y, train_size=TRAIN_TEST_SPLIT)
    print(x_train.shape)
    print(y_train.shape)
    print(x_test.shape)
    print(y_test.shape)
    np.save("Data/Train_x.npy", x_train)
    np.save("Data/Train_y.npy", y_train)
    np.save("Data/Validation_x.npy", x_test)
    np.save("Data/Validation_y.npy", y_test)


def main():
    image_reader = sitk.ImageFileReader()
    image_reader.SetImageIO("MetaImageIO")
    pathologies = os.listdir(DATA_PATH)

    x = []
    y = []
    for pathology in pathologies:
        path = os.path.join(DATA_PATH, pathology)
        folders = os.listdir(path)

        print("Loading", pathology)
        for folder in tqdm(folders):
            path1 = os.path.join(path, folder)
            files = os.listdir(path1)

            image_reader.SetFileName(os.path.join(
                path1, files[1], files[1]) + ".mha")
            t1 = image_reader.Execute()
            t1 = sitk.GetArrayFromImage(t1)

            image_reader.SetFileName(os.path.join(
                path1, files[4], files[4]) + ".mha")
            out = image_reader.Execute()
            out = sitk.GetArrayFromImage(out)

            for dimension in range(60, 120):
                x_example = t1[dimension, ...]
                y_example = out[dimension, ...]

                x.append(x_example)
                y.append(y_example)
    
    x = np.array(x)
    y = np.array(y)
    # shuffler = np.arange(x.shape[0])
    # np.random.shuffle(shuffler)
    # x = x[shuffler, ...]
    # y = y[shuffler, ...]

    Make(x, y)


def Imager():
    x=np.load("Data/Train_x.npy")
    for i in range(x.shape[0]):
        plt.imsave("Images/" + str(i) + ".png", x[i, ...])


if(__name__ == "__main__"):
    Imager()
