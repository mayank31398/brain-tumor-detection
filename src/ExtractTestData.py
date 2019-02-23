import cv2.cv2 as cv2
import os
import numpy as np
from sklearn.utils import shuffle
import matplotlib.pyplot as plt

NEW_SIZE = 240
RANDOM_SEED = 42


def PadImage(im):
    desired_size = 240
    old_size = im.shape[:2]

    ratio = float(desired_size) / max(old_size)
    new_size = tuple([int(x * ratio) for x in old_size])

    im = cv2.resize(im, (new_size[1], new_size[0]))

    delta_w = desired_size - new_size[1]
    delta_h = desired_size - new_size[0]
    top, bottom = delta_h // 2, delta_h - (delta_h // 2)
    left, right = delta_w // 2, delta_w - (delta_w // 2)

    color = [0, 0, 0]
    new_im = cv2.copyMakeBorder(im, top, bottom, left, right, cv2.BORDER_CONSTANT,
                                value=color)

    return new_im


def GetBrain(image):
    ret, thresh = cv2.threshold(image, 70, 255, cv2.THRESH_BINARY)

    # plt.subplot(121)
    # plt.imshow(image)
    # plt.subplot(122)
    # plt.imshow(thresh)
    # plt.show()
    
    ret, markers = cv2.connectedComponents(thresh)

    try:
        marker_area = [np.sum(markers == m)
                    for m in range(np.max(markers)) if m != 0]
        largest_component = np.argmax(marker_area) + 1
        brain_mask = markers == largest_component

        plt.imshow(brain_mask)
        plt.show()

        brain_out = image.copy()
        brain_out[brain_mask == False] = 0

        return brain_out
    except:
        return image


def main():
    dataset_x = []
    dataset_y = []

    files = os.listdir(os.path.join("Data", "Test_original", "abnormalsJPG"))
    for file in files:
        dataset_y.append(1)
        file = cv2.imread(os.path.join("Data", "Test_original", "abnormalsJPG", file))
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)

        file = PadImage(file)
        dataset_x.append(file)

    files = os.listdir(os.path.join("Data", "Test_original", "normalsJPG"))
    for file in files:
        dataset_y.append(0)
        file = cv2.imread(os.path.join("Data", "Test_original", "normalsJPG", file))
        file = cv2.cvtColor(file, cv2.COLOR_BGR2GRAY)

        file = PadImage(file)
        dataset_x.append(file)

    dataset_x, dataset_y = shuffle(
        dataset_x, dataset_y, random_state=RANDOM_SEED)

    dataset_x = np.array(dataset_x)
    dataset_y = np.array(dataset_y)

    # dataset_x = dataset_x.sum(axis = 0)
    # plt.hist(dataset_x.ravel())
    # plt.show()

    np.save(os.path.join("Data", "Test_x.npy"), dataset_x)
    np.save(os.path.join("Data", "Test_y.npy"), dataset_y)


def MyFunc():
    x=np.load("Data/Test_x.npy")
    for i in range(x.shape[0]):
        plt.imshow(x[i, ...])
        plt.show()


if(__name__ == "__main__"):
    main()
    # MyFunc()
