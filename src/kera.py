import keras
import numpy as np
from keras import backend as K
from keras.datasets import mnist
from keras.layers import (Activation, Conv2D, Dense, Dropout, Flatten,
                          MaxPooling2D)
from keras.models import Sequential
from sklearn.utils import shuffle

batch_size = 128
num_classes = 10
epochs = 100

x_train = np.load("Data/Train_x.npy")
y_train = np.load("Data/Train_y.npy")
x_test = np.load("Data/Validation_x.npy")
y_test = np.load("Data/Validation_y.npy")

x_train, y_train = shuffle(x_train, y_train)
x_test, y_test = shuffle(x_test, y_test)

x_train = x_train.reshape(x_train.shape[0], 240, 240, 1)
x_test = x_test.reshape(x_test.shape[0], 240, 240, 1)
input_shape = (240, 240, 1)

x_train = x_train.astype('float32')
x_test = x_test.astype('float32')

print('x_train shape:', x_train.shape)
print(x_train.shape[0], 'train samples')
print(x_test.shape[0], 'test samples')

# convert class vectors to binary class matrices
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

model = Sequential([
    # 240, 240, 1
    Conv2D(2, 9),
    MaxPooling2D(pool_size=(2, 2)),
    Activation("relu", alpha=0.05),
    # 116, 116, 2
    Conv2D(4, 7),
    MaxPooling2D(pool_size=(2, 2)),
    Activation("relu", alpha=0.05),
    # 55, 55, 4
    Conv2D(8, 5, strides=(2, 2)),
    Activation("relu", alpha=0.05),
    # 26, 26, 8
    Conv2D(16, 5),
    MaxPooling2D(pool_size=(2, 2)),
    Activation("relu", alpha=0.05),
    # 11, 11, 16
    Conv2D(32, 3, strides=(2, 2)),
    Activation("relu", alpha=0.05),
    # 5, 5, 32
    Conv2D(64, 3, strides=(2, 2)),
    MaxPooling2D(pool_size=(2, 2)),
    Activation("relu", alpha=0.05),
    # 1, 1, 64
    
    Flatten(),

    Dense(64),
    Activation("relu", alpha=0.05),
    Dense(32),
    Activation("relu", alpha=0.05),
    Dense(16),
    Activation("relu", alpha=0.05),
    Dense(8),
    Activation("relu", alpha=0.05),
    Dense(4),
    Activation("relu", alpha=0.05),
    Dense(2),
    Activation("relu", alpha=0.05),
    Dense(1),
    Activation("relu", alpha=0.05)
])

model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])

model.fit(x_train, y_train,
          batch_size=batch_size,
          epochs=epochs,
          verbose=1,
          validation_data=(x_test, y_test))
score = model.evaluate(x_test, y_test, verbose=0)
print('Test loss:', score[0])
print('Test accuracy:', score[1])
