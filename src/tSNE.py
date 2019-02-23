import os
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data
from tensorflow.contrib.tensorboard.plugins import projector
import numpy as np

PATH = os.getcwd()
LOG_DIR = os.path.join(PATH, "tSNE")
metadata = os.path.join(LOG_DIR, 'metadata.tsv')

x = np.load("Data/Validation_x.npy")
x = x.reshape(x.shape[0], -1)
y = np.load("Data/Validation_y.npy").sum(axis=1).sum(axis=1) != 0
os.makedirs("tSNE")

images = tf.Variable(x, name='images')
with open(metadata, 'w') as metadata_file:
    for row in range(x.shape[0]):
        c = y[row]
        metadata_file.write('{}\n'.format(c))

with tf.Session() as sess:
    saver = tf.train.Saver([images])

    print(images.shape)

    sess.run(images.initializer)
    saver.save(sess, os.path.join(LOG_DIR, 'images.ckpt'))

    config = projector.ProjectorConfig()
    embedding = config.embeddings.add()
    embedding.tensor_name = images.name
    embedding.metadata_path = metadata
    projector.visualize_embeddings(tf.summary.FileWriter(LOG_DIR), config)