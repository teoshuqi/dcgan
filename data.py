import tensorflow as tf
import json
from tensorflow.keras import layers
import numpy as np
import pathlib

with open('config_faces.json', 'r') as f_:
    config = json.load(f_)

BUFFER_SIZE = config['buffer_size']
BATCH_SIZE = config['batch_size']
AUTOTUNE = tf.data.AUTOTUNE


def MINST():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images - 127.5) / 127.5  # Normalize the images to [-1, 1]
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset


def parse_image(filename):
    image = tf.io.read_file(filename)
    image = tf.io.decode_jpeg(image)
    image = tf.image.convert_image_dtype(image, tf.float32)
    image = tf.image.resize(image, [128, 128])
    image = (image - 127.5) / 127.5
    return image


def load_custom_dataset(data_dir):
    train_ds = tf.keras.utils.image_dataset_from_directory(
        data_dir,
        labels=None,
        seed=123,
        image_size=(64, 64),
        batch_size=BATCH_SIZE)

#    train_ds = train_ds.cache().shuffle(buffer_size=BUFFER_SIZE)
    train_ds = train_ds.cache().prefetch(buffer_size=AUTOTUNE)
    normalization_layer = layers.Rescaling(scale=1./127.5, offset=-1)
    train_ds = train_ds.map(lambda x: (normalization_layer(x)))
    return train_ds
