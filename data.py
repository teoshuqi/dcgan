import tensorflow as tf
import json

with open('config.json', 'r') as f_:
    config = json.load(f_)

BUFFER_SIZE = config['buffer_size']
BATCH_SIZE = config['batch_size']


def MINST():
    (train_images, train_labels), (_, _) = tf.keras.datasets.mnist.load_data()
    train_images = train_images.reshape(train_images.shape[0], 28, 28, 1).astype('float32')
    train_images = (train_images- 127.5) / 127.5  # Normalize the images to [-1, 1]
    # Batch and shuffle the data
    train_dataset = tf.data.Dataset.from_tensor_slices(train_images).shuffle(BUFFER_SIZE).batch(BATCH_SIZE)
    return train_dataset
