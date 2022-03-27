import tensorflow as tf
from tensorflow.keras import layers
import numpy as np


def meanBinaryAccuracy(y_true, y_pred_logits):
    y_pred_class = tf.math.round(y_pred_logits)
    acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(y_true, y_pred_class))
    return acc


class Generator:
    def __init__(self, config):
        self.input_size = config['input_size']
        self.adam_lr = config['adam_lr']
        self.adam_beta1 = config['adam_beta1']
        self.activation_alpha = config['activation_alpha']
        self.layers = config['layers']
        self.filter = config['filters']
        self.filter_size = config['filter_size']
        self.stride = config['stride']
        self.image_len = config['image_len']
        self.colour_channel = config['colour_channel']
        self.optimizer = self.__optimizer()

        self.model = self.__builtModelLayers()

    def __builtModelLayers(self):
        model = tf.keras.Sequential()
        model.add(
            layers.Dense(self.image_len * self.image_len * self.filter, use_bias=False,
                         input_shape=(self.input_size,))
        )
        model.add(layers.BatchNormalization())
        model.add(layers.LeakyReLU(self.activation_alpha))

        model.add(layers.Reshape((self.image_len, self.image_len, self.filter)))
        assert model.output_shape == (None, self.image_len, self.image_len, self.filter)  # Note: None is the batch size

        for l in range(1, self.layers):
            model.add(layers.Conv2DTranspose(self.filter / (l + 1), (self.filter_size, self.filter_size),
                                             strides=(self.stride, self.stride),
                                             padding='same',
                                             use_bias=False))
            self.image_len = self.stride * self.image_len
            assert model.output_shape == (None, self.image_len, self.image_len, self.filter / (l + 1))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=self.activation_alpha))

        model.add(layers.Conv2DTranspose(self.colour_channel, (self.filter_size, self.filter_size),
                                         strides=(self.stride, self.stride),
                                         padding='same',
                                         use_bias=False,
                                         activation='tanh'))
        self.image_len = self.stride * self.image_len
        assert model.output_shape == (None, self.image_len, self.image_len, self.colour_channel)

        return model

    def loss(self, fake_output):
        """
        The generator's loss quantifies how well it was able to trick the discriminator. Intuitively, if the generator is performing well, the discriminator will classify the fake images as real (or 1). Here, compare the discriminators decisions on the generated images to an array of 1s.
        :param fake_output: raw model output from discriminator using generated images
        :return: binary cross entropy loss
        """
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        fake_loss = cross_entropy(tf.ones_like(fake_output), fake_output)
        return fake_loss

    def accuracy(self, fake_output):
        fake_acc = meanBinaryAccuracy(tf.ones_like(fake_output), fake_output)
        return fake_acc

    def __optimizer(self):
        return tf.keras.optimizers.Adam(lr=self.adam_lr, beta_1=self.adam_beta1)


class Discriminator:
    def __init__(self, config):
        self.input_size = config['input_size']
        self.adam_lr = config['adam_lr']
        self.adam_beta1 = config['adam_beta1']
        self.activation_alpha = config['activation_alpha']
        self.layers = config['layers']
        self.filter = config['filters']
        self.filter_size = config['filter_size']
        self.stride = config['stride']
        self.dropout = config['dropout']
        self.image_len = config['image_len']
        self.colour_channel = config['colour_channel']
        self.optimizer = self.__optimizer()

        self.model = self.__buildModel()

    def __buildModel(self):
        model = tf.keras.Sequential()

        for l in range(1, self.layers):
            model.add(layers.Conv2D(self.filter * l,
                                    (self.filter_size, self.filter_size),
                                    strides=(self.stride, self.stride),
                                    padding='same',
                                    input_shape=[self.image_len, self.image_len, self.colour_channel]))
            self.image_len = self.image_len / self.stride
            assert model.output_shape == (None, self.image_len, self.image_len, self.filter * l)
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=self.activation_alpha))
            model.add(layers.Dropout(self.dropout))

        model.add(layers.Flatten())
        model.add(layers.Dense(1))

        return model

    def loss(self, real_output, fake_output):
        """
        This method quantifies how well the discriminator is able to distinguish real images from fakes. It compares the discriminator's predictions on real images to an array of 1s, and the discriminator's predictions on fake (generated) images to an array of 0s.
        :param real_output: raw model output from discriminator using real images
        :param fake_output: raw model output from discriminator using generated images
        :return: binary crossentropy
        """
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def accuracy(self, real_output, fake_output):
        fake_acc = meanBinaryAccuracy(tf.zeros_like(fake_output), fake_output)
        real_acc = meanBinaryAccuracy(tf.ones_like(real_output), real_output)
        total_acc = (real_acc + fake_acc) / 2
        return total_acc

    def __optimizer(self):
        return tf.keras.optimizers.Adam(lr=self.adam_lr, beta_1=self.adam_beta1)
