import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import utils
import matplotlib.pyplot as plt
import time
import json

with open('config.json', 'r') as f:
    config = json.load(f)

dataname = config['dataname']
model_dir = config['model_dir']
predict_dir = config['predict_dir']
stats_dir = config['stats_dir']

def meanBinaryAccuracy(y_true, y_pred_logits):
    y_pred_class = tf.math.round(y_pred_logits)
    acc = tf.reduce_mean(tf.keras.metrics.binary_accuracy(y_true, y_pred_class))
    return acc


class DCGAN:
    def __init__(self, config):
        self.generator = Generator(config['generator'])
        self.discriminator = Discriminator(config['discriminator'])
        self.history = {'Gen Acc': [], 'Disc Acc': [], 'Gen Loss': [], 'Disc Loss': []}
        self.epochs = config["epochs"]

    def load_weights(self):
        self.generator.load_weights()
        self.discriminator.load_weights()

    def save_weights(self):
        self.generator.save_weights()
        self.discriminator.save_weights()

    def __logEpochPerformance(self, epoch, start):
        gen_loss = round(np.mean(self.history['Gen Loss']), 2)
        disc_loss = round(np.mean(self.history['Disc Loss']), 2)
        gen_acc = round(np.mean(self.history['Gen Acc']) * 100, 2)
        disc_acc = round(np.mean(self.history['Disc Acc']) * 100, 2)
        seconds = round(time.time() - start, 2)
        print('Time for epoch {} is {} sec'.format(epoch + 1, seconds))
        print(f"Loss | Gen: {gen_loss}, Disc: {disc_loss}")
        print(f"Acc | Gen: {gen_acc}, Disc: {disc_acc}")

    def __updatePerformanceStats(self, batch_acc, batch_loss):
        self.history['Gen Acc'].append(batch_acc[0])
        self.history['Disc Acc'].append(batch_acc[1])
        self.history['Gen Loss'].append(batch_loss[0])
        self.history['Disc Loss'].append(batch_loss[1])

    def train(self, dataset, test_input):

        for epoch in range(self.epochs):
            start = time.time()

            for image_batch in dataset:
                gen_acc, disc_acc, gen_loss, disc_loss = utils.train_step(image_batch, self.generator,
                                                                          self.discriminator)
                batch_acc, batch_loss = utils.summmariseBatchPerformance(gen_acc, disc_acc, gen_loss, disc_loss)
                self.__updatePerformanceStats(batch_acc, batch_loss)

                # Save the model every 15 epochs
                if (epoch + 1) % 5 == 0:
                    self.generator.save_weights()
                    self.discriminator.save_weights()
            filename = f'epoch_{epoch + 1:04d}.png'
            self.predict(test_input, filename)
            self.__logEpochPerformance(epoch, start)

        return history

    def predict(self, test_input, filename):
        # Notice `training` is set to False.
        # This is so all layers run in inference mode (batchnorm).
        predictions = self.generator.model(test_input, training=False)

        fig = plt.figure(figsize=(4, 4))

        for i in range(predictions.shape[0]):
            plt.subplot(4, 4, i + 1)
            plt.imshow((predictions[i, :, :, 0] - 127.5) / 127.5, cmap='gray')
            plt.axis('off')

        plt.savefig(f'./{predict_dir}/{dataname}/{filename}')
        return fig

    def saveModelPerformance(self, filename=f'./{stats_dir}/dcgan'):
        with open(f'{filename}.json', 'w') as f_:
            json.dump(self.history, f_)
        fig, (ax1, ax2) = plt.subplots(2, figsize=(10, 5), sharex=True)
        fig.suptitle("Generator and Discriminator Loss During Training")
        ax1.plot(self.history['Gen Loss'], label='G')
        ax1.plot(self.history['Disc Loss'], label='D')
        ax1.set_ylabel("Loss")
        ax2.plot(self.history['Gen Acc'], label='G')
        ax2.plot(self.history['Disc Acc'], label='D')
        ax2.set_ylabel("Accuracy")
        ax2.set_xlabel("Iterations")
        fig.savefig(f'{filename}.png', dpi=fig.dpi)


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
        self.initial_std = config['initial_std']
        self.initial_mean = config['initial_mean']
        self.initializer = tf.keras.initializers.RandomNormal(mean=self.initial_mean, stddev=self.initial_std)
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
                                             kernel_initializer=self.initializer,
                                             use_bias=False))
            self.image_len = self.stride * self.image_len
            assert model.output_shape == (None, self.image_len, self.image_len, self.filter / (l + 1))
            model.add(layers.BatchNormalization())
            model.add(layers.LeakyReLU(alpha=self.activation_alpha))

        model.add(layers.Conv2DTranspose(self.colour_channel, (self.filter_size, self.filter_size),
                                         strides=(self.stride, self.stride),
                                         padding='same',
                                         kernel_initializer=self.initializer,
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

    def load_weights(self):
        self.model.load_weights(f'./{model_dir}/{dataname}/generator.h5')

    def save_weights(self):
        self.model.save_weights(f'./{model_dir}/{dataname}/generator.h5', save_format="h5")
        return

    def __optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.adam_lr, beta_1=self.adam_beta1)


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
        self.initial_std = config['initial_std']
        self.initial_mean = config['initial_mean']
        self.initializer = tf.keras.initializers.RandomNormal(mean=self.initial_mean, stddev=self.initial_std)
        self.optimizer = self.__optimizer()

        self.model = self.__buildModel()

    def __buildModel(self):
        model = tf.keras.Sequential()

        for l in range(1, self.layers):
            self.filter *= 2
            model.add(layers.Conv2D(self.filter,
                                    (self.filter_size, self.filter_size),
                                    strides=(self.stride, self.stride),
                                    padding='same',
                                    kernel_initializer=self.initializer,
                                    input_shape=[self.image_len, self.image_len, self.colour_channel]))
            self.image_len = self.image_len / self.stride
            assert model.output_shape == (None, self.image_len, self.image_len, self.filter)
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
        cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True, label_smoothing=0.1)
        real_loss = cross_entropy(tf.ones_like(real_output), real_output)
        fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
        total_loss = real_loss + fake_loss
        return total_loss

    def accuracy(self, real_output, fake_output):
        fake_acc = meanBinaryAccuracy(tf.zeros_like(fake_output), fake_output)
        real_acc = meanBinaryAccuracy(tf.ones_like(real_output), real_output)
        total_acc = (real_acc + fake_acc) / 2
        return total_acc

    def load_weights(self):
        self.model.load_weights(f'./{model_dir}/{dataname}/discriminator.h5')

    def save_weights(self):
        self.model.save_weights(f'./{model_dir}/{dataname}/discriminator.h5', save_format="h5")

    def __optimizer(self):
        return tf.keras.optimizers.Adam(learning_rate=self.adam_lr, beta_1=self.adam_beta1)
