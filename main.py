import tensorflow as tf
import json
import utils
from models import Generator, Discriminator
from data import MINST
import matplotlib.pyplot as plt

with open('config.json', 'r') as f_:
    config = json.load(f_)

num_examples_to_generate = config['num_examples_to_generate']
noise_dim = config['noise_dim']

generator = Generator(config['generator'])
discriminator = Discriminator(config['discriminator'])

test_input = tf.random.normal([num_examples_to_generate, noise_dim])

print('Initial')
utils.generate_and_save_images(generator.model, test_input, 'image_0000.png')


print('Train')
train_dataset = MINST()
utils.train(train_dataset, test_input, generator, discriminator)

print('Final')
utils.generate_and_save_images(generator.model, test_input, 'image_final.png')


utils.saveImagesAsGIF('dcgan_minst.gif')