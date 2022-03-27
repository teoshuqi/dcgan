import tensorflow as tf
import json
import utils
import glob
from models import Generator, Discriminator
from data import MINST

with open('config.json', 'r') as f_:
    config = json.load(f_)

num_examples_to_generate = config['num_examples_to_generate']
noise_dim = config['noise_dim']
DATANAME = config['dataname']

generator = Generator(config['generator'])
discriminator = Discriminator(config['discriminator'])
if len(glob.glob(f"checkpoints\\*_{DATANAME}.h5")) > 1:
    generator.model.load_weights(f'./checkpoints/generator_{DATANAME}.h5')
    discriminator.model.load_weights(f'./checkpoints/discriminator_{DATANAME}.h5')

test_input = tf.random.normal([num_examples_to_generate, noise_dim])

print('Initial')
utils.generate_and_save_images(generator.model, test_input, f'{DATANAME}_0000.png')

print('Train')
train_dataset = MINST()
history = utils.train(train_dataset, test_input, generator, discriminator)

print('Final')
utils.generate_and_save_images(generator.model, test_input, f'{DATANAME}_final.png')

utils.saveImagesAsGIF(f'./stats/dcgan_{DATANAME}.gif')
utils.saveAndPlotModelPerformance(history, f'./stats/dcgan_{DATANAME}')


generator_pred = Generator(config['generator'])
discriminator_pred = Discriminator(config['discriminator'])
generator_pred.model.load_weights(f'./checkpoints/generator_{DATANAME}.h5')
discriminator_pred.model.load_weights(f'./checkpoints/discriminator_{DATANAME}.h5')
utils.generate_and_save_images(generator_pred.model, test_input, f'{DATANAME}_pred.png')
