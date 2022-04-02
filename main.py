import os
# os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import tensorflow as tf
import json
import utils
import glob
from models import DCGAN
from data import MINST, load_custom_dataset

with open('config_faces.json', 'r') as f_:
    config = json.load(f_)

num_examples_to_generate = config['num_examples_to_generate']
noise_dim = config['noise_dim']
DATANAME = config['dataname']
model_dir = config['model_dir']
predict_dir = config['predict_dir']
stats_dir = config['stats_dir']

if __name__ == "__main__":
    dcgan = DCGAN(config)
    # if len(glob.glob(f"{model_dir}\\{DATANAME}\\*_{DATANAME}.h5")) > 1:
    #     dcgan.load_weights()

    test = tf.random.normal([num_examples_to_generate, noise_dim])
    train_dataset = load_custom_dataset("../subset_faces")

    dcgan.predict(test, f'epoch_0000.png')

    print(f'Start training DCGAN for {DATANAME} dataset')
    history = dcgan.train(train_dataset, test)

    utils.saveImagesAsGIF()
    dcgan.saveModelPerformance()