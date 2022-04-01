import tensorflow as tf
import matplotlib.pyplot as plt
import numpy as np
import time
import json
import imageio
import glob

with open('config_faces.json', 'r') as f:
    config = json.load(f)

BATCH_SIZE = config['batch_size']
noise_dim = config['noise_dim']
EPOCHS = config['epochs']
DATANAME = config['dataname']
stats_dir = config['stats_dir']

# Notice the use of `tf.function`
# This annotation causes the function to be "compiled".
@tf.function
def train_step(images, generator, discriminator):
    noise = tf.random.normal([BATCH_SIZE, noise_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
        generated_images = generator.model(noise, training=True)

        real_output = discriminator.model(images, training=True)
        fake_output = discriminator.model(generated_images, training=True)

        gen_loss = generator.loss(fake_output)
        disc_loss = discriminator.loss(real_output, fake_output)
        gen_acc = generator.accuracy(fake_output)
        disc_acc = discriminator.accuracy(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.model.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.model.trainable_variables)

    generator.optimizer.apply_gradients(zip(gradients_of_generator, generator.model.trainable_variables))
    discriminator.optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.model.trainable_variables))
    return gen_acc, disc_acc, gen_loss, disc_loss


def createModelCheckpoint(generator, discriminator):
    checkpoint = tf.train.Checkpoint(generator_optimizer=generator.optimizer,
                                     discriminator_optimizer=discriminator.optimizer,
                                     generator=generator.model,
                                     discriminator=discriminator.model)
    return checkpoint



def extractFloatFromEagerTensor(eagerTensor, dp):
    decimals = float(tf.cast(eagerTensor, tf.float32))
    rounded = round(decimals, dp)
    return rounded


def summmariseBatchPerformance(gen_acc, disc_acc, gen_loss, disc_loss):
    decimal_gen_acc = extractFloatFromEagerTensor(gen_acc, 2)
    decimal_disc_acc = extractFloatFromEagerTensor(disc_acc, 2)
    decimal_gen_loss = extractFloatFromEagerTensor(gen_loss, 4)
    decimal_disc_loss = extractFloatFromEagerTensor(disc_loss, 4)
    return (decimal_gen_acc, decimal_disc_acc), (decimal_gen_loss, decimal_disc_loss)


def train(dataset, test_input, generator, discriminator):
    history = {'Gen Acc': [], 'Disc Acc': [], 'Gen Loss': [], 'Disc Loss': []}

    for epoch in range(EPOCHS):
        start = time.time()

        for image_batch in dataset:
            gen_acc, disc_acc, gen_loss, disc_loss = train_step(image_batch, generator, discriminator)
            batch_acc, batch_loss = summmariseBatchPerformance(gen_acc, disc_acc, gen_loss, disc_loss)
            history = updatePerformanceStats(batch_acc, batch_loss, history)

            # Save the model every 15 epochs
            if (epoch + 1) % 1 == 0:
                generator.model.save_weights(f'./checkpoints/generator_{DATANAME}.h5', save_format="h5")
                discriminator.model.save_weights(f'./checkpoints/discriminator_{DATANAME}.h5', save_format="h5")
        filename = f'{DATANAME}_epoch_{epoch + 1:04d}.png'
        generate_and_save_images(generator.model, test_input, filename)
        print('Time for epoch {} is {} sec'.format(epoch + 1, time.time() - start))
        print(f"Loss | Gen: {np.mean(history['Gen Loss'])}, Disc: {np.mean(history['Disc Loss'])}")
        print(f"Acc | Gen: {np.mean(history['Gen Acc'])}, Disc: {np.mean(history['Disc Acc'])}")
    return history


def saveImagesAsGIF(anim_file=f'./{stats_dir}/{DATANAME}/dcgan.gif'):
    with imageio.get_writer(anim_file, mode='I') as writer:
        filenames = glob.glob(f'{DATANAME}*.png')
        filenames = sorted(filenames)
        for filename in filenames:
            image = imageio.imread(filename)
            writer.append_data(image)

