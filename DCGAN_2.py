import tensorflow as tf
from tensorflow.keras import layers
import numpy as np
import os
import src.util as util
import matplotlib.image as mpimg


# Getting the data
basePath = "/home/francesco/UQ/Job/ArtGAN/"
datasetPath = "dataset/art-images-drawings-painting-sculpture-engraving/dataset/dataset_updated/"
classes = {"painting" : 0, "iconography" : 1, "engraving" : 2, "drawings" : 3, "sculpture" : 4}
size = 64
X, Y_ = util.getData(basePath + datasetPath, classes, size=size)
print(X.shape)
assert(X.max() == 1. and X.min() == -1.)

# Parameters
batchSize = 64
epochs = 3000
Z_dim = 100

# Prepare dataset
X_dataset = tf.data.Dataset.from_tensor_slices(X).repeat(epochs).shuffle(len(X)).batch(batchSize, drop_remainder=True)

def generatorModel(Z_dim):
    model = tf.keras.Sequential()
    model.add(layers.Dense(4*4*512, use_bias=False, input_shape=(Z_dim,)))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())
    model.add(layers.Reshape((4, 4, 512)))

    model.add(layers.Conv2DTranspose(256, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(128, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(64, (5, 5), strides=(2, 2), padding='same', use_bias=False))
    model.add(layers.BatchNormalization())
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2DTranspose(3, (5, 5), strides=(2, 2), padding='same', use_bias=False, activation='tanh'))

    return model


def discriminatorModel(imgSize):
    model = tf.keras.Sequential()
    model.add(layers.Conv2D(64, (5, 5), strides=(2, 2), padding='same', input_shape=[imgSize, imgSize, 3]))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(128, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())
    
    model.add(layers.Conv2D(256, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())

    model.add(layers.Conv2D(512, (5, 5), strides=(2, 2), padding='same'))
    model.add(layers.LeakyReLU())


    model.add(layers.Flatten())
    model.add(layers.Dense(1))

    return model

# Models
generator = generatorModel(Z_dim)
print(generator.summary())

discriminator = discriminatorModel(size)
print(discriminator.summary())

# This method returns a helper function to compute cross entropy loss
cross_entropy = tf.keras.losses.BinaryCrossentropy(from_logits=True)


def discriminator_loss(real_output, fake_output, labelSmoothing=1.0):
    real_loss = cross_entropy(tf.ones_like(real_output) * labelSmoothing, real_output)
    fake_loss = cross_entropy(tf.zeros_like(fake_output), fake_output)
    total_loss = real_loss + fake_loss
    return total_loss


def generator_loss(fake_output):
    return cross_entropy(tf.ones_like(fake_output), fake_output)


generator_optimizer = tf.keras.optimizers.Adam(1e-4)
discriminator_optimizer = tf.keras.optimizers.Adam(1e-4)


checkpoint = tf.train.Checkpoint(generator_optimizer=generator_optimizer,
                                 discriminator_optimizer=discriminator_optimizer,
                                 generator=generator,
                                 discriminator=discriminator)

@tf.function
def train_step(images):
    noise = tf.random.normal([batchSize, Z_dim])

    with tf.GradientTape() as gen_tape, tf.GradientTape() as disc_tape:
      generated_images = generator(noise, training=True)

      real_output = discriminator(images, training=True)
      fake_output = discriminator(generated_images, training=True)

      gen_loss = generator_loss(fake_output)
      disc_loss = discriminator_loss(real_output, fake_output)

    gradients_of_generator = gen_tape.gradient(gen_loss, generator.trainable_variables)
    gradients_of_discriminator = disc_tape.gradient(disc_loss, discriminator.trainable_variables)

    generator_optimizer.apply_gradients(zip(gradients_of_generator, generator.trainable_variables))
    discriminator_optimizer.apply_gradients(zip(gradients_of_discriminator, discriminator.trainable_variables))



def train(dataset, epochs):
	globalStep = 0
	for image_batch in dataset:
		train_step(image_batch)
		
		if globalStep % 2 == 0:
			save_images(generator, globalStep)
			checkpoint.save(file_prefix = basePath + 'checkpoints/ckpt-' + str(globalStep))
		globalStep += 1


def save_images(model, step, images=6):

	for i in range(images):
		noise = tf.random.normal([1, Z_dim])
		# training is False => Layers in inference mode (batchnorm)
		prediction = model(noise, training=False)

		mpimg.imsave(basePath + 'images/img-' + str(step) + "-" + str(i) + ".png",  ( (np.array(prediction[0]) * 127.5) + 127.5 ).astype(np.uint8) )


train(X_dataset, epochs)