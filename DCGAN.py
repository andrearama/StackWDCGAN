import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import src.util as util, src.network as network, src.loss as loss

# Variables
tf.logging.set_verbosity(tf.logging.ERROR)
basePath = "/home/francesco/UQ/Job/ArtGAN/"
imgSize = 64
X = util.getData(basePath + "dataset/SemArt/", ["semart_train.csv", "semart_val.csv", "semart_test.csv"], imgSize)
print(X.shape)
assert(X.max() == 1. and X.min() == -1.)

# Parameters
epochs = 4000
batchSize = 64
lr = 1e-4
beta1 = 0.5
Z_dim = 100

# Tensorflow
tf.reset_default_graph()

# Dataset
def generator():
    for el in X:
        yield el

dataset = tf.data.Dataset.from_generator(generator, (tf.float32), output_shapes=(tf.TensorShape([imgSize, imgSize, 3])))
dataset = dataset.repeat(epochs).shuffle(buffer_size=len(X)).batch(batchSize, drop_remainder=True)
iterator = dataset.make_one_shot_iterator()

# Inputs
X_tensor = iterator.get_next()
Z = tf.placeholder(tf.float32, [None, Z_dim])
isTraining = tf.placeholder(dtype=tf.bool)

# Networks
G_z = network.generator(Z, isTraining)
D_logits_real = network.discriminator(X_tensor, isTraining)
D_logits_fake = network.discriminator(G_z, isTraining)

# Loss and optimizer
D_loss, G_loss = loss.GAN_Loss(D_logits_real, D_logits_fake, 0.9)
D_optimizer, G_optimizer = loss.GAN_Optimizer(D_loss, G_loss, lr, beta1)

# Tensorboard | VISUALIZE => tensorboard --logdir=.
summaries_dir = basePath + "checkpoints"
tf.summary.scalar('D_loss', D_loss)
tf.summary.scalar('G_loss', G_loss)
merged_summary = tf.summary.merge_all()

# Training
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
    saver = tf.train.Saver(max_to_keep=3000)
    sess.run(tf.global_variables_initializer())
    summary_writer = tf.summary.FileWriter(summaries_dir, graph=tf.get_default_graph())

    globalStep = 0
    try: 
        while True:
            # Sample Gaussian noise
            noise = network.sample_noise([batchSize, Z_dim])

            # Train Discriminator
            _, summary = sess.run([D_optimizer, merged_summary], feed_dict={isTraining: True, Z: noise})
            summary_writer.add_summary(summary, globalStep)

            # Train Generator
            sess.run(G_optimizer, feed_dict={ isTraining: True, Z: noise })
                
            # Save checkpoints and images
            if globalStep % 250 == 0:
                save_path = saver.save(sess, basePath + "checkpoints/ckpt-" + str(globalStep))
                G_output = sess.run(G_z, feed_dict={ isTraining : False, Z: network.sample_noise([5, Z_dim]) })
                util.saveImages(basePath + "images/out-" + str(globalStep), G_output)
                
            globalStep += 1
    except tf.errors.OutOfRangeError:
        pass