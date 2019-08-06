import tensorflow as tf
import src.network as network
import matplotlib.pyplot as plt
import numpy as np

basePath = "/home/francesco/UQ/Job/ArtGAN/"
Z_dim = 100

tf.reset_default_graph()

batchSizeTensor = tf.placeholder(tf.int32)
Z = network.sample_noise(batchSizeTensor, Z_dim)
isTraining = tf.placeholder(dtype=tf.bool)
G_z = network.generator(Z, isTraining)

with tf.Session() as sess:
    saver = tf.train.Saver()
    sess.run(tf.global_variables_initializer())
    saver.restore(sess, basePath + "CHECKPOINTS_WDCGAN128/model-33400.ckpt")

    images = []
    for j in range(6):
        G_output = sess.run(G_z, feed_dict={ isTraining : False, batchSizeTensor : 1 })
        images.append(G_output[0])
    images = np.array(images)
    print(images.shape)