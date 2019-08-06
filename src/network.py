import tensorflow as tf
import numpy as np


def sample_noise(size, mu=0., sigma=1.):
    return np.random.normal(mu, sigma, size=size)
    #return np.random.uniform(-1., 1., size=size)


def generator(Z, isTraining):
    
    with tf.variable_scope('generator', reuse=tf.AUTO_REUSE):
        size1, size2, size3 = 4, 4, 512
        fc1 = tf.reshape(tf.layers.dense(Z, size1*size2*size3, use_bias=False), [-1, size1, size2, size3])
        lrelu0 = tf.nn.leaky_relu(tf.layers.batch_normalization(fc1, training=isTraining))

        deconv2 = tf.layers.conv2d_transpose(lrelu0, 256, 5, 2, padding='SAME', use_bias=False)
        lrelu2 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv2, training=isTraining))

        deconv3 = tf.layers.conv2d_transpose(lrelu2, 128, 5, 2, padding='SAME', use_bias=False)
        lrelu3 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv3, training=isTraining))

        deconv4 = tf.layers.conv2d_transpose(lrelu3, 64, 5, 2, padding='SAME', use_bias=False)
        lrelu4 = tf.nn.leaky_relu(tf.layers.batch_normalization(deconv4, training=isTraining))

        output = tf.tanh(tf.layers.conv2d_transpose(lrelu4, 3, 5, 2, padding='SAME', use_bias=False))

        print(Z)
        print(lrelu0)
        print(lrelu2)
        print(lrelu3)
        print(lrelu4)
        print(output)
        print()
        return output


def discriminator(X, isTraining):
    
    with tf.variable_scope('discriminator', reuse=tf.AUTO_REUSE):
        lrelu1 = tf.nn.leaky_relu(tf.layers.conv2d(X, 64, 5, 2, 'SAME'))
        lrelu2 = tf.nn.leaky_relu(tf.layers.conv2d(lrelu1, 128, 5, 2, 'SAME'))
        lrelu3 = tf.nn.leaky_relu(tf.layers.conv2d(lrelu2, 256, 5, 2, 'SAME'))
        lrelu4 = tf.nn.leaky_relu(tf.layers.conv2d(lrelu3, 512, 5, 2, 'SAME'))
        output = tf.layers.dense(tf.layers.flatten(lrelu4), 1)

        print(X)
        print(lrelu1)
        print(lrelu2)
        print(lrelu3)
        print(lrelu4)
        print(output)
        print()
        return output