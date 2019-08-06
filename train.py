import tensorflow as tf
import src.util as util, src.network as network, src.UNET_GAN as UNET_GAN, src.loss as loss

tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
basePath = "/scratch/cai/StackWDCGAN/"

# Get data in different sizes
imgSizes = [64, 256]
X = dict()
for size in imgSizes:
	X[size] = util.getData(basePath + "dataset/SemArt/", ["semart_train.csv", "semart_val.csv", "semart_test.csv"], size)
	print(X[size].shape)
	assert(X[size].max() == 1. and X[size].min() == -1.)

# Parameters
epochsG = 2500
epochsAE = 2000
batchSize = 64
lr = 1e-4
lam = 10.0
num_D = 5
beta1 = 0.
beta2 = 0.9
Z_dim = 100

# Datasets
def generatorG():
    for el in X[64]:
        yield el

def generatorAE():
	for el in X[256]:
		yield el

# Dataset 64x64
dataset64 = tf.data.Dataset.from_generator(generatorG, (tf.float32),
					output_shapes=(tf.TensorShape([64, 64, 3]))).repeat(epochsG).shuffle(buffer_size=len(X[64])).batch(batchSize, drop_remainder=True)
iterator64 = dataset64.make_initializable_iterator()
X64 = iterator64.get_next()

# Dataset 256x256
dataset256 = tf.data.Dataset.from_generator(generatorAE, (tf.float32),
					output_shapes=(tf.TensorShape([256, 256, 3]))).repeat(epochsAE).shuffle(buffer_size=len(X[256])).batch(batchSize, drop_remainder=True)
iterator256 = dataset256.make_initializable_iterator()
X256 = iterator256.get_next()

Z = tf.placeholder(tf.float32, [None, Z_dim])
isTraining = tf.placeholder(dtype=tf.bool)

''' 
	Networks
'''
# First GAN network
G_z = network.generator(Z, isTraining)
D_logits_real = network.discriminator(X64, isTraining)
D_logits_fake = network.discriminator(G_z, isTraining)

# UNET_GAN
G_AE = UNET_GAN.getAutoencoder(G_z, isTraining)
C_logits_real = UNET_GAN.getAutoencoderDiscriminator(X256, isTraining)
C_logits_fake = UNET_GAN.getAutoencoderDiscriminator(G_AE, isTraining)

'''
for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='generator'):
    print(i)
for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='discriminator'):
    print(i)
for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='autoencoder'):
    print(i)
for i in tf.get_collection(tf.GraphKeys.TRAINABLE_VARIABLES, scope='critic'):
    print(i)'''

# Compute gradient penalty for small network 64x64
eps = tf.random_uniform([batchSize, 1, 1, 1], minval=0., maxval=1.)
X_inter = eps * X64 + (1. - eps) * G_z
grad = tf.gradients(network.discriminator(X_inter, isTraining), [X_inter])[0]
gradients = tf.sqrt(tf.reduce_sum(tf.square(grad), [1, 2, 3]))
grad_penalty = lam * tf.reduce_mean(tf.square(gradients - 1))

# Losses
#D_loss, G_loss = loss.GAN_Loss(D_logits_real, D_logits_fake, 0.9)
D_loss, G_loss, = loss.WGAN_Loss(D_logits_real, D_logits_fake, grad_penalty)
C_loss, AE_loss = loss.GAN_Loss(C_logits_real, C_logits_fake, 0.9)

# Optimizers
D_optimizer, G_optimizer = loss.WGAN_Optimizer(D_loss, G_loss, lr, beta1, beta2)
C_optimizer, AE_optimizer = loss.Autoencoder_Optimizer(C_loss, AE_loss, lr, 0.5)

# Tensorboard | VISUALIZE => tensorboard --logdir=.
summaries_dir = basePath + "checkpoints"
G_summaries = [tf.summary.scalar('D_loss', -D_loss), tf.summary.scalar('G_loss', -G_loss)]
AE_summaries = [tf.summary.scalar('C_loss', C_loss), tf.summary.scalar('AE_loss', AE_loss)]

G_merged_summary = tf.summary.merge(G_summaries)
AE_merged_summary = tf.summary.merge(AE_summaries)

# Training
with tf.Session(config=tf.ConfigProto(gpu_options=tf.GPUOptions(allow_growth=True))) as sess:
	saver = tf.train.Saver(max_to_keep=3000)
	sess.run(tf.global_variables_initializer())
	G_summary_writer = tf.summary.FileWriter(summaries_dir, graph=tf.get_default_graph())
	AE_summary_writer = tf.summary.FileWriter(summaries_dir)

	# Train small network
	GStep = 0
	sess.run(iterator64.initializer)
	try:
		while True:
			# Sample Gaussian noise
			noise = network.sample_noise([batchSize, Z_dim])
			
			# Train discriminator (more at the beginning)
			D_iterations = 30 if (GStep < 5 or GStep % 500 == 0) else num_D
			for _ in range(D_iterations):
				_, summary = sess.run([D_optimizer, G_merged_summary], feed_dict={ isTraining: True, Z: noise })
				G_summary_writer.add_summary(summary, GStep)					

			# Train Generator
			sess.run(G_optimizer, feed_dict={ isTraining: True, Z: noise })

			# Save checkpoint and 64x64x3 images
			if GStep % 500 == 0:
				saver.save(sess, basePath + "checkpoints/ckpt-G-" + str(GStep))
				G_output = sess.run(G_z, feed_dict={ isTraining : False, Z: network.sample_noise([4, Z_dim]) })
				util.saveImages(basePath + "images/out-G-" + str(GStep), G_output)
			GStep += 1
	except tf.errors.OutOfRangeError:
		pass

	# Train AE GAN
	AEStep = 0
	sess.run(iterator256.initializer)
	try:
		while True:
			noise = network.sample_noise([batchSize, Z_dim])

			# Train Critic
			_, summary = sess.run([C_optimizer, AE_merged_summary], feed_dict={ isTraining: True, Z: noise })
			AE_summary_writer.add_summary(summary, AEStep)
			
			# Train AE only
			sess.run(AE_optimizer, feed_dict={ isTraining: True, Z: noise })

			if AEStep % 500 == 0:
				saver.save(sess, basePath + "checkpoints/ckpt-AE-" + str(AEStep))
				AE_output = sess.run(G_AE, feed_dict={ isTraining : False, Z: network.sample_noise([4, Z_dim]) })
				util.saveImages(basePath + "images/out-AE-" + str(AEStep), AE_output)
			AEStep += 1
	except tf.errors.OutOfRangeError:
		pass