import tensorflow as tf
import datetime

# discriminator
def discriminator(images, reuse_variables=None):
    pass


# generator
def generator(z, batch_size, z_dim):
    pass

# input placeholder
z_dimensions = 100
z_placeholder = tf.placeholder(tf.float32, [None, z_dimensions], name='z_placeholder')
x_placeholder = tf.placeholder(tf.float32, shape=[None, 28, 28, 1], name='x_placeholder')
Gz = generator(z_placeholder, batch_size=50, z_dim=z_dimensions)
Dx = discriminator(images=x_placeholder)
Dg = discriminator(Gz, reuse_variables=True)

# loss function
d_loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dx, labels=tf.ones_like(Dx)))
d_loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.zeros_like(Dg)))
g_loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=Dg, labels=tf.ones_like(Dg)))

# variable scope
tvars = tf.trainable_variables()
d_vars = [var for var in tvars if 'd_' in var.name]

g_vars = [var for var in tvars if 'g_' in var.name]

# trainer
d_trainer_fake = tf.train.AdamOptimizer(0.0003).minimize(d_loss_fake, var_list=d_vars)
d_trainer_real = tf.train.AdamOptimizer(0.0003).minimize(d_loss_real, var_list=d_vars)

g_trainer = tf.train.AdamOptimizer(0.0001).minimize(g_loss, var_list=g_vars)

# summary
tf.get_variable_scope().reuse_variables()

tf.summary.scalar('Generator loss', g_loss)
tf.summary.scalar('Discriminator_loss_fake', d_loss_fake)
tf.summary.scalar('Descriminator_loss_real', d_loss_real)

merged = tf.summary.merge_all()
logdir = "tensorboard/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S") + "/"
writer = tf.summary.FileWriter(logdir=logdir, session=sess.graph)

# training