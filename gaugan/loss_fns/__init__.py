import tensorflow as tf

from .content_loss import ContentLoss


def k_l_divergence(mean, logvar):
    return -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))


def generator_loss(y):
    return -tf.reduce_mean(y)
