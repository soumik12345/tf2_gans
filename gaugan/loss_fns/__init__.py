import tensorflow as tf

from .content_loss import ContentLoss


def kl_divergence_loss(mean, variance):
    return -0.5 * tf.reduce_sum(1 + variance - tf.square(mean) - tf.exp(variance))


def generator_loss(y):
    return -tf.reduce_mean(y)
