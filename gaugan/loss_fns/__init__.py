import tensorflow as tf


def k_l_divergence(mean, logvar):
    return -0.5 * tf.reduce_sum(1 + logvar - tf.square(mean) - tf.exp(logvar))
