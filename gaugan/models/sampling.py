import tensorflow as tf
from tensorflow.keras import layers


class GaussianSampling(layers.Layer):
    def __init__(self, batch_size: int, latent_dimension: int):
        super(GaussianSampling, self).__init__()
        self.batch_size = batch_size
        self.latent_dimension = latent_dimension

    def call(self, inputs):
        means, logvar = inputs
        epsilon = tf.random.normal(
            shape=(self.batch_size, self.latent_dimension), mean=0.0, stddev=1.0
        )
        return means + tf.exp(0.5 * logvar) * epsilon
