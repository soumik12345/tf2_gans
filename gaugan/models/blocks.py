import tensorflow as tf
from tensorflow.keras import layers

from .spade import SpatialAdaptiveNormalization


class ResidualBlock(layers.Layer):
    def __init__(self, n_filters: int, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_filters = n_filters

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.spade_1 = SpatialAdaptiveNormalization(input_filter)
        self.spade_2 = SpatialAdaptiveNormalization(self.n_filters)
        self.conv_1 = layers.Conv2D(self.n_filters, 3, padding="same")
        self.conv_2 = layers.Conv2D(self.n_filters, 3, padding="same")
        self.learned_skip = False
        if self.filters != input_filter:
            self.learned_skip = True
            self.spade_3 = SpatialAdaptiveNormalization(input_filter)
            self.conv_3 = layers.Conv2D(self.n_filters, 3, padding="same")

    def call(self, input_tensor, mask):
        x = self.spade_1(input_tensor, mask)
        x = self.conv_1(tf.nn.leaky_relu(x, 0.2))
        x = self.spade_2(x, mask)
        x = self.conv_2(tf.nn.leaky_relu(x, 0.2))
        if self.learned_skip:
            skip = self.spade_3(input_tensor, mask)
            skip = self.conv_3(tf.nn.leaky_relu(skip, 0.2))
        else:
            skip = input_tensor
        return skip + x


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
