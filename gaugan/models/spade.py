import tensorflow as tf
from tensorflow.keras import layers


class SpatialAdaptiveNormalization(layers.Layer):
    def __init__(self, n_filters: int, epsilon: float = 1e-5, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.epsilon = epsilon
        self.conv_layer = layers.Conv2D(128, 3, padding="same", activation="relu")
        self.gamma_conv_layer = layers.Conv2D(n_filters, 3, padding="same")
        self.beta_conv_layer = layers.Conv2D(n_filters, 3, padding="same")

    def build(self, input_shape):
        self.resize_shape = input_shape[1:3]

    def call(self, input_tensor, raw_mask):
        mask = tf.image.resize(raw_mask, self.resize_shape, method="nearest")
        x = self.conv_layer(mask)
        gamma = self.gamma_conv_layer(x)
        beta = self.beta_conv_layer(x)
        mean, var = tf.nn.moments(input_tensor, axes=(0, 1, 2), keepdims=True)
        std = tf.sqrt(var + self.epsilon)
        normalized = (input_tensor - mean) / std
        return gamma * normalized + beta
