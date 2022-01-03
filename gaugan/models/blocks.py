import tensorflow as tf
from tensorflow.keras import Sequential
from tensorflow.keras import layers, initializers
import tensorflow_addons as tfa

from .spade import SPADE


class ResidualBlock(layers.Layer):
    def __init__(self, filters: int, alpha: float, **kwargs):
        super().__init__(**kwargs)
        self.filters = filters
        self.alpha = alpha

    def build(self, input_shape):
        input_filter = input_shape[-1]
        self.spade_1 = SPADE(input_filter)
        self.spade_2 = SPADE(self.filters)
        self.conv_1 = layers.Conv2D(self.filters, 3, padding="same")
        self.conv_2 = layers.Conv2D(self.filters, 3, padding="same")
        self.learned_skip = False

        if self.filters != input_filter:
            self.learned_skip = True
            self.spade_3 = SPADE(input_filter)
            self.conv_3 = layers.Conv2D(self.filters, 3, padding="same")

    def call(self, input_tensor, mask):
        x = self.spade_1(input_tensor, mask)
        x = self.conv_1(tf.nn.leaky_relu(x, self.alpha))
        x = self.spade_2(x, mask)
        x = self.conv_2(tf.nn.leaky_relu(x, self.alpha))
        skip = (
            self.conv_3(tf.nn.leaky_relu(self.spade_3(input_tensor, mask), self.alpha))
            if self.learned_skip
            else input_tensor
        )
        return skip + x


def downsample_block(
    channels,
    kernels,
    strides=2,
    apply_norm=True,
    apply_activation=True,
    apply_dropout=False,
    alpha: float = 0.2,
    dropout: float = 0.5,
) -> Sequential:
    block = Sequential()
    block.add(
        layers.Conv2D(
            channels,
            kernels,
            strides=strides,
            padding="same",
            use_bias=False,
            kernel_initializer=initializers.GlorotNormal(),
        )
    )
    if apply_norm:
        block.add(tfa.layers.InstanceNormalization())
    if apply_activation:
        block.add(layers.LeakyReLU(alpha))
    if apply_dropout:
        block.add(layers.Dropout(dropout))
    return block
