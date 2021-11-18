import tensorflow as tf
import tensorflow_addons as tfa
from tensorflow.keras import layers, initializers

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
        if self.n_filters != input_filter:
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


def downsample_block(
    input_tensor,
    channels,
    kernels,
    strides=2,
    apply_norm=True,
    apply_activation=True,
    apply_dropout=False,
):
    x = layers.Conv2D(
        channels,
        kernels,
        strides=strides,
        padding="same",
        use_bias=False,
        kernel_initializer=initializers.GlorotNormal(),
    )(input_tensor)
    x = tfa.layers.InstanceNormalization()(x) if apply_norm else x
    x = layers.LeakyReLU(0.2)(x) if apply_activation else x
    x = layers.Dropout(0.5)(x) if apply_dropout else x
    return x
