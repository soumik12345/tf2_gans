from tensorflow import keras
from tensorflow.keras import layers, Model, Input

from .blocks import downsample_block


def build_discriminator(image_size: int = 256, downsample_factor: int = 64):
    image_1 = Input(shape=[image_size, image_size, 3])
    image_2 = Input(shape=[image_size, image_size, 3])
    x = layers.Concatenate()([image_1, image_2])
    x1 = downsample_block(x, downsample_factor, 4, apply_norm=False)
    x2 = downsample_block(x1, 2 * downsample_factor, 4)
    x3 = downsample_block(x2, 4 * downsample_factor, 4)
    x4 = downsample_block(x3, 8 * downsample_factor, 4)
    x5 = layers.Conv2D(1, 4)(x4)
    return Model([image_1, image_2], [x1, x2, x3, x4, x5], name="discriminator")
