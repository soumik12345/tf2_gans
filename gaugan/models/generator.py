import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from .blocks import downsample_block, ResidualBlock


def build_encoder(
    image_size: int = 256, encoding_dimension: int = 64, latent_dimension: int = 256
):
    image = keras.Input(shape=[image_size, image_size, 3])
    x = downsample_block(encoding_dimension, 3, norm=False)(image)
    x = downsample_block(2 * encoding_dimension, 3)(x)
    x = downsample_block(4 * encoding_dimension, 3)(x)
    x = downsample_block(8 * encoding_dimension, 3)(x)
    x = downsample_block(8 * encoding_dimension, 3)(x)
    x = layers.Flatten()(x)
    mean = layers.Dense(latent_dimension, name="mean")(x)
    logvar = layers.Dense(latent_dimension, name="logvar")(x)
    return keras.Model(image, [mean, logvar], name="encoder")


def build_generator(
    image_size: int = 256, latent_dimension: int = 256, n_classes: int = 12
):
    latent_input = keras.Input(shape=[latent_dimension])
    label_map_input = keras.Input(shape=[image_size, image_size, n_classes])
    x = layers.Dense(4 * 4 * 1024)(latent_input)
    x = layers.Reshape((4, 4, 1024))(x)
    x = ResidualBlock(n_filters=1024)(x, label_map_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = ResidualBlock(n_filters=1024)(x, label_map_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = ResidualBlock(n_filters=1024)(x, label_map_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = ResidualBlock(n_filters=512)(x, label_map_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = ResidualBlock(n_filters=256)(x, label_map_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = ResidualBlock(n_filters=128)(x, label_map_input)
    x = layers.UpSampling2D((2, 2))(x)
    x = tf.nn.leaky_relu(x, 0.2)
    output_image = layers.Conv2D(3, 4, activation="tanh", padding="same")(x)
    return keras.Model([latent_input, label_map_input], output_image, name="generator")
