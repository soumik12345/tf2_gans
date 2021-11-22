import tensorflow as tf
from tensorflow.keras import layers, Model, Input

from .blocks import downsample_block, ResidualBlock


def build_encoder(
    image_size: int = 256, encoding_dimension: int = 64, latent_dimension: int = 256
) -> Model:
    """Build GauGAN Encoder Model

    Reference: https://arxiv.org/abs/1903.07291

    Args:
        image_size (int): input image size
        encoding_dimension (int): encoding dimension
        latent_dimension (int): latent dimension
    
    Returns:
        Encoder Model (tf.keras.Model)
    """
    image = Input(shape=[image_size, image_size, 3])
    x = downsample_block(image, encoding_dimension, 3, apply_norm=False)
    x = downsample_block(x, 2 * encoding_dimension, 3)
    x = downsample_block(x, 4 * encoding_dimension, 3)
    x = downsample_block(x, 8 * encoding_dimension, 3)
    x = downsample_block(x, 8 * encoding_dimension, 3)
    x = layers.Flatten()(x)
    mean = layers.Dense(latent_dimension, name="mean")(x)
    variance = layers.Dense(latent_dimension, name="variance")(x)
    return Model(image, [mean, variance], name="encoder")


def build_generator(
    image_size: int = 256, latent_dimension: int = 256, n_classes: int = 12
) -> Model:
    """Build GauGAN Generator Model

    Reference: https://arxiv.org/abs/1903.07291

    Args:
        image_size (int): input image size
        latent_dimension (int): latent dimension
        n_classes (int): number of semantic classes
    
    Returns:
        Generator Model (tf.keras.Model)
    """
    latent_input = Input(shape=[latent_dimension])
    label_map_input = Input(shape=[image_size, image_size, n_classes])
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
    return Model([latent_input, label_map_input], output_image, name="generator")
