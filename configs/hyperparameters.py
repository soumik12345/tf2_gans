import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.latent_dimention = 256
    config.feature_loss_coefficient = 10.0
    config.vgg_feature_loss_coefficient = 0.1
    config.kl_divergence_loss_coefficient = 0.1
    config.encoder_downsample_factor = 64
    config.discriminator_downsample_factor = 64
    config.generator_learning_rate = 1e-4
    config.discriminator_learning_rate = 4e-4
    config.epochs = 15

    return config
