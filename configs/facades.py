import ml_collections

from configs import commons


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.common_configs = commons.get_config()

    config.image_height = 256
    config.image_width = 256
    config.num_classes = 12
    config.batch_size = 4
    config.split_fraction = 0.2
    config.latent_dimention = 256
    config.feature_loss_coefficient = 10.0
    config.vgg_feature_loss_coefficient = 0.1
    config.kl_divergence_loss_coefficient = 0.1
    config.encoder_downsample_factor = 64
    config.discriminator_downsample_factor = 64
    config.generator_learning_rate = 1e-4
    config.discriminator_learning_rate = 4e-4
    config.epochs = 15
    config.epoch_interval = 5
    config.plot_save_dir = "checkpoints/plots"
    config.model_save_dir = "checkpoint/models"

    return config
