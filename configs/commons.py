import ml_collections


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.alpha = 0.2  # Hyperparameter for LeakyReLU in the downsample block
    config.dropout = 0.5  # Dropout fraction in the downsample block

    return config
