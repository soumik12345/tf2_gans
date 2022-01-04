import ml_collections

from configs import hyperparameters


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.hyperparameters = hyperparameters.get_config()

    config.dataset_dir = "facades_data"
    config.image_height = 256
    config.image_width = 256
    config.num_classes = 12
    config.batch_size = 4
    config.split_fraction = 0.2

    config.epoch_interval = 5
    config.plot_save_dir = "checkpoints/plots"
    config.model_save_dir = "checkpoint/models"

    config.wandb_project = "GauGAN" # Setting this to None won't log anything to `wandb`.

    return config
