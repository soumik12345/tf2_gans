import ml_collections

from configs import hyperparameters


def get_config() -> ml_collections.ConfigDict:
    config = ml_collections.ConfigDict()

    config.hyperparameters = hyperparameters.get_config()

    config.dataset_dir = "cocostuff10k_data"
    config.image_height = 256
    config.image_width = 256
    config.num_classes = 172
    config.batch_size = 16
    config.split_fraction = 0.1

    config.epoch_interval = 5
    config.plot_save_dir = "checkpoints/plots"
    config.model_save_dir = "checkpoints/models"

    config.wandb_project = "GauGAN"
    config.wandb_experiment_name = "cocostuff10k"

    return config
