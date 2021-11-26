from datetime import datetime
import ml_collections
import os


def get_config():
    configs = ml_collections.ConfigDict()

    # Dataset.
    configs.dataset_root_dir = "./facades/"
    configs.batch_size = 256
    configs.num_classes = 12
    configs.latent_dim = 256
    configs.batch_size = 16

    # Training.
    configs.wandb_project = "gaugan"
    configs.generator_lr = 1e-4
    configs.discriminator_lr = 4e-4
    configs.num_epochs = 50
    timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")
    configs.training_dir = f"./gaugan-training_dir-{timestamp}"
    os.makedirs(configs.training_dir, exist_ok=True)

    return configs
