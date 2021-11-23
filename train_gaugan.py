from gaugan.models import gaugan
from gaugan.utils import GANMonitor
from gaugan.dataloader import PairedTranslationDataLoader
from gaugan import configs


# Initialize training configs.
training_configs = configs.get_config()
if training_configs["wandb_project"]:
    import wandb

    wandb.init(project=training_configs["wandb_project"], entity="gaugan")


# Prepare datasets.
dataloader = PairedTranslationDataLoader()
train_dataset, val_dataset = dataloader.get_datasets(
    dataset_path=training_configs["dataset_root_dir"]
)

# Initialize model.
gaugan_model = gaugan.GauGAN(train_encoder=training_configs.train_encoder)
gaugan_model.compile(
    generator_lr=training_configs.generator_lr,
    discriminator_lr=training_configs.discriminator_lr,
)

# Initialize GAN monitor.
validation_batch = next(iter(val_dataset))
gan_monitor = GANMonitor(
    validation_batch[1], validation_batch[2], training_configs.training_dir
)

# Train the GauGAN model.
gan_monitor.fit(
    train_dataset, validation_data=val_dataset, epochs=training_configs.num_epochs
)

# Serialize the model.
gaugan_model.save(training_configs.training_dir)
