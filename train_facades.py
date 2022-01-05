import os
import datetime
from absl import app
from absl import flags
from absl import logging

from wandb.keras import WandbCallback
import wandb

from ml_collections.config_flags import config_flags
from tensorflow import keras

from gaugan.dataloader import FacadesDataLoader
from gaugan.models import GauGAN
from gaugan.callbacks import GanMonitor


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("facades_configs")


def main(_):
    if FLAGS.facades_configs.wandb_project:
        wandb.login()
        wandb.init(
            project=FLAGS.facades_configs.wandb_project,
            config=FLAGS.facades_configs.to_dict(),
        )

    logging.info("Building TensorFlow Datasets...")
    data_loader = FacadesDataLoader(
        target_image_height=FLAGS.facades_configs.image_height,
        target_image_width=FLAGS.facades_configs.image_width,
        num_classes=FLAGS.facades_configs.num_classes,
        data_dir=FLAGS.facades_configs.dataset_dir,
    )
    train_dataset, val_dataset = data_loader.get_datasets(
        batch_size=FLAGS.facades_configs.batch_size,
        split_fraction=FLAGS.facades_configs.split_fraction,
    )
    logging.info("Done!!!")

    logging.info("Building GauGAN Model...")
    gaugan_model = GauGAN(
        image_size=FLAGS.facades_configs.image_height,
        num_classes=FLAGS.facades_configs.num_classes,
        batch_size=FLAGS.facades_configs.batch_size,
        hyperparameters=FLAGS.facades_configs.hyperparameters,
    )
    logging.info("Done!!!")

    logging.info("Compiling GauGAN Model...")
    gaugan_model.compile(
        gen_lr=FLAGS.facades_configs.hyperparameters.generator_learning_rate,
        disc_lr=FLAGS.facades_configs.hyperparameters.discriminator_learning_rate,
    )
    logging.info("Done!!!")

    logging.info("Creating callbacks...")
    if not os.path.isdir(FLAGS.facades_configs.plot_save_dir):
        os.makedirs(FLAGS.facades_configs.plot_save_dir)
    gan_monitor_callback = GanMonitor(
        val_dataset,
        FLAGS.facades_configs.batch_size,
        epoch_interval=FLAGS.facades_configs.epoch_interval,
        use_wandb=True if FLAGS.facades_configs.wandb_project else False,
        plot_save_dir=None,  # Change `FLAGS.facades_configs.plot_save_dir` to control this.
    )
    checkpoint_callback = keras.callbacks.ModelCheckpoint(
        FLAGS.facades_configs.model_save_dir,
        monitor="val_kid",
        save_best_only=True,
        save_weights_only=True,
        mode="max",
    )
    logging.info("Done!!!")

    logging.info("Training GauGAN on Facades Dataset...")
    gaugan_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FLAGS.facades_configs.hyperparameters.epochs,
        callbacks=[gan_monitor_callback, WandbCallback(), checkpoint_callback],
    )
    logging.info("Training completed successfully!!!")

    logging.info("Load the best generator and discriminator checkpoints...")
    gaugan_model.load_weights(FLAGS.facades_configs.model_save_dir)

    logging.info(f"Saving models at {FLAGS.facades_configs.model_save_dir}")
    timestamp = datetime.datetime.utcnow().strftime("%y%m%d-%H%M%S")
    gaugan_model.save(
        os.path.join(FLAGS.facades_configs.model_save_dir, timestamp),
    )
    logging.info("Done!!!")


if __name__ == "__main__":
    app.run(main)
