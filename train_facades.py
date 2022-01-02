from absl import app
from absl import flags
from absl import logging

from ml_collections.config_flags import config_flags

from gaugan.dataloader import FacadesDataLoader
from gaugan.models import GauGAN
from gaugan.callbacks import GanMonitor


FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("facades_configs")


def main(_):
    data_loader = FacadesDataLoader(
        target_image_height=FLAGS.facades_configs.image_height,
        target_image_width=FLAGS.facades_configs.image_width,
        num_classes=FLAGS.facades_configs.num_classes,
    )
    train_dataset, val_dataset = data_loader.get_datasets(
        batch_size=FLAGS.facades_configs.batch_size,
        split_fraction=FLAGS.facades_configs.split_fraction,
    )
    gaugan_model = GauGAN(
        image_size=FLAGS.facades_configs.image_height,
        num_classes=FLAGS.facades_configs.num_classes,
        batch_size=FLAGS.facades_configs.batch_size,
        latent_dim=FLAGS.facades_configs.latent_dimention,
        feature_loss_coeff=FLAGS.facades_configs.feature_loss_coefficient,
        vgg_feature_loss_coeff=FLAGS.facades_configs.vgg_feature_loss_coefficient,
        kl_divergence_loss_coeff=FLAGS.facades_configs.kl_divergence_loss_coefficient,
        encoder_downsample_factor=FLAGS.facades_configs.encoder_downsample_factor,
        discriminator_downsample_factor=FLAGS.facades_configs.discriminator_downsample_factor,
        alpha=FLAGS.facades_configs.common_configs.alpha,
        dropout=FLAGS.facades_configs.common_configs.dropout,
    )
    gaugan_model.compile(
        gen_lr=FLAGS.facades_configs.generator_learning_rate,
        disc_lr=FLAGS.facades_configs.discriminator_learning_rate,
    )
    gaugan_model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=FLAGS.facades_configs.epochs,
        callbacks=[GanMonitor(val_dataset, FLAGS.facades_configs.batch_size)],
    )


if __name__ == "__main__":
    app.run(main)
