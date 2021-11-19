import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, models, Model

from .generator import build_encoder, build_generator
from .discriminator import build_discriminator
from .sampling import GaussianSampling
from ..loss_fns import ContentLoss, generator_loss, k_l_divergence


class GauGAN(Model):
    def __init__(
        self,
        image_size: int = 256,
        encoding_dimension: int = 64,
        latent_dimension: int = 256,
        downsample_factor: int = 64,
        n_classes: int = 12,
        feature_loss_weight: float = 10.0,
        content_loss_weight: float = 0.1,
        kl_divergence_weight: float = 0.1,
        batch_size: int = 16,
        *args,
        **kwargs
    ):
        super().__init__(*args, **kwargs)
        self.encoder = build_encoder(image_size, encoding_dimension, latent_dimension)
        self.generator = build_generator(image_size, latent_dimension, n_classes)
        self.discriminator = build_discriminator(image_size, downsample_factor)
        self.sampler = GaussianSampling(batch_size, latent_dimension)
        # Default value of the loss coefficients have been taken from https://arxiv.org/pdf/1711.11585.pdf
        self.feature_loss_weight = feature_loss_weight
        self.content_loss_weight = content_loss_weight
        self.kl_divergence_weight = kl_divergence_weight
        self.batch_size = batch_size
        self.patch_size = self.discriminator.output_shape[-1][1]

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.encoder.summary(line_length, positions, print_fn)
        self.generator.summary(line_length, positions, print_fn)
        self.discriminator.summary(line_length, positions, print_fn)

    def compile(self, *args, **kwargs):
        super().compile(*args, **kwargs)
        self.generator_optimizer = optimizers.Adam(learning_rate=1e-4, beta_1=0.0)
        self.discriminator_optimizer = optimizers.Adam(learning_rate=4e-4, beta_1=0.0)
        self.content_loss = ContentLoss()
        self.discriminator_hinge_loss = losses.Hinge()
        self.feature_matching_loss = losses.MeanAbsoluteError()

    def train_discriminator(self, latent, real_images_A, real_images_B, labels_A):
        fake_images_B = self.generator([latent, labels_A])
        with tf.GradientTape() as d_tape:
            pred_fake = self.discriminator(
                [real_images_A, fake_images_B], training=True
            )[-1]
            pred_real = self.discriminator(
                [real_images_A, real_images_B], training=True
            )[-1]
            loss_fake = self.discriminator_hinge_loss(pred_fake, -1.0)
            loss_real = self.discriminator_hinge_loss(pred_real, 1.0)
            total_loss = 0.5 * (loss_fake + loss_real)
        gradients = d_tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )
        return total_loss

    def train_generator(
        self, latent, real_images_A, real_images_B, labels_A, mean, logvar
    ):
        with tf.GradientTape() as g_tape:
            real_d_output = self.discriminator([real_images_A, real_images_B])
            fake_d_output, fake_image = self.generator(
                [latent, labels_A, real_images_A], training=True
            )
            pred = fake_d_output[-1]
            g_loss = generator_loss(pred)
            kl_loss = self.kl_divergence_weight * k_l_divergence(mean, logvar)
            content_loss = self.content_loss_weight * self.content_loss(
                real_images_B, fake_image
            )
            feature_loss = self.feature_loss_weight * tf.reduce_mean(
                self.feature_matching_loss(real_d_output, fake_d_output)
            )
            total_loss = g_loss + kl_loss + content_loss + feature_loss
        gradients = g_tape.gradient(total_loss, self.generator.trainable_variables)
        self.generator_optimizer.apply_gradients(
            zip(gradients, self.generator.trainable_variables)
        )
        return g_loss, kl_loss, content_loss, feature_loss, total_loss

    def train_step(self, data):
        real_images_A, real_images_B, labels_A = data
        mean, logvar = self.encoder(real_images_B)
        latent = self.sampler([mean, logvar])
        total_discriminator_loss = self.train_discriminator(
            latent, real_images_A, real_images_B, labels_A
        )
        (
            g_loss,
            kl_loss,
            content_loss,
            feature_loss,
            total_generator_loss,
        ) = self.train_generator(
            latent, real_images_A, real_images_B, labels_A, mean, logvar
        )
        return {
            "total_discriminator_loss": total_discriminator_loss,
            "total_discriminator_loss": total_discriminator_loss,
            "generator_loss": g_loss,
            "kl_loss": kl_loss,
            "content_loss": content_loss,
            "feature_loss": feature_loss,
            "total_generator_loss": total_generator_loss,
        }

    def save(
        self,
        filepath,
        overwrite=True,
        include_optimizer=True,
        save_format=None,
        signatures=None,
        options=None,
        save_traces=True,
    ):
        self.generator.save(
            filepath + "_generator",
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )
        self.discriminator.save(
            filepath + "_discriminator",
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.generator.save_weights(
            filepath + "_generator.h5",
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        self.discriminator.save_weights(
            filepath + "_discriminator.h5",
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(
        self,
        generator_filepath,
        discriminator_filepath,
        by_name=False,
        skip_mismatch=False,
        options=None,
    ):
        self.generator.load_weights(
            generator_filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )
        self.discriminator.load_weights(
            discriminator_filepath,
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )

    def load(
        self,
        generator_filepath,
        discriminator_filepath,
        custom_objects=None,
        compile=True,
        options=None,
    ):
        self.generator = models.load_model(
            filepath=generator_filepath,
            custom_objects=custom_objects,
            compile=compile,
            options=options,
        )
        self.discriminator = models.load_model(
            filepath=generator_filepath,
            custom_objects=custom_objects,
            compile=compile,
            options=options,
        )
