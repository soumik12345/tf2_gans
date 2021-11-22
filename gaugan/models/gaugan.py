import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import optimizers, losses, models, Model

from .generator import build_encoder, build_generator
from .discriminator import build_discriminator
from .sampling import GaussianSampling
from ..loss_fns import ContentLoss, generator_loss, kl_divergence_loss


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
        train_encoder: bool = False,
        *args,
        **kwargs
    ):
        """GauGAN Model

        Reference: https://arxiv.org/abs/1903.07291

        Args:
            image_size (int): Input image size
            encoding_dimension (int): Encoding dimension
            latent_dimension (int): Latent dimension
            downsample_factor (int): Downsample factor for Discriminator
            n_classes (int): Number of semantic classes
            feature_loss_weight (float): Coefficient of feature loss
            content_loss_weight (float): Coefficient of content loss
            kl_divergence_weight (float): Coefficient of KL divergence loss
            batch_size (int): Batch size
            train_encoder (bool): Flag to train encoder or not during generator train step
        """
        super().__init__(*args, **kwargs)
        self.encoder = build_encoder(image_size, encoding_dimension, latent_dimension)
        self.generator = build_generator(image_size, latent_dimension, n_classes)
        self.discriminator = build_discriminator(image_size, downsample_factor)
        self.sampler = GaussianSampling(batch_size, latent_dimension)
        # Default value of the loss coefficients have been taken
        # from https://arxiv.org/pdf/1711.11585.pdf
        self.feature_loss_weight = feature_loss_weight
        self.content_loss_weight = content_loss_weight
        self.kl_divergence_weight = kl_divergence_weight
        self.batch_size = batch_size
        self.train_encoder = train_encoder
        self.patch_size = self.discriminator.output_shape[-1][1]

    def summary(self, line_length=None, positions=None, print_fn=None):
        self.encoder.summary(line_length, positions, print_fn)
        self.generator.summary(line_length, positions, print_fn)
        self.discriminator.summary(line_length, positions, print_fn)

    def compile(
        self,
        generator_lr: float = 1e-4,
        discriminator_lr: float = 4e-4,
        *args,
        **kwargs
    ):
        super().compile(*args, **kwargs)
        self.generator_optimizer = optimizers.Adam(
            learning_rate=generator_lr, beta_1=0.0
        )
        self.discriminator_optimizer = optimizers.Adam(
            learning_rate=discriminator_lr, beta_1=0.0
        )
        self.content_loss = ContentLoss()
        self.discriminator_hinge_loss = losses.Hinge()
        self.mean_absolute_error = losses.MeanAbsoluteError()

    def feature_matching_loss(self, real_features, fake_features):
        loss = 0
        for i in range(len(real_features) - 1):
            loss += self.mean_absolute_error(real_features[i], fake_features[i])
        return loss

    def _compute_discriminator_loss(self, real_images_A, real_images_B, fake_images_B):
        fake_prediction = self.discriminator([real_images_A, fake_images_B])[-1]
        real_prediction = self.discriminator([real_images_A, real_images_B])[-1]
        fake_loss = self.discriminator_hinge_loss(-1.0, fake_prediction)
        real_loss = self.discriminator_hinge_loss(1.0, real_prediction)
        total_discriminator_loss = 0.5 * (fake_loss + real_loss)
        return total_discriminator_loss

    def _compute_generator_losss(
        self, latent, real_images_A, real_images_B, labels_A, mean, variance
    ):
        real_d_output = self.discriminator([real_images_A, real_images_B])
        fake_images = self.generator([latent, labels_A])
        fake_d_output = self.discriminator([real_images_A, fake_images])
        prediction = fake_d_output[-1]
        g_loss = generator_loss(prediction)
        kl_loss = kl_divergence_loss(mean, variance)
        content_loss = self.content_loss(real_images_B, fake_images)
        feature_loss = self.feature_matching_loss(real_d_output, fake_d_output)
        total_generator_loss = (
            g_loss
            + self.kl_divergence_weight * kl_loss
            + self.content_loss_weight * content_loss
            + self.feature_loss_weight * feature_loss
        )
        return g_loss, kl_loss, content_loss, feature_loss, total_generator_loss

    def _discriminator_train_step(self, latent, real_images_A, real_images_B, labels_A):
        fake_images_B = self.generator([latent, labels_A])
        with tf.GradientTape() as d_tape:
            total_loss = self._compute_discriminator_loss(
                real_images_A, real_images_B, fake_images_B
            )
        gradients = d_tape.gradient(total_loss, self.discriminator.trainable_variables)
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )
        return total_loss

    def _generator_train_step(
        self, latent, real_images_A, real_images_B, labels_A, mean, variance
    ):
        with tf.GradientTape() as g_tape:
            (
                g_loss,
                kl_loss,
                content_loss,
                feature_loss,
                total_generator_loss,
            ) = self._compute_generator_losss(
                latent, real_images_A, real_images_B, labels_A, mean, variance
            )
        trainable_variables = self.generator.trainable_variables
        if self.train_encoder:
            trainable_variables += self.encoder.trainable_variables
        gradients = g_tape.gradient(total_generator_loss, trainable_variables)
        self.generator_optimizer.apply_gradients(zip(gradients, trainable_variables))
        return g_loss, kl_loss, content_loss, feature_loss

    def train_step(self, data):
        real_images_A, real_images_B, labels_A = data
        mean, variance = self.encoder(real_images_B)
        latent_input = self.sampler([mean, variance])
        discriminator_loss = self._discriminator_train_step(
            latent_input, real_images_A, real_images_B, labels_A
        )
        g_loss, kl_loss, content_loss, feature_loss = self._generator_train_step(
            latent_input, real_images_A, real_images_B, labels_A, mean, variance
        )
        return {
            "discriminator_loss": discriminator_loss,
            "generator_loss": g_loss,
            "kl_divergence_loss": kl_loss,
            "content_loss": content_loss,
            "feature_loss": feature_loss,
        }

    def test_step(self, data):
        real_images_A, real_images_B, labels_A = data
        mean, variance = self.encoder(real_images_B)
        latent_input = self.sampler([mean, variance])
        discriminator_loss = self._compute_discriminator_loss(
            latent_input, real_images_A, real_images_B, labels_A
        )
        (
            g_loss,
            kl_loss,
            content_loss,
            feature_loss,
            total_generator_loss,
        ) = self._compute_generator_losss(
            latent_input, real_images_A, real_images_B, labels_A, mean, variance
        )
        return {
            "discriminator_loss": discriminator_loss,
            "generator_loss": total_generator_loss,
            "kl_divergence_loss": kl_loss,
            "content_loss": content_loss,
            "feature_loss": feature_loss,
        }

    def call(self, inputs, training=None, mask=None):
        real_images_B, labels_A = inputs[0], inputs[1]
        mean, variance = self.encoder(real_images_B)
        latent_input = self.sampler([mean, variance])
        generated_images = self.generator([latent_input, labels_A])
        return generated_images

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
