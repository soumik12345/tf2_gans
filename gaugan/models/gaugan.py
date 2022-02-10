"""References:
- [Hands-On Image Generation with TensorFlow](https://www.packtpub.com/product/hands-on-image-generation-with-tensorflow/9781838826789)
- [Implementing SPADE using fastai](https://towardsdatascience.com/implementing-spade-using-fastai-6ad86b94030a)
- 
"""


import os
import ml_collections
import tensorflow as tf
from tensorflow.keras import Input, Model
from tensorflow.keras import optimizers, models

from .sampling import GaussianSampler
from .networks import build_encoder, build_generator, build_discriminator
from ..losses import (
    generator_loss,
    kl_divergence_loss,
    DiscriminatorLoss,
    FeatureMatchingLoss,
    VGGFeatureMatchingLoss,
)
from ..metrics import KID


class GauGAN(Model):
    def __init__(
        self,
        image_size: int,
        num_classes: int,
        batch_size: int,
        hyperparameters: ml_collections.ConfigDict,
        **kwargs,
    ):
        super().__init__(**kwargs)

        self.image_size = image_size
        self.latent_dim = hyperparameters.latent_dimention
        self.batch_size = batch_size
        self.num_classes = num_classes
        self.image_shape = (image_size, image_size, 3)
        self.mask_shape = (image_size, image_size, num_classes)
        self.feature_loss_coeff = hyperparameters.feature_loss_coefficient
        self.vgg_feature_loss_coeff = hyperparameters.vgg_feature_loss_coefficient
        self.kl_divergence_loss_coeff = hyperparameters.kl_divergence_loss_coefficient

        self.discriminator = build_discriminator(
            self.image_shape,
            downsample_factor=hyperparameters.discriminator_downsample_factor,
            alpha=hyperparameters.alpha,
            dropout=hyperparameters.dropout,
        )
        self.generator = build_generator(
            self.mask_shape, latent_dim=self.latent_dim, alpha=hyperparameters.alpha
        )
        self.encoder = build_encoder(
            self.image_shape,
            encoder_downsample_factor=hyperparameters.encoder_downsample_factor,
            latent_dim=self.latent_dim,
            alpha=hyperparameters.alpha,
            dropout=hyperparameters.dropout,
        )
        self.sampler = GaussianSampler(batch_size, self.latent_dim)
        self.patch_size, self.combined_model = self.build_combined_generator()

        self.disc_loss_tracker = tf.keras.metrics.Mean(name="disc_loss")
        self.gen_loss_tracker = tf.keras.metrics.Mean(name="gen_loss")
        self.feat_loss_tracker = tf.keras.metrics.Mean(name="feat_loss")
        self.vgg_loss_tracker = tf.keras.metrics.Mean(name="vgg_loss")
        self.kl_loss_tracker = tf.keras.metrics.Mean(name="kl_loss")
        self.kid = KID(image_size)

    @property
    def metrics(self):
        return [
            self.disc_loss_tracker,
            self.gen_loss_tracker,
            self.feat_loss_tracker,
            self.vgg_loss_tracker,
            self.kl_loss_tracker,
            self.kid,
        ]

    def build_combined_generator(self):
        # This method builds a model that takes as inputs the following:
        # latent vector, one-hot encoded segmentation label map, and
        # a segmentation map. It then (i) generates an image with the generator,
        # (ii) passes the generated images and segmentation map to the discriminator.
        # Finally, the model produces the following outputs: (a) discriminator outputs,
        # (b) generated image.
        # We will be using this model to simplify the implementation.
        self.discriminator.trainable = False
        mask_input = Input(shape=self.mask_shape, name="mask")
        image_input = Input(shape=self.image_shape, name="image")
        latent_input = Input(shape=(self.latent_dim), name="latent")
        generated_image = self.generator([latent_input, mask_input])
        discriminator_output = self.discriminator([image_input, generated_image])
        patch_size = discriminator_output[-1].shape[1]
        combined_model = Model(
            [latent_input, mask_input, image_input],
            [discriminator_output, generated_image],
        )
        return patch_size, combined_model

    def compile(self, gen_lr: float = 1e-4, disc_lr: float = 4e-4, **kwargs):
        super().compile(**kwargs)
        self.generator_optimizer = optimizers.Adam(gen_lr, beta_1=0.0, beta_2=0.999)
        self.discriminator_optimizer = optimizers.Adam(
            disc_lr, beta_1=0.0, beta_2=0.999
        )
        self.discriminator_loss = DiscriminatorLoss()
        self.feature_matching_loss = FeatureMatchingLoss()
        self.vgg_loss = VGGFeatureMatchingLoss()

    def train_discriminator(self, latent_vector, segmentation_map, real_image, labels):
        fake_images = self.generator([latent_vector, labels])
        with tf.GradientTape() as gradient_tape:
            pred_fake = self.discriminator([segmentation_map, fake_images])[-1]
            pred_real = self.discriminator([segmentation_map, real_image])[-1]
            loss_fake = self.discriminator_loss(pred_fake, False)
            loss_real = self.discriminator_loss(pred_real, True)
            total_loss = 0.5 * (loss_fake + loss_real)

        self.discriminator.trainable = True
        gradients = gradient_tape.gradient(
            total_loss, self.discriminator.trainable_variables
        )
        self.discriminator_optimizer.apply_gradients(
            zip(gradients, self.discriminator.trainable_variables)
        )
        return total_loss

    def train_generator(
        self, latent_vector, segmentation_map, labels, image, mean, variance
    ):
        # Generator learns through the signal provided by the discriminator. During
        # backpropagation, we only update the generator parameters.
        self.discriminator.trainable = False
        with tf.GradientTape() as tape:
            real_d_output = self.discriminator([segmentation_map, image])
            fake_d_output, fake_image = self.combined_model(
                [latent_vector, labels, segmentation_map]
            )
            pred = fake_d_output[-1]

            # Compute generator losses.
            g_loss = generator_loss(pred)
            kl_loss = self.kl_divergence_loss_coeff * kl_divergence_loss(mean, variance)
            vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(image, fake_image)
            feature_loss = self.feature_loss_coeff * self.feature_matching_loss(
                real_d_output, fake_d_output
            )
            total_loss = g_loss + kl_loss + vgg_loss + feature_loss

        all_trainable_variables = (
            self.combined_model.trainable_variables + self.encoder.trainable_variables
        )

        gradients = tape.gradient(total_loss, all_trainable_variables,)
        self.generator_optimizer.apply_gradients(
            zip(gradients, all_trainable_variables,)
        )
        return total_loss, feature_loss, vgg_loss, kl_loss

    def train_step(self, data):
        segmentation_map, image, labels = data
        mean, variance = self.encoder(image)
        latent_vector = self.sampler([mean, variance])
        discriminator_loss = self.train_discriminator(
            latent_vector, segmentation_map, image, labels
        )
        (generator_loss, feature_loss, vgg_loss, kl_loss) = self.train_generator(
            latent_vector, segmentation_map, labels, image, mean, variance
        )

        # Report progress.
        self.disc_loss_tracker.update_state(discriminator_loss)
        self.gen_loss_tracker.update_state(generator_loss)
        self.feat_loss_tracker.update_state(feature_loss)
        self.vgg_loss_tracker.update_state(vgg_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        results = {m.name: m.result() for m in self.metrics[:-1]}
        return results

    def test_step(self, data):
        segmentation_map, image, labels = data
        # Obtain the learned moments of the real image distribution.
        mean, variance = self.encoder(image)

        # Sample a latent from the distribution defined by the learned moments.
        latent_vector = self.sampler([mean, variance])

        # Generate the fake images,
        fake_images = self.generator([latent_vector, labels])

        # Calculate the losses.
        pred_fake = self.discriminator([segmentation_map, fake_images])[-1]
        pred_real = self.discriminator([segmentation_map, image])[-1]
        loss_fake = self.discriminator_loss(pred_fake, False)
        loss_real = self.discriminator_loss(pred_real, True)
        total_discriminator_loss = 0.5 * (loss_fake + loss_real)
        real_d_output = self.discriminator([segmentation_map, image])
        fake_d_output, fake_image = self.combined_model(
            [latent_vector, labels, segmentation_map]
        )
        pred = fake_d_output[-1]
        g_loss = generator_loss(pred)
        kl_loss = self.kl_divergence_loss_coeff * kl_divergence_loss(mean, variance)
        vgg_loss = self.vgg_feature_loss_coeff * self.vgg_loss(image, fake_image)
        feature_loss = self.feature_loss_coeff * self.feature_matching_loss(
            real_d_output, fake_d_output
        )
        total_generator_loss = g_loss + kl_loss + vgg_loss + feature_loss

        # Report progress.
        self.kid.update_state(image, fake_images)
        self.disc_loss_tracker.update_state(total_discriminator_loss)
        self.gen_loss_tracker.update_state(total_generator_loss)
        self.feat_loss_tracker.update_state(feature_loss)
        self.vgg_loss_tracker.update_state(vgg_loss)
        self.kl_loss_tracker.update_state(kl_loss)
        results = {m.name: m.result() for m in self.metrics}
        return results

    def call(self, inputs):
        latent_vectors, labels = inputs
        return self.generator([latent_vectors, labels])

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
            os.path.join(filepath, "generator"),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )
        self.discriminator.save(
            os.path.join(filepath, "discriminator"),
            overwrite=overwrite,
            include_optimizer=include_optimizer,
            save_format=save_format,
            signatures=signatures,
            options=options,
            save_traces=save_traces,
        )

    def load(self, generator_filepath: str, discriminator_filepath: str):
        self.generator = models.load_model(generator_filepath)
        self.discriminator = models.load_model(discriminator_filepath)

    def save_weights(self, filepath, overwrite=True, save_format=None, options=None):
        self.generator.save_weights(
            os.path.join(filepath, "generator-checkpoints"),
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )
        self.discriminator.save_weights(
            os.path.join(filepath, "discriminator-checkpoints"),
            overwrite=overwrite,
            save_format=save_format,
            options=options,
        )

    def load_weights(self, filepath, by_name=False, skip_mismatch=False, options=None):
        self.generator.load_weights(
            os.path.join(filepath, "generator-checkpoints"),
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )
        self.discriminator.load_weights(
            os.path.join(filepath, "discriminator-checkpoints"),
            by_name=by_name,
            skip_mismatch=skip_mismatch,
            options=options,
        )
