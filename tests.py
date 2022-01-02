import unittest
import tensorflow as tf

from gaugan.dataloader import FacadesDataLoader
from gaugan.models import build_encoder, build_generator, build_discriminator, GauGAN


class FacadesDataLoaderTester(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.data_loader = FacadesDataLoader()

    def test_datasets(self) -> None:
        train_dataset, val_dataset = self.data_loader.get_datasets(
            batch_size=1, split_fraction=0.2
        )
        segmentation_map, image, labels = next(iter(train_dataset))
        assert segmentation_map.shape == (1, 256, 256, 3)
        assert image.shape == (1, 256, 256, 3)
        assert labels.shape == (1, 256, 256, 12)
        segmentation_map, image, labels = next(iter(val_dataset))
        assert segmentation_map.shape == (1, 256, 256, 3)
        assert image.shape == (1, 256, 256, 3)
        assert labels.shape == (1, 256, 256, 12)


class ModelTester(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)

    def test_encoder(self):
        encoder = build_encoder(
            image_shape=(256, 256, 3),
            encoder_downsample_factor=64,
            latent_dim=256,
            alpha=0.2,
            dropout=0.5,
        )
        mean, logvar = encoder(tf.zeros((1, 256, 256, 3), dtype=tf.float32))
        assert mean.shape == (1, 256)
        assert logvar.shape == (1, 256)

    def test_generator(self):
        generator = build_generator(mask_shape=(256, 256, 12), latent_dim=256)
        output_image = generator(
            [
                tf.zeros((1, 256), dtype=tf.float32),
                tf.zeros((1, 256, 256, 12), dtype=tf.float32),
            ]
        )
        assert output_image.shape == (1, 256, 256, 3)

    def test_discriminator(self):
        discriminator = build_discriminator(
            image_shape=(256, 256, 3), downsample_factor=64, alpha=0.2, dropout=0.5
        )
        x1, x2, x3, x4, x5 = discriminator(
            [
                tf.zeros((1, 256, 256, 3), dtype=tf.float32),
                tf.zeros((1, 256, 256, 3), dtype=tf.float32),
            ]
        )
        assert x1.shape == (1, 128, 128, 64)
        assert x2.shape == (1, 64, 64, 128)
        assert x3.shape == (1, 32, 32, 256)
        assert x4.shape == (1, 32, 32, 512)
        assert x5.shape == (1, 29, 29, 1)


class GauGANTester(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)

    def test_combined_model(self):
        gaugan = GauGAN(
            image_size=256,
            num_classes=12,
            batch_size=4,
            latent_dim=256,
            feature_loss_coeff=10,
            vgg_feature_loss_coeff=0.1,
            kl_divergence_loss_coeff=0.1,
            encoder_downsample_factor=64,
            discriminator_downsample_factor=64,
            alpha=0.2,
            dropout=0.5
        )
        (x1, x2, x3, x4, x5), _ = gaugan.combined_model(
            [
                tf.zeros((1, 256), dtype=tf.float32),
                tf.zeros((1, 256, 256, 12), dtype=tf.float32),
                tf.zeros((1, 256, 256, 3), dtype=tf.float32),
            ]
        )
        assert x1.shape == (1, 128, 128, 64)
        assert x2.shape == (1, 64, 64, 128)
        assert x3.shape == (1, 32, 32, 256)
        assert x4.shape == (1, 32, 32, 512)
        assert x5.shape == (1, 29, 29, 1)
        assert gaugan.patch_size == 29
