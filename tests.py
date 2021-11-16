import unittest
import tensorflow as tf

from gaugan import PairedTranslationDataLoader, models


class PairedTranslationDataLoaderTester(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)

    def test_facades(self):
        data_loader = PairedTranslationDataLoader()
        train_dataset, val_dataset = data_loader.get_datasets(dataset_path="./facades/")
        image, segmentation_map, one_hot_encoded_labels = next(iter(train_dataset))
        assert image.shape == (16, 256, 256, 3)
        assert segmentation_map.shape == (16, 256, 256, 3)
        assert one_hot_encoded_labels.shape == (16, 256, 256, 12)
        image, segmentation_map, one_hot_encoded_labels = next(iter(val_dataset))
        assert image.shape == (16, 256, 256, 3)
        assert segmentation_map.shape == (16, 256, 256, 3)
        assert one_hot_encoded_labels.shape == (16, 256, 256, 12)


class ModelTester(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)

    def test_encoder(self):
        encoder = models.build_encoder()
        mean, logvar = encoder(tf.zeros((1, 256, 256, 3), dtype=tf.float32))
        assert mean.shape == (1, 256)
        assert logvar.shape == (1, 256)

    def test_generator(self):
        generator = models.build_generator()
        output_image = generator(
            [
                tf.zeros((1, 256), dtype=tf.float32),
                tf.zeros((1, 256, 256, 12), dtype=tf.float32),
            ]
        )
        assert output_image.shape == (1, 256, 256, 3)

    def test_discriminator(self):
        discriminator = models.build_discriminator()
        x1, x2, x3, x4, x5 = discriminator(
            [
                tf.zeros((1, 256, 256, 3), dtype=tf.float32),
                tf.zeros((1, 256, 256, 3), dtype=tf.float32),
            ]
        )
        assert x1.shape == (1, 128, 128, 64)
        assert x2.shape == (1, 64, 64, 128)
        assert x3.shape == (1, 32, 32, 256)
        assert x4.shape == (1, 16, 16, 512)
        assert x5.shape == (1, 13, 13, 1)
