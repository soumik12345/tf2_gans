from typing import List
import tensorflow as tf
from abc import ABC, abstractmethod


_AUTOTUNE = tf.data.AUTOTUNE


class PairedTranslationDataLoader(ABC):
    def __init__(
        self, target_image_height: int, target_image_width: int, num_classes: int
    ) -> None:
        self.target_image_height = target_image_height
        self.target_image_width = target_image_width
        self.num_classes = num_classes

    def _random_crop(self, image, segmentation_map, labels):
        crop_size = tf.convert_to_tensor(
            (self.target_image_height, self.target_image_width)
        )
        image_shape = tf.shape(image)[:2]
        margins = image_shape - crop_size
        y1 = tf.random.uniform(shape=(), maxval=margins[0], dtype=tf.int32)
        x1 = tf.random.uniform(shape=(), maxval=margins[1], dtype=tf.int32)
        y2 = y1 + crop_size[0]
        x2 = x1 + crop_size[1]
        cropped_images = []
        images = [image, segmentation_map, labels]
        for img in images:
            cropped_images.append(img[y1:y2, x1:x2])
        return cropped_images

    def _load_data(self, image_file, segmentation_map_file, label_file):
        image = tf.image.decode_png(tf.io.read_file(image_file), channels=3)
        segmentation_map = tf.image.decode_png(
            tf.io.read_file(segmentation_map_file), channels=3
        )
        labels = tf.image.decode_bmp(tf.io.read_file(label_file), channels=0)
        labels = tf.squeeze(labels)
        image = tf.cast(image, tf.float32) / 127.5 - 1
        segmentation_map = tf.cast(segmentation_map, tf.float32) / 127.5 - 1
        return segmentation_map, image, labels

    def _configure_dataset(
        self,
        image_files: List[str],
        segmentation_map_files: List[str],
        label_files: List[str],
        batch_size: int,
        is_train: bool,
    ):
        dataset = tf.data.Dataset.from_tensor_slices(
            (image_files, segmentation_map_files, label_files)
        )
        dataset = dataset.shuffle(batch_size * 10) if is_train else dataset
        dataset = dataset.map(self._load_data, num_parallel_calls=_AUTOTUNE)
        dataset = dataset.map(self._random_crop, num_parallel_calls=_AUTOTUNE)
        dataset = dataset.map(
            lambda x, y, z: (x, y, tf.one_hot(z, self.num_classes)),
            num_parallel_calls=_AUTOTUNE,
        )
        return dataset.batch(batch_size, drop_remainder=True)

    @abstractmethod
    def download_dataset(self):
        pass

    @abstractmethod
    def get_datasets(self, batch_size: int, split_fraction: float):
        pass
