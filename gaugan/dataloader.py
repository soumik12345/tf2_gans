import os
import numpy as np
from PIL import Image
from glob import glob
import tensorflow as tf


_AUTOTUNE = tf.data.AUTOTUNE


class PairedTranslationDataLoader:
    def __init__(self, image_size: int = 256, n_classes: int = 12) -> None:
        self.image_size = image_size
        self.n_classes = n_classes

    def _random_crop(self, image, segmentation_map, labels):
        image_shape = tf.shape(image)[:2]
        width = tf.random.uniform(
            shape=(), maxval=image_shape[1] - self.image_size + 1, dtype=tf.int32
        )
        height = tf.random.uniform(
            shape=(), maxval=image_shape[0] - self.image_size + 1, dtype=tf.int32
        )
        image_cropped = image[
            height : height + self.image_size, width : width + self.image_size
        ]
        segmentation_map_cropped = segmentation_map[
            height : height + self.image_size, width : width + self.image_size
        ]
        labels_cropped = labels[
            height : height + self.image_size, width : width + self.image_size
        ]
        return image_cropped, segmentation_map_cropped, labels_cropped

    def _one_hot_encode(self, image, labels):
        h, w, _ = image.shape
        one_hot_encoded_labels = np.zeros((h, w, self.n_classes), dtype=np.float32)
        for i in range(self.n_classes):
            one_hot_encoded_labels[labels == i, i] = 1
        return one_hot_encoded_labels

    def _load_data(self, image_file, segmentation_map_file, label_file):
        image_file = image_file.numpy().decode("utf-8")
        segmentation_map_file = segmentation_map_file.numpy().decode("utf-8")
        label_file = label_file.numpy().decode("utf-8")
        image = np.array(Image.open(image_file))
        segmentation_map = np.array(Image.open(segmentation_map_file))
        labels = np.array(Image.open(label_file), dtype=np.uint8)
        image = image / 127.5 - 1
        segmentation_map = segmentation_map / 127.5 - 1
        one_hot_encoded_labels = self._one_hot_encode(image, labels)
        return image, segmentation_map, one_hot_encoded_labels

    def _load_data_tf(self, image_file, segmentation_map_file, label_file):
        image = tf.image.decode_png(tf.io.read_file(image_file), channels=3)
        segmentation_map = tf.image.decode_png(tf.io.read_file(segmentation_map_file), channels=3)
        labels = tf.image.decode_bmp(tf.io.read_file(label_file), channels=0)

        image = tf.cast(image, tf.float32) / 127.5 - 1
        segmentation_map = tf.cast(segmentation_map, tf.float32) / 127.5 - 1
        one_hot_encoded_labels = tf.one_hot(labels, self.n_classes)
        return image, segmentation_map, one_hot_encoded_labels

    def _parse_fn(self, image_file, segmentation_map_file, label_file):
        return tf.py_function(
            self._load_data,
            [image_file, segmentation_map_file, label_file],
            [tf.float32, tf.float32, tf.float32],
        )

    def _get_batched_dataset(self, image_files, batch_size):
        segmentation_map_files = [
            image_file.replace("images", "segmentation_map").replace("jpg", "png")
            for image_file in image_files
        ]
        label_files = [
            image_file.replace("images", "segmentation_labels").replace("jpg", "bmp")
            for image_file in image_files
        ]
        dataset = tf.data.Dataset.from_tensor_slices(
            (image_files, segmentation_map_files, label_files)
        )
        dataset = dataset.map(self._load_data_tf, num_parallel_calls=_AUTOTUNE)
        dataset = dataset.map(self._random_crop, num_parallel_calls=_AUTOTUNE)
        dataset = dataset.batch(batch_size, drop_remainder=True)
        return dataset.prefetch(_AUTOTUNE)

    def get_datasets(
        self, dataset_path: str, val_split: float = 0.2, batch_size: int = 16
    ):
        image_files = glob(os.path.join(dataset_path, "images", "*"))
        split_index = int(len(image_files) * (1 - val_split))
        train_image_files = image_files[:split_index]
        val_image_files = image_files[split_index:]
        train_dataset = self._get_batched_dataset(train_image_files, batch_size)
        val_dataset = self._get_batched_dataset(val_image_files, batch_size)
        return train_dataset, val_dataset
