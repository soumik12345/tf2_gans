import numpy as np
import tensorflow as tf
from scipy.io import loadmat
from typing import List


_AUTO = tf.data.AUTOTUNE


class CocoStuff10KDataLoader:
    def __init__(
        self,
        target_image_height: int = 256,
        target_image_width: int = 256,
        num_classes: int = 172,
        data_dir: str = "cocostuff10k_data",
    ) -> None:
        super().__init__()
        self.target_image_height = target_image_height
        self.target_image_width = target_image_width
        self.num_classes = num_classes
        self.data_dir = data_dir

    def get_dataset_paths(self, filenames: np.ndarray):
        image_filepaths = [
            f"{self.data_dir}/images/{filename}.jpg" for filename in filenames
        ]
        annotation_filepaths = [
            f"{self.data_dir}/annotations/{filename}.mat" for filename in filenames
        ]
        return image_filepaths, annotation_filepaths

    def get_datasets(self, batch_size: int, val_split_fraction: float):
        train_names = np.loadtxt(f"{self.data_dir}/imageLists/train.txt", dtype=str)
        test_names = np.loadtxt(f"{self.data_dir}/imageLists/test.txt", dtype=str)

        train_image_paths, train_anno_paths = self.get_dataset_paths(train_names)
        split_index = int(len(train_image_paths) * (1 - val_split_fraction))
        new_train_image_paths = train_image_paths[:split_index]
        new_train_anno_paths = train_anno_paths[:split_index]

        val_image_paths = train_image_paths[split_index:]
        val_anno_paths = train_anno_paths[split_index:]

        test_image_paths, test_anno_paths = self.get_dataset_paths(test_names)

        training_set = self.prepare_dataset(
            new_train_image_paths, new_train_anno_paths, batch_size
        )
        validation_set = self.prepare_dataset(
            val_image_paths, val_anno_paths, batch_size, train=False
        )
        test_set = self.prepare_dataset(
            test_image_paths, test_anno_paths, batch_size, train=False
        )
        return training_set, validation_set, test_set

    def load_image(self, image_path: str):
        image = tf.io.read_file(image_path)
        image = tf.image.decode_jpeg(image, channels=3)
        image = tf.cast(image, tf.float32) / 127.5 - 1
        return image

    def load_annotation(self, annotation_path: str):
        segmentation_map = loadmat(annotation_path)["S"]
        segmentation_map = segmentation_map.astype("float32") / 127.5 - 1
        return segmentation_map

    def random_crop(self, segmentation_map: tf.Tensor, image: tf.Tensor):
        crop_size = tf.convert_to_tensor(
            (self.target_image_height, self.target_image_width)
        )

        image_shape = tf.shape(image)[:2]
        margins = image_shape - crop_size

        y1 = tf.random.uniform(shape=(), maxval=margins[0], dtype=tf.int32)
        x1 = tf.random.uniform(shape=(), maxval=margins[1], dtype=tf.int32)
        y2 = y1 + crop_size[0]
        x2 = x1 + crop_size[1]

        labels = tf.identity(segmentation_map)
        labels = ((labels + 1) / 2) * 255.0
        labels = tf.cast(labels, tf.uint8)

        cropped_images = []
        images = [segmentation_map, image, labels]
        for img in images:
            cropped_images.append(img[y1:y2, x1:x2])
        return cropped_images

    def standard_resize(self, segmentation_map: tf.Tensor, image: tf.Tensor):
        segmentation_map.set_shape([None, None])
        image.set_shape([None, None, 3])

        segmentation_map = tf.image.resize(
            segmentation_map[..., None],
            (self.target_image_height, self.target_image_width),
        )
        image = tf.image.resize(
            image, self.target_image_height, self.target_image_width
        )

        labels = tf.identity(segmentation_map)
        labels = ((labels + 1) / 2) * 255.0
        labels = tf.cast(labels, tf.uint8)

        return tf.squeeze(segmentation_map), image, tf.squeeze(labels)

    def one_hot_encoding(
        self, segmentation_map: tf.Tensor, image: tf.Tensor, labels: tf.Tensor
    ):
        segmentation_map_ohe = tf.one_hot(labels, self.num_classes)
        return segmentation_map, image, segmentation_map_ohe

    def prepare_dataset(
        self,
        image_paths: List[str],
        annotation_paths: List[str],
        batch_size: int,
        train=True,
    ):
        image_ds = tf.data.Dataset.from_tensor_slices(image_paths).map(
            self.load_image, num_parallel_calls=_AUTO
        )
        annotation_ds = tf.data.Dataset.from_tensor_slices(annotation_paths)
        annotation_ds = annotation_ds.map(
            lambda x: tf.numpy_function(self.load_annotation, [x], tf.float32),
            num_parallel_calls=_AUTO,
        ).cache()

        dataset = tf.data.Dataset.zip((annotation_ds, image_ds))
        dataset = dataset.shuffle(batch_size * 10) if train else dataset
        map_fn = self.random_crop if train else self.standard_resize
        dataset = dataset.map(map_fn, num_parallel_calls=_AUTO)
        dataset = dataset.map(self.one_hot_encoding, num_parallel_calls=_AUTO)

        return dataset.batch(batch_size, drop_remainder=True).prefetch(_AUTO)
