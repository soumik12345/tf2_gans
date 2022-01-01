import os
import gdown
from glob import glob
from typing import List, Tuple

from .base import PairedTranslationDataLoader
from .commons import extract_archive


class FacadesDataLoader(PairedTranslationDataLoader):
    def __init__(
        self, target_image_height: int, target_image_width: int, num_classes: int
    ) -> None:
        super().__init__(target_image_height, target_image_width, num_classes)
        self.data_dir = "facades_data"
        self.archive_path = "facades_data.zip"
        self.data_url = (
            "https://drive.google.com/uc?id=1q4FEjQg1YSb4mPx2VdxL7LXKYu3voTMj"
        )

    def download_dataset(self) -> None:
        if not os.path.isdir(self.data_dir):
            if not os.path.isfile(self.archive_path):
                gdown.download(self.data_url)
            extract_archive(self.archive_path, ".")

    def _get_file_list(self, image_files: List[str]) -> Tuple[List[str], List[str]]:
        segmentation_map_files = [
            image_file.replace("images", "segmentation_map").replace("jpg", "png")
            for image_file in image_files
        ]
        label_files = [
            image_file.replace("images", "segmentation_labels").replace("jpg", "bmp")
            for image_file in image_files
        ]
        return segmentation_map_files, label_files

    def get_datasets(self, batch_size: int, split_fraction: float):
        files = glob(os.path.join(self.data_dir, "*.jpg"))
        split_index = int(len(files) * (1 - split_fraction))
        train_image_files = files[:split_index]
        val_image_files = files[split_index:]
        train_segmentation_map_files, train_label_files = self._get_file_list(
            train_image_files
        )
        val_segmentation_map_files, val_label_files = self._get_file_list(
            val_image_files
        )
        train_dataset = self._configure_dataset(
            train_image_files,
            train_segmentation_map_files,
            train_label_files,
            batch_size,
            is_train=True,
        )
        val_dataset = self._configure_dataset(
            val_image_files,
            val_segmentation_map_files,
            val_label_files,
            batch_size,
            is_train=False,
        )
        return train_dataset, val_dataset
