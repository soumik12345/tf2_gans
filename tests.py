import unittest

from gaugan.dataloader import FacadesDataLoader


class FacadesDataLoaderTester(unittest.TestCase):
    def __init__(self, methodName: str = ...) -> None:
        super().__init__(methodName=methodName)
        self.data_loader = FacadesDataLoader(
            target_image_height=256, target_image_width=256, num_classes=12
        )
        self.data_loader.download_dataset()

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
