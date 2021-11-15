import unittest
from gaugan import PairedTranslationDataLoader


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
