"""
For now, this is just a driver script. In future, it will be evolved to a training script as
`train_facades.py`.
"""

from gaugan.dataloader import CocoStuff10KDataLoader
from configs import cocostuff10k


cocostuff10k_config = cocostuff10k.get_config()

data_loader = CocoStuff10KDataLoader(
    target_image_height=cocostuff10k_config.image_height,
    target_image_width=cocostuff10k_config.image_width,
    num_classes=cocostuff10k_config.num_classes,
    data_dir=cocostuff10k_config.dataset_dir,
)
train_ds, validation_ds, test_ds = data_loader.get_datasets(
    batch_size=cocostuff10k_config.batch_size,
    val_split_fraction=cocostuff10k_config.split_fraction,
)
print(train_ds.element_spec)
print(validation_ds.element_spec)
print(test_ds.element_spec)


sample_train_batch = next(iter(train_ds))
sample_val_batch = next(iter(validation_ds))
sample_test_batch = next(iter(validation_ds))
