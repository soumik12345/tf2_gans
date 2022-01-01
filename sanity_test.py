import tensorflow as tf
import matplotlib.pyplot as plt

from gaugan.dataloader import FacadesDataLoader
from gaugan.models import GauGAN


IMAGE_SIZE = 256
NUM_CLASSES = 12
LATENT_DIMENTION = 256

data_loader = FacadesDataLoader(
    target_image_height=IMAGE_SIZE,
    target_image_width=IMAGE_SIZE,
    num_classes=NUM_CLASSES,
)
train_dataset, val_dataset = data_loader.get_datasets(batch_size=1, split_fraction=0.2)

sample_train_batch = next(iter(train_dataset))
print(f"Segmentation map batch shape: {sample_train_batch[0].shape}.")
print(f"Image batch shape: {sample_train_batch[1].shape}.")
print(f"One-hot encoded label map shape: {sample_train_batch[2].shape}.")

for segmentation_map, real_image in zip(sample_train_batch[0], sample_train_batch[1]):
    fig = plt.figure(figsize=(10, 10))
    fig.add_subplot(1, 2, 1).set_title("Segmentation Map")
    plt.imshow((segmentation_map + 1) / 2)
    fig.add_subplot(1, 2, 2).set_title("Real Image")
    plt.imshow((real_image + 1) / 2)
    plt.show()

gaugan = GauGAN(
    image_size=IMAGE_SIZE,
    num_classes=NUM_CLASSES,
    batch_size=4,
    latent_dim=LATENT_DIMENTION,
)
gaugan.combined_model.summary()
print(gaugan.patch_size)
