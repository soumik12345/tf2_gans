import matplotlib.pyplot as plt
from gaugan.dataloader import FacadesDataLoader


data_loader = FacadesDataLoader(
    target_image_height=256, target_image_width=256, num_classes=12
)
data_loader.download_dataset()
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