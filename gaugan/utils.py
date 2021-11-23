import matplotlib.pyplot as plt
from tensorflow import keras
import os


def plot_results(images, titles, save_figure=False, figure_size=(12, 12)):
    """Plot results in a row.

    Args:
        images: List of images (PIL or numpy arrays).
        titles: List of titles corresponding to images.
        save_figure: If we are serializing the figure.
        figure_size: Size of figure.
    """
    fig = plt.figure(figsize=figure_size)
    for i in range(len(images)):
        if titles:
            fig.add_subplot(1, len(images), i + 1).set_title(titles[i])
        else:
            fig.add_subplot(1, len(images), i + 1)
        _ = plt.imshow(images[i])
        plt.axis("off")
    if save_figure:
        return plt
    else:
        plt.show()


class GANMonitor(keras.callbacks.Callback):
    """Monitors the progress of Generator during training."""

    def __init__(self, segmentation_maps, labels, root_dir, epoch_interval=5):
        """Initialize the GANMonitor.

        Args:
            segmentation_maps: NumPy array of segmentation maps (can be tf.Tensor too).
            labels: Numpy array of one-hot encoded labels corresponding to segmentation maps (can be tf.Tensor too).
            root_dir: Root directory where to save the generated images.
            epoch_interval: Interval between epochs to plot.
        """
        self.segmentation_maps = segmentation_maps
        self.labels = labels
        self.root_dir = root_dir
        self.epoch_interval = epoch_interval

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_interval == 0:
            generated_images = self.model.predict((self.segmentation_maps, self.labels))
            figure_object = plot_results(generated_images, None, save_figure=True)
            figure_path = os.path.join(
                self.root_dir,
                "intermediate_images",
                f"generated_images_epoch_{epoch}.png",
            )
            figure_object.savefig(figure_path, bbox_inches="tight", dpi=300)
