import tensorflow as tf
from tensorflow.keras import callbacks

from .utils import plot_results


class GanMonitor(callbacks.Callback):
    def __init__(self, val_dataset, n_samples, epoch_interval=5):
        self.val_images = next(iter(val_dataset))
        self.n_samples = n_samples
        self.epoch_interval = epoch_interval

    def infer(self):
        latent_vector = tf.random.normal(
            shape=(self.model.batch_size, self.model.latent_dim), mean=0.0, stddev=2.0
        )
        return self.model.predict([latent_vector, self.val_images[2]])

    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.epoch_interval == 0:
            generated_images = self.infer()
            for index in range(self.n_samples):
                plot_results(
                    [
                        self.val_images[index][0],
                        self.val_images[index][1],
                        generated_images[index],
                    ],
                    ["Segmentation Map", "Ground Truth", "Generated Image"],
                    figure_size=(18, 18),
                )
