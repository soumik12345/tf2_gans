import os
import matplotlib.pyplot as plt
import wandb

import tensorflow as tf
from tensorflow.keras import callbacks


class GanMonitor(callbacks.Callback):
    def __init__(
        self,
        val_dataset: tf.data.Dataset,
        n_samples: int,
        epoch_interval: int,
        use_wandb: bool,
        plot_save_dir,
    ):
        self.val_images = next(iter(val_dataset))
        self.n_samples = n_samples
        self.epoch_interval = epoch_interval
        self.use_wandb = use_wandb
        self.plot_save_dir = plot_save_dir

    def infer(self):
        latent_vector = tf.random.normal(
            shape=(self.model.batch_size, self.model.latent_dim), mean=0.0, stddev=2.0
        )
        return self.model.predict([latent_vector, self.val_images[2]])

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % self.epoch_interval == 0:
            generated_images = self.infer()

            grid_row = min(generated_images.shape[0], 3)
            fig, axarr = plt.subplots(4, 3, figsize=(18, grid_row * 6))

            for row in range(16):
                ax = axarr if grid_row == 1 else axarr[row]
                ax[0].imshow((self.val_images[0][row] + 1) / 2)
                ax[0].axis("off")
                ax[0].set_title("Mask", fontsize=20)
                ax[1].imshow((self.val_images[1][row] + 1) / 2)
                ax[1].axis("off")
                ax[1].set_title("Reference Image", fontsize=20)
                ax[2].imshow((generated_images[row] + 1) / 2)
                ax[2].axis("off")
                ax[2].set_title("Generated Image", fontsize=20)

            if (self.plot_save_dir is None) and (not self.use_wandb):
                plt.show()
            elif self.use_wandb:
                wandb.log({f"validation_images_{epoch}": fig})
            else:
                plt.savefig(os.path.join(self.plot_save_dir, f"{epoch}.png"))
