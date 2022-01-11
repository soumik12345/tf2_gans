import os
import matplotlib.pyplot as plt
import wandb
import logging

import tensorflow as tf
from tensorflow.keras import callbacks


class GanMonitor(callbacks.Callback):
    def __init__(
        self,
        val_dataset: tf.data.Dataset,
        n_samples: int,
        epoch_interval: int,
        use_wandb: bool,
        plot_save_dir: str,
    ):
        self.val_images = next(iter(val_dataset))
        self.n_samples = n_samples
        self.epoch_interval = epoch_interval
        self.plot_save_dir = plot_save_dir
        self.use_wandb = use_wandb
        self.wandb_table = wandb.Table(
            columns=["Epoch", "#", "Semantic Mask", "Ground Truth", "Generated Image"]
        )

        if self.plot_save_dir:
            logging.info(f"Intermediate images will be serialized to: {plot_save_dir}.")

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

            for i in range(4):
                for j in range(3):
                    # ax = axarr if grid_row == 1 else axarr[row]
                    if j == 0:
                        axarr[i, j].imshow((self.val_images[0][i] + 1) / 2)
                        axarr[i, j].axis("off")
                        axarr[i, j].set_title("Mask", fontsize=20)
                    if j == 1:
                        axarr[i, j].imshow((self.val_images[1][i] + 1) / 2)
                        axarr[i, j].axis("off")
                        axarr[i, j].set_title("Reference Image", fontsize=20)
                    if j == 2:
                        axarr[i, j].imshow((generated_images[i] + 1) / 2)
                        axarr[i, j].axis("off")
                        axarr[i, j].set_title("Generated Image", fontsize=20)
                    self.wandb_table.add_data(
                        epoch,
                        j,
                        wandb.Image((self.val_images[0][i] + 1) / 2),
                        wandb.Image((self.val_images[1][i] + 1) / 2),
                        wandb.Image((generated_images[i] + 1) / 2),
                    )

            if (self.plot_save_dir is None) and (not self.use_wandb):
                plt.show()
            elif self.use_wandb:
                wandb.log({f"validation_images_{epoch}": fig})
                wandb.log({"GANMonitor": self.wandb_table})
            elif self.plot_save_dir:
                fig.savefig(os.path.join(self.plot_save_dir, f"{epoch}.png"))
