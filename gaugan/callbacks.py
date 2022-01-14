import os
import matplotlib.pyplot as plt
import wandb
import logging

import tensorflow as tf
from tensorflow.keras import callbacks

from .utils import plot_results


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
        self.columns = ["Semantic Mask", "Ground Truth", "Generated Image"]

        if self.plot_save_dir:
            logging.info(f"Intermediate images will be serialized to: {plot_save_dir}.")

    def infer(self):
        latent_vector = tf.random.normal(
            shape=(self.model.batch_size, self.model.latent_dim), mean=0.0, stddev=2.0
        )
        return self.model.predict([latent_vector, self.val_images[2]])

    def log_to_tables(self, epoch, semantic_input, ground_truth, generated_images):
        wandb_table = wandb.Table(columns=self.columns)
        for i in range(4):
            wandb_table.add_data(
                wandb.Image(semantic_input[i]),
                wandb.Image(ground_truth[i]),
                wandb.Image(generated_images[i]),
            )
        wandb.log({f"GANMonitor Epoch {epoch}": wandb_table})

    def on_epoch_end(self, epoch, logs=None):
        if epoch == 0 or (epoch + 1) % self.epoch_interval == 0:
            generated_images = self.infer()

            semantic_input = (self.val_images[0] + 1) / 2
            ground_truth = (self.val_images[1] + 1) / 2
            generated_images = (generated_images + 1) / 2

            if self.use_wandb:
                self.log_to_tables(
                    epoch + 1, semantic_input, ground_truth, generated_images
                )

            for i in range(4):
                fig = plot_results(
                    [semantic_input[i], ground_truth[i], generated_images[i]],
                    self.columns,
                    figure_size=(18, 18),
                )
                if (self.plot_save_dir is None) and (not self.use_wandb):
                    plt.show()
                elif self.plot_save_dir:
                    fig.savefig(os.path.join(self.plot_save_dir, f"{epoch}_{i}.png"))


class CheckpointArtifactCallback(callbacks.Callback):
    def __init__(self, experiment_name: str, model_name: str, wandb_run):
        super().__init__()
        self.model_name = model_name
        self.wandb_run = wandb_run
        self.experiment_name = experiment_name

    def on_epoch_end(self, epoch, logs=None):
        self.model.save(self.model_name)
        artifact = wandb.Artifact(name=self.experiment_name, type="model")
        artifact.add_dir(self.model_name)
        self.wandb_run.log_artifact(artifact)
