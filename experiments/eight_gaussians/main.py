import jax
import numpy as np
import seaborn as sns
from absl import app, flags, logging
from dataloader import get_data_loader
from jax import random as jr
from make_model_fns import (
  get_forward_fn,
  get_inverse_fn,
)
from matplotlib import pyplot as plt

from experiments.eight_gaussians.train_flow import train_flow
from experiments.eight_gaussians.train_mlp import train_mlp

FLAGS = flags.FLAGS
flags.DEFINE_string("workdir", None, "work directory")
flags.mark_flags_as_required(["workdir"])


def sample(rng_key, workdir, data_loader):
  forward_fn = get_forward_fn(workdir)
  invert_fn = get_inverse_fn(workdir)

  data = data_loader(1_000)
  y, labels = data["inputs"], data["condition"]
  z = np.array(forward_fn(y, labels))
  y_hat = np.array(invert_fn(z, labels))

  labels = np.squeeze(labels)
  fig, axes = plt.subplots(figsize=(12, 4), ncols=2)
  colors = sns.color_palette("magma", as_cmap=False, n_colors=8).as_hex()
  colors = np.array(colors)[labels]
  axes[0].scatter(
    z[:, 0], z[:, 1], marker="H", color=colors, alpha=0.25, label="pullback"
  )
  axes[0].scatter(
    y[:, 0], y[:, 1], marker="1", color=colors, alpha=0.25, label="data"
  )
  axes[1].scatter(
    z[:, 0], z[:, 1], marker="H", color=colors, alpha=0.25, label="pullback"
  )
  axes[1].scatter(
    y_hat[:, 0],
    y_hat[:, 1],
    marker="1",
    color=colors,
    alpha=0.25,
    label="pushforward",
  )
  for ax in axes:
    ax.set_xlim(-1.5, 1.5)
    ax.set_ylim(-1.5, 1.5)
    ax.legend()
    ax.spines[["right", "top", "left", "bottom"]].set_visible(False)
    ax.grid(True)
  plt.tight_layout()
  fig.savefig("figures/samples.png")
  plt.show()


def main(argv):
  del argv
  logging.set_verbosity(logging.INFO)
  data_loader = get_data_loader(rng_key=jr.PRNGKey(0))
  train_flow(jr.PRNGKey(1), FLAGS.workdir, data_loader)
  train_mlp(jr.PRNGKey(2), FLAGS.workdir, data_loader)
  sample(jr.PRNGKey(3), FLAGS.workdir, data_loader)


if __name__ == "__main__":
  jax.config.config_with_absl()
  app.run(main)
