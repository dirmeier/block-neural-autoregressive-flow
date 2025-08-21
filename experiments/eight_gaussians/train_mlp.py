import os

import jax
import numpy as np
from absl import logging
from checkpointer import get_checkpointer_fns, new_train_state
from jax import numpy as jnp
from jax import random as jr

from bnaf import get_inverter_model
from experiments.eight_gaussians.make_model_fns import get_forward_fn


@jax.jit
def step_fn(step_key, state, batch):
  def loss_fn(params, rng):
    y_hat = state.apply_fn(
      variables={"params": params},
      inputs=batch["z"],
      condition=batch["condition"],
    )
    return jnp.mean(jnp.square(y_hat - batch["y"]))

  loss, grads = jax.value_and_grad(loss_fn)(state.params, step_key)
  new_state = state.apply_gradients(grads=grads)
  return loss, new_state


def train_mlp(rng_key, workdir, data_loader, n_iter=200_000):
  state_key, rng_key = jr.split(rng_key)

  forward_fn = get_forward_fn(workdir)
  model = get_inverter_model()
  ckpt_save_fn, _ = get_checkpointer_fns(
    os.path.join(workdir, "checkpoints", "inverse")
  )
  state = new_train_state(state_key, model, next(iter(data_loader)))
  step_key, rng_key = jr.split(rng_key)

  losses = np.zeros(n_iter)
  best_state, best_loss = state, np.inf
  for step in range(1, n_iter + 1):
    train_key, sample_key = jr.split(jr.fold_in(step_key, step))
    data = data_loader(256)
    y, condition = data["inputs"], data["condition"]
    z = forward_fn(y, condition)
    loss, state = step_fn(
      train_key, state, {"z": z, "y": y, "condition": condition}
    )
    losses[step - 1] = float(loss)
    if step == 1 or step % 10_000 == 0:
      logging.info(f"loss at epoch {step}: {loss}")
    if losses[step - 1] < best_loss:
      best_state = state
      best_loss = losses[step - 1]
  ckpt_save_fn(best_state)
