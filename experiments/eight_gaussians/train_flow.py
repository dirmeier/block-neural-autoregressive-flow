import os

import jax
import numpy as np
from absl import logging
from checkpointer import get_checkpointer_fns, new_train_state
from jax import numpy as jnp
from jax import random as jr
from tensorflow_probability.substrates.jax import distributions as tfd

from bnaf import get_bnaf_model


@jax.jit
def step_fn(step_key, state, batch):
    base_distribution = tfd.Independent(
        tfd.Normal(jnp.zeros(2), 0.25), reinterpreted_batch_ndims=1
    )

    def loss_fn(params, rng):
        z, logdet = state.apply_fn(
            variables={"params": params},
            inputs=batch["inputs"],
            condition=batch["condition"],
        )
        lp = base_distribution.log_prob(z)
        ll = lp + logdet
        return -jnp.mean(ll)

    loss, grads = jax.value_and_grad(loss_fn)(state.params, step_key)
    new_state = state.apply_gradients(grads=grads)
    return loss, new_state


def train_flow(rng_key, workdir, data_loader, n_iter=200_000):
    state_key, rng_key = jr.split(rng_key)

    model = get_bnaf_model()
    ckpt_save_fn, _ = get_checkpointer_fns(
        os.path.join(workdir, "checkpoints", "forward")
    )
    state = new_train_state(state_key, model, next(iter(data_loader)))
    step_key, rng_key = jr.split(rng_key)

    losses = np.zeros(n_iter)
    best_state, best_loss = state, np.inf
    for step in range(1, n_iter + 1):
        train_key, sample_key = jr.split(jr.fold_in(step_key, step))
        loss, state = step_fn(train_key, state, data_loader(256))
        losses[step - 1] = float(loss)
        if step == 1 or step % 10_000 == 0:
            logging.info(f"loss at epoch {step}: {loss}")
        if losses[step - 1] < best_loss:
            best_state = state
            best_loss = losses[step - 1]
    ckpt_save_fn(best_state)
