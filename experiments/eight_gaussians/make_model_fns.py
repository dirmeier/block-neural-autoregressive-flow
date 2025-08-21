import os

from jax import jit

from bnaf import get_bnaf_model, get_inverter_model
from experiments.eight_gaussians.checkpointer import get_checkpointer_fns


def get_forward_fn(workdir):
  model = get_bnaf_model()
  _, restore_fn = get_checkpointer_fns(
    os.path.join(workdir, "checkpoints", "forward")
  )
  params = restore_fn()["params"]

  def fn(inputs, condition):
    z, _ = model.apply({"params": params}, inputs=inputs, condition=condition)
    return z

  return jit(fn)


def get_inverse_fn(workdir):
  model = get_inverter_model()
  _, restore_fn = get_checkpointer_fns(
    os.path.join(workdir, "checkpoints", "inverse")
  )
  params = restore_fn()["params"]

  def fn(z, condition):
    y = model.apply({"params": params}, z, condition)
    return y

  return jit(fn)
