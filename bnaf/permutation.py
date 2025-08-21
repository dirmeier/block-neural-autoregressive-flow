from typing import Any

from flax import linen as nn
from jax import numpy as jnp


class Permutation(nn.Module):
  permutation: Any

  def __call__(self, inputs, **kwargs):
    return self.forward_and_log_det(inputs, **kwargs)

  def forward_and_log_det(self, inputs, *args, **kwargs):
    return inputs[..., self.permutation], jnp.full(jnp.shape(inputs)[:-1], 0.0)
