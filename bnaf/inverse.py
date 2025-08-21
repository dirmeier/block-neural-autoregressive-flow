from collections.abc import Callable, Sequence

from flax import linen as nn


class Inverter(nn.Module):
  hidden_sizes: Sequence[int]
  n_out_dimension: int
  activation: Callable = nn.gelu

  @nn.compact
  def __call__(self, inputs, condition):
    hidden = inputs
    condition = nn.one_hot(condition, 8)
    hidden = hidden + nn.Dense(hidden.shape[-1])(condition)
    for i, hidden_size in enumerate(self.hidden_sizes):
      hidden = nn.Dense(hidden_size)(hidden)
      hidden = self.activation(hidden)
    out = nn.Dense(self.n_out_dimension)(hidden)
    return out


def get_inverter_model(
  n_dimension=2, hidden_sizes=(128, 128, 128, 128, 128), activation=nn.gelu
):
  return Inverter(list(hidden_sizes), n_dimension, activation)
