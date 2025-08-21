from collections.abc import Callable

from flax import linen as nn


class Chain(nn.Module):
  transforms: list[Callable]

  def __call__(self, inputs, **kwargs):
    return self.forward_and_log_det(inputs, **kwargs)

  def forward_and_log_det(self, inputs, condition=None, **kwargs):
    hidden, log_det = self.transforms[0](inputs, condition=condition)
    for transform in self.transforms[1:]:
      hidden, new_log_det = transform(hidden, condition=condition)
      log_det += new_log_det
    return hidden, log_det
