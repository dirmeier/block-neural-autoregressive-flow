import jax.numpy as jnp
from flax import linen as nn
from jax.scipy.linalg import block_diag
from numpyro.distributions.util import logmatmulexp

from bnaf.activations import tanh
from bnaf.chain import Chain
from bnaf.permutation import Permutation
from bnaf.util import block_triu_mask


def _kernel_init(mask_diag, mask_triu):
  init_fn = nn.initializers.variance_scaling(1.0, "fan_in", "truncated_normal")
  mask_diag = jnp.astype(mask_diag, jnp.bool)
  mask_triu = jnp.astype(mask_triu, jnp.bool)

  def _init(rng_key, shape):
    kernel = init_fn(rng_key, shape)
    ret_kernel = jnp.where(mask_triu, kernel, 0.0)
    ret_kernel = jnp.where(mask_diag, nn.softplus(kernel), ret_kernel)
    return ret_kernel

  return _init


class _BlockMaskedDense(nn.Module):
  """Adapted from
  https://github.com/pyro-ppl/numpyro/blob/f478772b6abee06b7bfd38c11b3f832e01e089f5/numpyro/nn/block_neural_arn.py#L15
  """

  n_dimension: int
  in_features: int
  out_features: int
  use_bias: bool = True

  def setup(self):
    shape = (
      self.in_features * self.n_dimension,
      self.out_features * self.n_dimension,
    )
    mask_diag = block_diag(
      *jnp.ones((self.n_dimension, self.in_features, self.out_features))
    )
    # we do the upper triangular matrix, cause we compute x @ w and not w @ x
    mask_triu = block_triu_mask(
      self.n_dimension, (self.in_features, self.out_features)
    )
    self.kernel = self.param(
      "kernel",
      _kernel_init(mask_diag, mask_triu),
      shape,
    )
    self.scale = self.param("scale", nn.initializers.uniform(), (shape[-1],))
    if self.use_bias:
      self.bias = self.param("bias", nn.initializers.uniform(), (shape[-1],))

  def _normalize(self):
    mask_diag = block_diag(
      *jnp.ones((self.n_dimension, self.in_features, self.out_features))
    )
    # we do the upper triangular matrix, cause we compute x @ w and not w @ x
    mask_triu = block_triu_mask(
      self.n_dimension, (self.in_features, self.out_features)
    )
    w = jnp.exp(self.kernel) * mask_diag + self.kernel * mask_triu
    w_norm = jnp.linalg.norm(w, axis=-2, keepdims=True)
    w = jnp.exp(self.scale) * w / w_norm
    return w, w_norm

  def forward(self, inputs):
    w, w_norm = self._normalize()
    outputs = jnp.dot(inputs, w)
    if self.use_bias:
      outputs = outputs + self.bias
    return outputs

  def forward_and_log_det(self, inputs, logdets):
    w, w_norm = self._normalize()
    outputs = jnp.dot(inputs, w)
    if self.use_bias:
      outputs = outputs + self.bias
    new_logdet = self.forward_log_det(w_norm)
    if logdets is None:
      logdets = jnp.broadcast_to(
        new_logdet, inputs.shape[:-1] + new_logdet.shape
      )
    else:
      logdets = logmatmulexp(logdets, new_logdet)
    return outputs, logdets

  def forward_log_det(self, w_norm=None):
    mask_diag = block_diag(
      *jnp.ones((self.n_dimension, self.in_features, self.out_features))
    )
    if w_norm is None:
      _, w_norm = self._normalize()
    logdet = self.scale + self.kernel - jnp.log(w_norm)
    idxs = jnp.where(
      mask_diag,
      size=self.in_features * self.out_features * self.n_dimension,
    )
    logdet = logdet[idxs].reshape(
      self.n_dimension, self.in_features, self.out_features
    )
    return logdet


class BNAF(nn.Module):
  n_dimension: int
  block_sizes: tuple[int]

  def setup(self):
    layers = []
    in_size = 1
    for out_size in list(self.block_sizes) + [1]:
      layers.append(_BlockMaskedDense(self.n_dimension, in_size, out_size))
      if out_size != 1:
        layers.append(tanh())
      in_size = out_size
    self._layers = layers

  @nn.compact
  def __call__(self, inputs, condition, **kwargs):
    return self.forward_and_log_det(inputs, condition, **kwargs)

  def forward(self, inputs, condition):
    hidden, _ = self.forward_and_log_det(inputs, condition)
    return hidden

  def forward_and_log_det(self, inputs, condition):
    # transforms from p(z) -> p(y)
    hidden, logdets = inputs, None
    condition = nn.one_hot(condition, 8)
    hidden = hidden + nn.Dense(hidden.shape[-1])(condition)
    for layer in self._layers:
      hidden, logdets = layer.forward_and_log_det(hidden, logdets)
    return hidden, jnp.sum(logdets, axis=tuple(range(1, logdets.ndim)))


def get_bnaf_model(n_dimensions=2, n_layers=10, block_sizes=(20, 20, 20)):
  order = jnp.arange(n_dimensions)
  transforms = []
  n_steps = n_layers
  for i in range(n_steps):
    transforms.append(BNAF(n_dimensions, list(block_sizes)))
    if i < n_steps - 1:
      transforms.append(Permutation(order))
      order = order[::-1]
  return Chain(transforms)
