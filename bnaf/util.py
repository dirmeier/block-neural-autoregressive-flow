from jax import numpy as jnp
from numpyro.distributions.util import vec_to_tril_matrix


def block_triu_mask(n_blocks: int, shape: tuple):
  n_elements = jnp.ones(n_blocks * (n_blocks - 1) // 2)
  mask = vec_to_tril_matrix(n_elements, -1).T[..., None]
  mask = jnp.tile(mask, (1, *shape)).reshape(
    (shape[0] * n_blocks, shape[1] * n_blocks)
  )
  return mask
