from flax import linen as nn
from flax.linen import softplus
from jax import numpy as jnp


class tanh(nn.Module):
    def forward_and_log_det(self, inputs, logdets, **kwargs):
        out = jnp.tanh(inputs)
        tanh_logdet = self.forward_log_det(inputs, logdets)
        return out, logdets + tanh_logdet

    def forward_log_det(self, inputs, logdets):
        tanh_logdet = _tanh_log_grad(inputs)
        tanh_logdet = tanh_logdet.reshape(
            logdets.shape[:-2] + (1, logdets.shape[-1])
        )
        return tanh_logdet

    def inverse_and_log_det(self, inputs, logdets):
        out = jnp.arctanh(inputs)
        tanh_logdet = self.forward_log_det(out, logdets)
        return out, logdets - tanh_logdet


def _tanh_log_grad(x):
    return -2 * (x + softplus(-2 * x) - jnp.log(2.0))


class leaky_tanh(nn.Module):
    max_val: float

    def setup(self):
        self.linear_grad = jnp.exp(_tanh_log_grad(self.max_val))
        self.intercept = (
            jnp.tanh(self.max_val) - self.linear_grad * self.max_val
        )

    def forward_and_log_det(self, inputs, logdets, **kwargs):
        out = self.forward(inputs)
        tanh_logdet = self.forward_log_det(inputs, logdets)
        return out, logdets + tanh_logdet

    def forward(self, inputs):
        is_linear = jnp.abs(inputs) >= self.max_val
        linear_y = self.linear_grad * inputs + jnp.sign(inputs) * self.intercept
        tanh_y = jnp.tanh(inputs)
        return jnp.where(is_linear, linear_y, tanh_y)

    def forward_log_det(self, inputs, logdets):
        tanh_logdet = jnp.where(
            jnp.abs(inputs) >= self.max_val,
            jnp.log(self.linear_grad),
            _tanh_log_grad(inputs),
        )
        tanh_logdet = tanh_logdet.reshape(
            logdets.shape[:-2] + (1, logdets.shape[-1])
        )
        return tanh_logdet
