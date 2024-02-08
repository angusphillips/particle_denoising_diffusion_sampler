import haiku as hk
import jax.numpy as jnp
from check_shapes import check_shapes


class NoneCorrection(hk.Module):
    @check_shapes("x: [batch_size, dim]", "t: [batch_size]")
    def __call__(self, x, t):
        return jnp.zeros_like(x[..., 0])
