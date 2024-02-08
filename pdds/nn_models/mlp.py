import haiku as hk
import jax
import jax.numpy as jnp
import jax.nn as jnn
import functools
from check_shapes import check_shapes
from pdds.utils.shaping import broadcast


def gelu(x):
    """We use this in place of jax.nn.relu because the approximation used.

    Args:
      x: input

    Returns:
      GELU activation
    """
    return x * 0.5 * (1.0 + jax.scipy.special.erf(x / jnp.sqrt(2.0)))


activations = {
    "elu": jnn.elu,
    "relu": jnn.relu,
    "gelu": gelu,
    "lrelu": functools.partial(jnn.leaky_relu, negative_slope=0.01),
    "swish": jnn.swish,
    "sin": jnp.sin,
    "none": lambda x: x,
    "const": lambda x: jnp.ones_like(x),
}


class LinearConsInit(hk.Module):
    """Linear layer with constant init."""

    def __init__(self, output_size, alpha=1, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.output_size = output_size

    def __call__(self, x):
        j, k = x.shape[-1], self.output_size
        w_init = hk.initializers.Identity(gain=self.alpha)
        w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=w_init)
        b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.zeros)

        return jnp.dot(x, w) + b


class LinearZero(hk.Module):
    """Linear layer with zero init."""

    def __init__(self, output_size, alpha=1, name=None):
        super().__init__(name=name)
        self.alpha = alpha
        self.output_size = output_size

    def __call__(self, x):
        j, k = x.shape[-1], self.output_size
        w = hk.get_parameter("w", shape=[j, k], dtype=x.dtype, init=jnp.zeros)
        b = hk.get_parameter("b", shape=[k], dtype=x.dtype, init=jnp.zeros)

        return jnp.dot(x, w) + b


class PISGRADNet(hk.Module):
    """PIS Grad network. Other than detaching should mimic the PIS Grad network.

    We detach the ULA gradients treating them as just features leading to much
    more stable training than PIS Grad.

    FROM DDS codebase, detaching removed.
    """

    def __init__(
        self,
        hidden_shapes: list,
        act: str,
        dim: int,
    ):
        super().__init__()

        self.hidden_shapes = hidden_shapes
        self.n_layers = len(hidden_shapes)
        self.act = activations[act]

        # For most PIS_GRAD experiments channels = 64
        self.channels = hidden_shapes[0]
        self.timestep_phase = hk.get_parameter(
            "timestep_phase", shape=[1, self.channels], init=jnp.zeros
        )

        # Exact time_step coefs used in PIS GRAD
        self.timestep_coeff = jnp.linspace(start=0.1, stop=100, num=self.channels)[None]

        # This implements the time embedding for the non grad part of the network
        self.t_encoder = hk.Sequential(
            [
                hk.Linear(self.channels),
                self.act,
                hk.Linear(self.channels),
            ]
        )

        # This carries out the time embedding for the NN(t) * log grad target
        self.smooth_net = hk.Sequential(
            [hk.Linear(self.channels)]
            + [
                hk.Sequential([self.act, hk.Linear(self.channels)])
                for _ in range(self.n_layers)
            ]
            + [self.act, LinearConsInit(dim, 0)]
        )

        # Time embedding and state concatenated network NN(x, emb(t))
        # This differs to PIS_grad where they do NN(Wx + emb(t))
        self.nn = hk.Sequential(
            [hk.Sequential([hk.Linear(x), self.act]) for x in self.hidden_shapes]
            + [LinearZero(dim)]
        )

    @check_shapes("lbd: [batch_size, 1]", "return: [batch_size, channels]")
    def get_pis_timestep_embedding(self, lbd):
        sin_embed_cond = jnp.sin((self.timestep_coeff * lbd) + self.timestep_phase)
        cos_embed_cond = jnp.cos((self.timestep_coeff * lbd) + self.timestep_phase)
        return jnp.concatenate([sin_embed_cond, cos_embed_cond], axis=-1)

    @check_shapes("lbd: [batch_size]", "return: [batch_size]")
    def smoothing_function(self, lbd):
        lbd_emb = self.get_pis_timestep_embedding(lbd[..., None])
        zero_emb = self.get_pis_timestep_embedding(jnp.zeros_like(lbd)[..., None])
        return self.smooth_net(lbd_emb)[..., 0] - self.smooth_net(zero_emb)[..., 0]

    @check_shapes(
        "x: [batch_size, dim]",
        "lbd: [batch_size]",
        "residual: [batch_size]",
        "return: [batch_size]",
    )
    def __call__(self, lbd, x, residual):
        lbd = broadcast(lbd, x)
        smooth = self.smoothing_function(lbd)
        lbd_emb = self.get_pis_timestep_embedding(lbd[..., None])
        lbd_emb = self.t_encoder(lbd_emb)

        # could be beneficial to embed x but that would cause disparity with PIS and DDS networks
        net_out = self.nn(jnp.concatenate([x, lbd_emb], axis=-1))
        sp_out = jax.vmap(lambda x, y: jnp.dot(x, y))(net_out, x)

        return smooth * sp_out + (jnp.ones_like(lbd) - smooth) * residual
