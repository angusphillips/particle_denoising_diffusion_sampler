import jax
import jax.numpy as jnp


def broadcast(lbd, x):
    """Broadcast lbd from [] to [batch_size,]"""
    if len(lbd.shape) == 0:
        lbd = lbd * jnp.ones_like(x[..., 0])
    return lbd
