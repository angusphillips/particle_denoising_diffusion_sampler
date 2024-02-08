import jax
import jax.numpy as jnp
import typing as tp
from jaxtyping import PRNGKeyArray as Key


def _get_key_iter(init_key: Key) -> tp.Iterator["jax.random.PRNGKey"]:
    while True:
        init_key, next_key = jax.random.split(init_key)
        yield next_key


def x_gradient(func):
    return lambda lbd, x: jax.vjp(lambda x: func(lbd=lbd, x=x), x)[1](
        jnp.ones_like(x)[..., 0]
    )[0]


def x_gradient_stateful(func):
    def stateful_gradient(lbd, x, density_state):
        _, vjpfun, density_state = jax.vjp(
            lambda x: func(lbd=lbd, x=x, density_state=density_state), x, has_aux=True
        )
        grad = vjpfun(jnp.ones_like(x)[..., 0])[0]
        return grad, density_state

    return stateful_gradient


def x_gradient_no_t(func):
    return lambda x: jax.vjp(func, x)[1](jnp.ones_like(x)[..., 0])[0]


def x_gradient_no_t_stateful(func):
    def stateful_gradient_no_t(x, density_state):
        _, vjpfun, density_state = jax.vjp(
            lambda x: func(x=x, density_state=density_state), x, has_aux=True
        )
        grad = vjpfun(jnp.ones_like(x)[..., 0])[0]
        return grad, density_state

    return stateful_gradient_no_t


def x_gradient_parametrised(func):
    return lambda params, lbd, x: jax.vjp(
        lambda x: func(params=params, lbd=lbd, x=x), x
    )[1](jnp.ones_like(x)[..., 0])[0]


def x_gradient_stateful_parametrised(func):
    def stateful_gradient_parametrised(params, lbd, x, density_state):
        _, vjpfun, density_state = jax.vjp(
            lambda x: func(params=params, lbd=lbd, x=x, density_state=density_state),
            x,
            has_aux=True,
        )
        grad = vjpfun(jnp.ones_like(x)[..., 0])[0]
        return grad, density_state

    return stateful_gradient_parametrised
