r"""
Routines for resampling and computing effective sample size.
"""

import jax
import jax.numpy as jnp
from functools import partial

from jaxtyping import Array, PRNGKeyArray
import typing as tp
from check_shapes import check_shapes

Key = PRNGKeyArray


class ResamplingResult(tp.NamedTuple):
    samples: Array
    lw: Array


@check_shapes("lw: [b]")
def essl(lw: Array):
    # Mostly copied from github.com/nchopin/particles
    """ESS (Effective sample size) computed from log-weights.

    Parameters
    ----------
    lw: (N,) ndarray
        log-weights

    Returns
    -------
    float
        the ESS of weights w = exp(lw), i.e. the quantity
        sum(w**2) / (sum(w))**2

    Note
    ----
    The ESS is a popular criterion to determine how *uneven* are the weights.
    Its value is in the range [1, N], it equals N when weights are constant,
    and 1 if all weights but one are zero.

    """
    w = jnp.exp(lw - lw.max())
    res = (w.sum()) ** 2 / jnp.sum(w**2)
    return res


@jax.jit
@check_shapes("log_weights: [b]", "samples: [b, d]")
def optionally_resample(
    rng: Key, log_weights: Array, samples: Array, ess_threshold
) -> tp.Dict:
    """
    Optionally resample when the ESS ratio, determined by the log_weights, is lower than the sample size times `ess_threshold`. See documentation of ``resampler`` for remaining arguments.
    """
    N = samples.shape[0]
    lambda_no_resample = lambda x: {"samples": x[1], "lw": x[2], "resampled": False}
    lambda_resample = lambda x: resampler(*x)
    resample_result = jax.lax.cond(
        essl(log_weights) <= ess_threshold * N,
        lambda_resample,
        lambda_no_resample,
        (rng, samples, log_weights),
    )

    return resample_result


# tested
@jax.jit
@check_shapes("log_weights: [b]", "samples: [b, d]")
def resampler(
    rng: Key,
    samples: Array,
    log_weights: Array,
) -> tp.Dict:
    r"""
    Select elements from `samples` with weights defined by `log_weights` using systematic resampling.

    Parameters
    ----------
    rng:
        random key
    samples:
        a sequence of elements to be resampled. Must have the same length as `log_weights`

    Returns
    -------
    Dict object
        contains three attributes:
            * ``samples``, giving the resampled elements
            * ``lw``, giving the new logweights
            * ``resampled``, True
    """
    N = log_weights.shape[0]

    # permute order of samples
    rng, rng_ = jax.random.split(rng)
    log_weights = jax.random.permutation(rng_, log_weights)
    samples = jax.random.permutation(rng_, samples)

    # Generates the uniform variates depending on sampling mode
    rng, rng_ = jax.random.split(rng)
    sorted_uniform = (jax.random.uniform(rng_, (1,)) + jnp.arange(N)) / N

    # Performs resampling given the above uniform variates
    new_idx = jnp.searchsorted(jnp.cumsum(_softmax(log_weights)), sorted_uniform)
    samples = samples[new_idx, :]

    return {"samples": samples, "lw": jnp.zeros(N) - jnp.log(N), "resampled": True}


@check_shapes("lw: [b]")
def _softmax(lw: Array):
    # Mostly copied from github.com/nchopin/particles
    """Exponentiate, then normalise (so that sum equals one).

    Arguments
    ---------
    lw: ndarray
        log weights.

    Returns
    -------
    W: ndarray of the same shape as lw
        W = exp(lw) / sum(exp(lw))

    Note
    ----
    uses the log_sum_exp trick to avoid overflow (i.e. subtract the max
    before exponentiating)
    """
    # noinspection PyArgumentList
    w = jnp.exp(lw - lw.max())
    return w / w.sum()


@check_shapes("v: [b]")
def log_sum_exp(v: Array):
    # Copied from github.com/nchopin/particles
    """Log of the sum of the exp of the arguments.

    Parameters
    ----------
    v: ndarray

    Returns
    -------
    l: float
        l = log(sum(exp(v)))

    Note
    ----
    use the log_sum_exp trick to avoid overflow: i.e. we remove the max of v
    before exponentiating, then we add it back
    """
    # noinspection PyArgumentList
    m = v.max()
    return m + jnp.log(jnp.sum(jnp.exp(v - m)))
