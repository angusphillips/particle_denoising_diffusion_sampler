import jax
import jax.numpy as jnp

import abc

import diffrax as dfx
from diffrax import (
    AbstractSolver,
    ConstantStepSize,
    AbstractAdaptiveSolver,
)

# Types:
from jaxtyping import Array, PRNGKeyArray
import typing as tp
from check_shapes import check_shapes

from pdds.utils.jax import x_gradient_no_t, x_gradient_no_t_stateful

Key = PRNGKeyArray

from jaxtyping import install_import_hook

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.distributions import (
    Distribution,
    NormalDistribution,
    NormalDistributionWrapper,
)
from pdds.utils.shaping import broadcast


class Scheduler(metaclass=abc.ABCMeta):
    def __init__(self) -> None:
        pass

    @abc.abstractmethod
    @check_shapes("t: [b]", "return: [b]")
    def beta_t(self, t: Array) -> Array:
        """Implements the noise schedule"""

    @abc.abstractmethod
    @check_shapes("t: [b]", "return: [b]")
    def beta_int(self, t: Array) -> Array:
        """Indefinite integral of beta_t"""

    @check_shapes("t1: [b]", "t0: [b]", "return: [b]")
    def lambda_t(self, t1: Array, t0: Array) -> Array:
        """Computes the time rescaling:
        lambda_t = 1 - exp(-2 * int_t0^t1(beta(s)ds))
        Ensures the integral is always positive, so order of t1 and t0 do not matter"""
        t_int = jnp.abs(self.beta_int(t1) - self.beta_int(t0))
        return 1 - jnp.exp(-2 * t_int)

    @check_shapes("t: [b]", "return: [b]")
    def lambda_t0(self, t: Array) -> Array:
        return self.lambda_t(t, jnp.zeros_like(t))

    @abc.abstractmethod
    @check_shapes("lbd: [b]")
    def beta_t_inv_lambda(self, lbd: Array) -> Array:
        """Computes beta_t for a given lambda"""


class ConstantScheduler(Scheduler):
    def __init__(self, t_0: float, t_f: float, beta: float):
        self.t_0 = t_0
        self.t_f = t_f
        self.beta = beta

    @check_shapes("t: [b]", "return: [b]")
    def beta_t(self, t: Array) -> Array:
        return self.beta * jnp.ones_like(t)

    @check_shapes("t: [b]", "return: [b]")
    def beta_int(self, t: Array) -> Array:
        return self.beta * t

    @check_shapes("lbd: [b]", "return: [b]")
    def beta_t_inv_lambda(self, lbd: Array) -> Array:
        return self.beta


class LinearScheduler(Scheduler):
    def __init__(self, t_0: float, t_f: float, beta_0: float, beta_f: float):
        self.t_0 = t_0
        self.t_f = t_f
        self.beta_0 = beta_0
        self.beta_f = beta_f

    @check_shapes("t: [b]", "return: [b]")
    def beta_t(self, t: Array) -> Array:
        normed_t = (t - self.t_0) / (self.t_f - self.t_0)
        return self.beta_0 + normed_t * (self.beta_f - self.beta_0)

    @check_shapes("t: [b]", "return: [b]")
    def beta_int(self, t: Array) -> Array:
        return 0.5 * t**2 * (self.beta_f - self.beta_0) / self.t_f + t * self.beta_0

    @check_shapes("int: [b]", "return: [b]")
    def inv_beta_int(self, int: Array) -> Array:
        a = 0.5 * (self.beta_f - self.beta_0) / self.t_f
        b = self.beta_0
        c = -int
        return b - jnp.sqrt(b**2 - 4 * a * c) / (2 * a)

    @check_shapes("lbd: [b]", "return: [b]")
    def inv_lambda_t(self, lbd: Array) -> Array:
        beta_int = -0.5 * jnp.log(1 - lbd)
        return -self.inv_beta_int(beta_int)

    @check_shapes("lbd: [b]", "return: [b]")
    def beta_t_inv_lambda(self, lbd: Array) -> Array:
        return self.beta_t(self.inv_lambda_t(lbd))


class CosineScheduler(Scheduler):
    def __init__(self, t_0: float, t_f: float, s: float = 0.008):
        self.t_0 = t_0
        self.t_f = t_f
        self.s = s

    @check_shapes("t: [b]")
    def beta_t(self, t: Array):
        return NotImplementedError

    @check_shapes("t: [b]")
    def beta_int(self, t: Array):
        return NotImplementedError

    @check_shapes("t: [b]", "return: [b]")
    def lambda_t0(self, t: Array) -> Array:
        return 1 - jnp.cos(0.5 * jnp.pi * ((t / self.t_f) + self.s) / (1 + self.s)) ** 2

    @check_shapes("t1: [b]", "t0: [b]", "return: [b]")
    def lambda_t(self, t1: Array, t0: Array) -> Array:
        return 1 - jnp.exp(
            -jnp.abs(
                jnp.log(
                    jnp.cos(0.5 * jnp.pi * ((t1 / self.t_f) + self.s) / (1 + self.s))
                    ** 2
                )
                - jnp.log(
                    jnp.cos(0.5 * jnp.pi * ((t0 / self.t_f) + self.s) / (1 + self.s))
                    ** 2
                )
            )
        )

    @check_shapes("lbd: [b]")
    def beta_t_inv_lambda(self, lbd: Array):
        return NotImplementedError  # TODO


class DDSScheduler(Scheduler):
    """Clumsy implementation of the DDS scheduler - does not fit nicely with our scheduling framework since
    we work with t which gives lots of flexibility but the DDS schedule is discrete.
    """

    def __init__(
        self, t_0: float, t_f: float, num_steps: int, alpha_max: float, s: float = 0.008
    ):
        self.t_0 = t_0
        self.t_f = t_f
        self.alpha_max = alpha_max
        self.s = s
        self.num_steps = num_steps
        self.dt = (self.t_f - self.t_0) / self.num_steps
        self.ts = jnp.linspace(self.t_0, self.t_f, self.num_steps + 1)
        self.lambdas_incr = self.lambda_t(self.ts[1:], self.ts[:-1])
        self.lambdas_0 = 1 - jnp.cumprod(1 - self.lambdas_incr)
        self.lambdas_0 = jnp.concatenate([jnp.array([0.0]), self.lambdas_0])

    @check_shapes("t: [b]")
    def beta_t(self, t: Array):
        return NotImplementedError

    @check_shapes("t: [b]")
    def beta_int(self, t: Array):
        return NotImplementedError

    @check_shapes("t1: [b]", "t0: [b]", "return: [b]")
    def lambda_t(self, t1: Array, t0: Array) -> Array:
        """Caution only a good approximation when t1 and t0 are one 'step' apart.
        Need to use lambda_t0 method for computing lambda_t in potential approximation
        Have used t1+t0/2 to make sure agnostic to ordering of time inputs.
        """
        t = (t1 + t0) / (2 * self.t_f)
        phase = 0.5 * jnp.pi * (1.0 - t + self.s) / (1 + self.s)
        return (self.alpha_max * jnp.cos(phase) ** 2) ** 2

    @check_shapes("t: [b]", "return: [b]")
    def lambda_t0(self, t: Array) -> Array:
        k = jnp.array(
            t * self.num_steps / self.t_f, dtype=int
        )  # slightly annoying converting back from times to steps
        return self.lambdas_0[k]

    @check_shapes("lbd: [b]")
    def beta_t_inv_lambda(self, lbd: Array):
        return NotImplementedError


class SDE:
    """
    Useful functions for simulating a Variance Preserving SDE.
    Reference distribution is N(0, sigma**2).
    This class is agnostic to the number of discretisation steps.
    """

    def __init__(self, scheduler: Scheduler, sigma: float = 1.0, dim: int = 1):
        super().__init__()
        self.scheduler = scheduler
        self.sigma = sigma
        self.dim = dim
        self.reference_dist = NormalDistributionWrapper(mean=0.0, scale=sigma, dim=dim)

    @check_shapes("t_new: [b]", "t_prev: [b]", "x_prev: [b, d]", "return: [b, d]")
    def expected_denoising(self, t_new: Array, t_prev: Array, x_prev: Array) -> Array:
        """Mean of the denoising distribution p(x_tnew|x_tprev) with the expectation that t_new < t_prev."""
        lbd = self.scheduler.lambda_t(t_new, t_prev)
        return jnp.sqrt(1 - lbd)[..., None] * x_prev

    @check_shapes("t_new: [b]", "t_prev: [b]", "x_prev: [b, d]")
    def reverse_transition_dist(
        self, t_new: Array, t_prev: Array, x_prev: Array
    ) -> Distribution:
        """Calculate the distribution of:
                p(x_tnew | x_tprev) = N(x_tnew; sqrt(1-lambda_{t_new, t_prev}) * x_tprev, lambda_{t_new, t_prev} * sigma **2)
        with the expectation that t_new < t_prev"""
        lbd = self.scheduler.lambda_t(t_new, t_prev)
        means = jnp.sqrt(1 - lbd)[..., None] * x_prev
        scale = (
            jnp.sqrt(lbd)[0] * self.sigma
        )  # WARNING assumes all samples have same lbd TODO generalise this?
        return NormalDistribution(means, scale, dim=self.dim)

    @check_shapes("t_new: [b]", "t_prev: [b]", "x_prev: [b, d]")
    def forward_transition_dist(
        self, t_new: Array, t_prev: Array, x_prev: Array
    ) -> Distribution:
        """Calculate the distribution of:
                p(x_tnew | x_tprev) = N(x_tnew; sqrt(1-lambda_{t_new, t_prev}) * x_tprev, lambda_{t_new, t_prev} * sigma **2)
        with the expectation that t_new > t_prev"""
        lbd = self.scheduler.lambda_t(t_new, t_prev)
        means = jnp.sqrt(1 - lbd)[..., None] * x_prev
        scale = (
            jnp.sqrt(lbd)[0] * self.sigma
        )  # WARNING assumes all samples have same lbd TODO generalise this?
        return NormalDistribution(means, scale, dim=self.dim)

    @check_shapes("t: [b]", "x0: [b, d]")
    def forward_path_marginal_dist(self, t: Array, x0: Array) -> Distribution:
        """Calculate the distribution of:
        p(x_t | x_0) = N(x_t; sqrt(1-lambda_{t, 0}) * x_0, lambda_{t, 0} * sigma **2)
        in the forward time convention
        """
        return self.forward_transition_dist(t, broadcast(jnp.array(0.0), x0), x0)

    @check_shapes("t: [b]", "xT: [b, d]")
    def reverse_path_marginal_dist(self, t: Array, xT: Array) -> Distribution:
        """Calculate the distribution of:
        p(x_t | x_T) = N(x_t; sqrt(1-lambda_{t, T}) * x_T, lambda_{t, T} * sigma **2)
        in the forward time convention
        """
        return self.reverse_transition_dist(
            t, broadcast(jnp.array(self.scheduler.t_f), xT), xT
        )

    @check_shapes("lbd: [b]", "t: [b]", "x: [b, d]")
    def dsm_loss(
        self,
        key: Key,
        lbd: Array,
        t: Array,
        x: Array,
        score_function,
        density_state: int,
        likelihood_weight=False,
    ):
        """Evaluates the denoising score-matching objective:
        mean_i{w * ||\nabla\log\pi_\theta(x_t^i, t) - \nabla\log N(x_t^i; \sqrt{1-\lambda_t}x_0^i, sigma^2 \lambda_t)||^2}
        where w is either the standard scale weighting or the likelihood weighting (https://papers.nips.cc/paper/2021/file/0a9fdbb17feb6ccb7ec405cfb85222c4-Paper.pdf)

        Args:
            key: PRNGKeyArray, Jax random key
            lbd: Array, time rescaling coefficient for given timestep
            t: Optional Array, timestep used when computing likelihood weighting
            x: Array, samples from (approximate) target
            score_function: Callable, approximation of \nabla\log\pi_t
            density_state: Int, density state
            likelihood weight: bool, whether to use likelihood weighting, default False
        Returns:
            loss: score-matching loss
            density_state: updated density state
        """
        key, skey = jax.random.split(key)

        mean = jnp.sqrt(1 - lbd)[..., None] * x
        scale = jnp.sqrt(lbd)[..., None] * self.sigma
        noise = jax.random.normal(skey, x.shape)
        x_t = mean + noise * scale  # [b, d]

        score_t, density_state = score_function(
            lbd=lbd, x=x_t, density_state=density_state
        )  # [b, d]

        if likelihood_weight:
            if t is None:
                return ValueError(
                    "Likelihood weighting cannot be used when sampling lbd"
                )
            weight = self.scheduler.beta_t(t)  # brownian weight squared
            losses = weight[..., None] * jnp.square(score_t + noise / scale)
        else:
            losses = jnp.square(scale * score_t + noise)  # [b, d]

        return jnp.mean(jnp.sum(losses, axis=-1)), density_state

    @check_shapes("lbd: [b]", "t: [b]", "x: [b, d]")
    def guidance_loss(
        self,
        key: Key,
        lbd: Array,
        t: Array,
        x: Array,
        score_function,
        log_g0,
        density_state: int,
    ):
        """Implements the guidance loss (Novel Score Matching loss) from our paper:
        mean_i{w * ||\nabla\log g_\theta(x_t^i, t) - \nabla\log g_0(x_0^i)||^2}

        Args:
            key: PRNGKeyArray, Jax random key
            lbd: Array, time rescaling coefficient for given timestep
            t: Optional Array, unused in this loss function
            x: Array, samples from (approximate) target
            score_function: Callable, approximation of \nabla\log g_t
            log_g0: Callable, log-potential function at time t=0
            density_state: Int, density state
        Returns:
            loss: score-matching loss
            density_state: updated density state
        """
        key, skey = jax.random.split(key)

        mean = jnp.sqrt(1 - lbd)[..., None] * x
        scale = jnp.sqrt(lbd)[..., None] * self.sigma
        noise = jax.random.normal(skey, x.shape)
        x_t = mean + noise * scale

        score_t, density_state = score_function(
            lbd=lbd, x=x_t, density_state=density_state
        )  # [b, d]
        grad_log_g, density_state = x_gradient_no_t_stateful(log_g0._log_g0)(
            x, density_state
        )  # [b, d]
        noisy_potential = jnp.sqrt(1 - lbd)[..., None] * grad_log_g
        losses = jnp.square(score_t - noisy_potential)

        return jnp.mean(jnp.sum(losses, axis=-1)), density_state

    @check_shapes("x_curr: [batch_size, dim]", "t: [b]")
    def drift(self, t: Array, x_curr: Array):
        return -self.scheduler.beta_t(t)[..., None] * x_curr

    @check_shapes("x_curr: [batch_size, dim]", "t: [b]")
    def diffusion(self, t: Array, x_curr: Array):
        scale = jnp.sqrt(2 * self.sigma**2 * self.scheduler.beta_t(t))
        return scale[..., None] * jnp.ones_like(x_curr)

    @check_shapes("x_curr: [batch_size, dim]", "t: [b]")
    def reverse_drift_ode(self, t: Array, x_curr: Array, grad_log_pi) -> Array:
        lbd = self.scheduler.lambda_t0(t)
        second_term = (
            self.sigma**2
            * self.scheduler.beta_t(t)[..., None]
            * grad_log_pi(lbd=lbd, x=x_curr)  # [batch_size, dim]
        )
        return (
            self.drift(t, x_curr) - second_term
        )  # diffrax already puts in the - sign when time is reversed

    @check_shapes("x_curr: [batch_size, dim]", "t: [b]")
    def reverse_drift_sde(self, t: Array, x_curr: Array, grad_log_pi) -> Array:
        lbd = self.scheduler.lambda_t0(t)
        second_term = (
            2
            * self.sigma**2
            * self.scheduler.beta_t(t)[..., None]
            * grad_log_pi(lbd=lbd, x=x_curr)  # [batch_size, dim]
        )
        return (
            self.drift(t, x_curr) - second_term
        )  # diffrax already puts in the - sign when time is reversed


def dsm_loss(
    key: Key,
    sde: SDE,
    score_function,
    x: Array,
    density_state: int,
    likelihood_weight=False,
    sample_lbd=False,
    eps=1e-3,
):
    """Wrapper around SDE.dsm_loss. Deals with sampling time-steps either uniformly or sampling lambda uniformly.

    Args:
        key: PRNGKeyArray, Jax key
        sde: SDE class, instance of SDE
        score_function: Callable, approximation of \nabla\log\pi_t
        x: Array, samples from (approximate) target)
        density_state: Int, density_state
        likelihood_weight: bool, whether to uselikelihood weight (sde.dsm_loss), default False
        sample_lbd: bool, whether to sample lambda uniformly as opposed to t uniformly, default False
        eps: float, minimum sampling time threshold.
    Returns:
        loss: score-matching loss
        density_state: updated density state
    """
    batch_size = x.shape[0]
    key, subkey = jax.random.split(key)
    # lbd = jax.random.uniform(subkey, (batch_size,))  # sample lambda between 0 and 1

    t0 = sde.scheduler.t_0
    t1 = sde.scheduler.t_f
    if sample_lbd:
        lbd = jax.random.uniform(
            subkey,
            (batch_size,),
            minval=0.0 + eps,
            maxval=sde.scheduler.lambda_t(
                broadcast(jnp.array(t0), x), broadcast(jnp.array(t1), x)
            ),
        )
        t = None
    else:
        # Low-discrepancy sampling over t to reduce variance
        t = jax.random.uniform(
            subkey, (batch_size,), minval=t0 + eps, maxval=t1 / batch_size
        )
        t = t + (t1 / batch_size) * jnp.arange(batch_size)
        lbd = sde.scheduler.lambda_t0(t)

    loss, density_state = sde.dsm_loss(
        key, lbd, t, x, score_function, density_state, likelihood_weight
    )

    return loss, density_state


def guidance_loss(
    key: Key,
    sde: SDE,
    score_function,
    x,
    density_state,
    log_g0,
    sample_lbd=True,
    eps=1e-3,
):
    """Wrapper around SDE.guidance_loss. Deals with sampling time-steps either uniformly or sampling lambda uniformly.

    Args:
        key: PRNGKeyArray, Jax key
        sde: SDE class, instance of SDE
        score_function: Callable, approximation of \nabla\log\g_t
        x: Array, samples from (approximate) target)
        density_state: Int, density_state
        log_g0: Callable, log-potential function at time t=0
        sample_lbd: bool, whether to sample lambda uniformly as opposed to t uniformly, default False
        eps: float, minimum sampling time threshold.
    Returns:
        loss: NSM loss
        density_state: updated density state
    """
    batch_size = x.shape[0]
    key, subkey = jax.random.split(key)

    t0 = sde.scheduler.t_0
    t1 = sde.scheduler.t_f
    if sample_lbd:
        lbd = jax.random.uniform(
            subkey,
            (batch_size,),
            minval=0.0 + eps,
            maxval=sde.scheduler.lambda_t(
                broadcast(jnp.array(t0), x), broadcast(jnp.array(t1), x)
            ),
        )
        t = None
    else:
        # Low-discrepancy sampling over t to reduce variance
        t = jax.random.uniform(
            subkey, (batch_size,), minval=t0 + eps, maxval=t1 / batch_size
        )
        t = t + (t1 / batch_size) * jnp.arange(batch_size)
        lbd = sde.scheduler.lambda_t0(t)

    losses, density_state = sde.guidance_loss(
        key, lbd, t, x, score_function, log_g0, density_state
    )
    return jnp.mean(losses), density_state


def sde_solve(
    sde: SDE,
    grad_log_pi,
    x,
    *,
    key,
    prob_flow: bool = False,
    num_steps: int = 100,
    solver: AbstractSolver = dfx.Heun(),
    rtol: tp.Union[None, float] = 1e-4,
    atol: tp.Union[None, float] = 1e-5,
    forward: bool = False,
    ts=None,
    tf=None,
    max_steps=4096,
):
    if (rtol is None or atol is None) or not isinstance(solver, AbstractAdaptiveSolver):
        stepsize_controller = ConstantStepSize()
    else:
        stepsize_controller = dfx.PIDController(rtol=rtol, atol=atol)

    # NOTE: default is time-reversal
    if forward:
        t0, t1 = sde.scheduler.t_0, sde.scheduler.t_f
        drift_sde = lambda t, xt, args: sde.drift(broadcast(t, xt), xt)
    else:
        t1, t0 = sde.scheduler.t_0, sde.scheduler.t_f
        drift_sde = lambda t, xt, args: sde.reverse_drift_sde(
            broadcast(t, xt), xt, grad_log_pi
        )

    t1 = tf if tf is not None else t1
    dt = (t1 - t0) / num_steps  # TODO: dealing properly with endpoint?

    if prob_flow:
        reverse_drift_ode = lambda t, xt, args: sde.reverse_drift_ode(
            broadcast(t, xt), xt, grad_log_pi
        )
        terms = dfx.ODETerm(reverse_drift_ode)
    else:
        shape = jax.ShapeDtypeStruct(x.shape, x.dtype)
        key, subkey = jax.random.split(key)
        # if isinstance(stepsize_controller, AbstractAdaptiveStepSizeController):
        bm = dfx.VirtualBrownianTree(
            t0=t0, t1=t1, tol=jnp.abs(dt), shape=shape, key=subkey
        )
        # else:
        #     bm = dfx.UnsafeBrownianPath(shape=shape, key=subkey)
        diffusion = lambda t, xt, args: sde.diffusion(broadcast(t, xt), xt)

        terms = dfx.MultiTerm(
            dfx.ODETerm(drift_sde), dfx.WeaklyDiagonalControlTerm(diffusion, bm)
        )

    if ts is None:
        saveat = dfx.SaveAt(t1=True)
    elif ts == []:
        saveat = dfx.SaveAt(steps=True)
    else:
        saveat = dfx.SaveAt(ts=ts)
    out = dfx.diffeqsolve(
        terms,
        solver=solver,
        t0=t0,
        t1=t1,
        dt0=dt,
        y0=x,
        stepsize_controller=stepsize_controller,
        saveat=saveat,
        # max_steps=max(max_steps, num_steps),
    )
    xs = out.ys
    return xs
