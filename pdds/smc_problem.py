import jax
import jax.numpy as jnp

from jaxtyping import Float as f, Array, install_import_hook, PRNGKeyArray
import typing as tp
from check_shapes import check_shapes

Key = PRNGKeyArray
NoneType = type(None)

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.distributions import (
    NormalDistribution,
    NormalDistributionWrapper,
)
from pdds.sde import SDE
from pdds.potentials import ApproxPotential
from pdds.utils.shaping import broadcast
from pdds.utils.jax import x_gradient, x_gradient_stateful


class SMCProblem:
    """Class self-containing all the relevant SMC aspects including:
    * Initial Distribution
    * Initial Reweighter
    * Reweighter
    * Markov Kernel
    * Number Steps
    * Tester methods"""

    def __init__(
        self, sde: SDE, approx_potential: ApproxPotential, num_steps: int
    ) -> None:
        """
        Args:
            sde: SDE class giving methods for the chosen SDE
            approx_potential: ApproxPotential, giving either naive approximation or neural network approximation to the log potential function
            num_steps: number of SMC steps to use.
        """
        self.sde = sde
        self.approx_potential = approx_potential

        self.dim = sde.dim
        self.num_steps = num_steps
        self.sigma = sde.sigma
        self.tf = self.sde.scheduler.t_f

        self.initial_distribution = NormalDistributionWrapper(0.0, self.sigma, self.dim)

    @check_shapes("t: []", "x: [b, d]", "return[0]: [b]")
    def _approx_log_g(
        self, t: tp.Union[Array, float], x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        """Evaluate approximate log-potential function at time t on samples x using the given class approximate potential"""
        t = broadcast(jnp.array(t), x)
        lbd = self.sde.scheduler.lambda_t0(t)
        return self.approx_potential.approx_log_gt(
            lbd=lbd, x=x, density_state=density_state
        )

    @check_shapes("t: []", "x: [b, d]", "return[0]: [b, d]")
    def _approx_gradlog_g(
        self, t: tp.Union[Array, float], x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        """Evaluate gradient of approximatee log potential function at time t on samples x using the given approximate potential"""
        t = broadcast(jnp.array(t), x)
        lbd = self.sde.scheduler.lambda_t0(t)
        grad_log_g_fun = x_gradient_stateful(self.approx_potential.approx_log_gt)
        return grad_log_g_fun(lbd=lbd, x=x, density_state=density_state)

    @check_shapes("t: []", "x: [b, d]", "return[0]: [b, d]")
    def _approx_gradlog_pi(
        self, t: tp.Union[Array, float], x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        """Evaluate gradient of approximate log pi_t at time t on samples x"""
        approx_grad_log_g_, density_state = self._approx_gradlog_g(t, x, density_state)
        out = -x / self.sigma**2 + approx_grad_log_g_
        return out, density_state

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def initial_reweighter(self, x: Array, density_state: int) -> tp.Tuple[Array, int]:
        """Reweighting function at first SMC step"""
        return self._approx_log_g(t=self.tf, x=x, density_state=density_state)

    @check_shapes("t_new: []", "t_prev: []", "x_prev: [b, d]")
    def markov_kernel_apply(
        self,
        x_prev: Array,
        t_new: tp.Union[Array, float],
        t_prev: tp.Union[Array, float],
        density_state: int,
    ) -> tp.Tuple[NormalDistribution, int]:
        """Calculate the proposal distribution moving from samples x_prev at time t_prev to time t_new."""
        lbd = self.sde.scheduler.lambda_t(
            broadcast(t_new, x_prev), broadcast(t_prev, x_prev)
        )
        gradlogg, density_state = self._approx_gradlog_g(
            t=t_prev, x=x_prev, density_state=density_state
        )
        means = (
            jnp.sqrt(1 - lbd)[..., None] * x_prev
            + lbd[..., None] * self.sigma**2 * gradlogg
        )
        scale = jnp.sqrt(lbd)[0] * self.sigma
        return NormalDistribution(mean=means, scale=scale, dim=self.dim), density_state

    @check_shapes(
        "t_new: []", "t_prev: []", "x_new: [b, d]", "x_prev: [b, d]", "return[0]: [b]"
    )
    def reweighter(
        self,
        x_new: Array,
        x_prev: Array,
        t_new: tp.Union[Array, float],
        t_prev: tp.Union[Array, float],
        density_state: int,
    ) -> tp.Tuple[Array, int]:
        """Calculate log weights for samples x_new at time t_new given samples x_prev at time t_prev."""
        reverse_logpdf, density_state = self.sde.reverse_transition_dist(
            t_new=broadcast(t_new, x_new),
            t_prev=broadcast(t_prev, x_prev),
            x_prev=x_prev,
        ).evaluate_log_density(x_new, density_state)
        proposal, density_state = self.markov_kernel_apply(
            x_prev=x_prev, t_new=t_new, t_prev=t_prev, density_state=density_state
        )
        proposal_logpdf, density_state = proposal.evaluate_log_density(
            x_new, density_state
        )
        log_g_step, density_state = self._approx_log_g(
            t=t_new, x=x_new, density_state=density_state
        )
        log_g_step_p1, density_state = self._approx_log_g(
            t=t_prev, x=x_prev, density_state=density_state
        )
        res = reverse_logpdf - proposal_logpdf + log_g_step - log_g_step_p1
        return res, density_state

    @check_shapes("x: [b, d]")
    def _get_MCMC_prop_dist(
        self,
        t: tp.Union[Array, float],
        x: Array,
        gamma: tp.Union[Array, float],
        density_state: int,
    ) -> tp.Tuple[NormalDistribution, int]:
        """Calculates the proposal distribution of MALA targeting pi_t"""
        grad_log_pi, density_state = self._approx_gradlog_pi(t, x, density_state)
        dist = NormalDistribution(
            mean=x + gamma * grad_log_pi,
            scale=jnp.sqrt(2 * gamma),
            dim=x.shape[-1],
        )
        return dist, density_state

    @check_shapes("t: []", "x: [b, d]", "return[0]: [b]")
    def _log_pi(
        self, t: tp.Union[Array, float], x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        """Approximates log pi_t(x)"""
        approx_log_g, density_state = self._approx_log_g(t, x, density_state)
        norm_density, density_state = NormalDistributionWrapper(
            0.0, self.sigma, x.shape[-1]
        ).evaluate_log_density(x, density_state)
        log_pi = approx_log_g + norm_density
        return log_pi, density_state

    @check_shapes("t: []", "gamma: []")
    def get_MCMC_kernel(
        self, t: tp.Union[Array, float], gamma: tp.Union[Array, float]
    ) -> tp.Callable:
        """Wraps one step of invariant MCMC kernel (MALA) targetting pi_t for (optionally) refreshing samples after resampling."""

        @check_shapes("carry[0]: [b, d]")
        def MCMC_kernel(
            carry: tp.Tuple[Array, int], rng: Key
        ) -> tp.Tuple[Array, Array, int]:
            x = carry[0]
            density_state = carry[1]
            rng, rng1, rng2 = jax.random.split(rng, 3)
            prop_dist, density_state = self._get_MCMC_prop_dist(
                t, x, gamma, density_state
            )
            proposal = prop_dist.sample(rng1, num_samples=x.shape[0])
            prop_log_pi_term, density_state = self._log_pi(t, proposal, density_state)
            rev_prop_dist, density_state = self._get_MCMC_prop_dist(
                t, proposal, gamma, density_state
            )
            rev_prop_term, density_state = rev_prop_dist.evaluate_log_density(
                x, density_state
            )
            x_log_pi_term, density_state = self._log_pi(t, x, density_state)
            fwd_prop_term, density_state = prop_dist.evaluate_log_density(
                proposal, density_state
            )
            log_acceptance_prob = (
                prop_log_pi_term + rev_prop_term - x_log_pi_term - fwd_prop_term
            )

            acceptance_prob = jnp.minimum(
                jnp.exp(log_acceptance_prob), jnp.ones_like(log_acceptance_prob)
            )
            u = jax.random.uniform(rng2, shape=(x.shape[0],))
            accept_ratio = jnp.mean(acceptance_prob > u)
            x_new = jnp.where(jnp.expand_dims(acceptance_prob > u, -1), proposal, x)
            return (x_new, density_state), accept_ratio

        return MCMC_kernel

    # ====== Methods for testing ======

    @staticmethod
    def _batch_scalar_prod(A, B):
        return jnp.sum(jnp.array(A) * jnp.array(B), axis=1)

    def tester_weight_A_(self, x_new, x_prev, t_new, t_prev, density_state):
        # should have been private
        log_gt, density_state = self._approx_log_g(t_new, x_new, density_state)  # (N,)
        log_gtp1, density_state = self._approx_log_g(
            t_prev, x_prev, density_state
        )  # (N,)
        gradlog_g, density_state = self._approx_gradlog_g(t_prev, x_prev, density_state)
        approx_ratio = self._batch_scalar_prod(gradlog_g, x_new - x_prev)
        return log_gt - log_gtp1 - approx_ratio, density_state

    def tester_weight_B_(self, x_prev, t_new, t_prev, density_state):
        # should have been private
        lbd = self.sde.scheduler.lambda_t(
            broadcast(t_new, x_prev), broadcast(t_prev, x_prev)
        )
        gradlog, density_state = self._approx_gradlog_g(
            t_prev, x_prev, density_state
        )  # (N, dim)
        sq_norm = (gradlog**2).sum(axis=1)  # (N,)
        scp = self._batch_scalar_prod(gradlog, x_prev)  # (N,)
        prefactor = float(1 - jnp.sqrt(1 - lbd)[0])
        return lbd * self.sigma**2 / 2 * sq_norm - prefactor * scp, density_state
