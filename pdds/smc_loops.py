import jax
import jax.numpy as jnp
import numpy as np
import typing as tp
from tqdm import tqdm

from jaxtyping import Float as f, Array, PRNGKeyArray, install_import_hook
from check_shapes import check_shapes

Key = PRNGKeyArray
Callable = tp.Callable

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.resampling import (
    log_sum_exp,
    optionally_resample,
    essl,
)
from pdds.smc_problem import SMCProblem

from pdds.utils.loggers_pl import LoggerCollection


@check_shapes("samples: [b, d]", "log_weights: [b]")
def inner_loop(
    rng: Key,
    smc_problem: SMCProblem,
    t_new: f[Array, ""],
    t_prev: f[Array, ""],
    samples: Array,
    log_weights: Array,
    num_particles: int,
    ess_threshold: float,
    num_mcmc_steps: int,
    mcmc_step_size: float,
    density_state: int,
) -> tp.Dict:
    """Single step of the PDDS smc algorithm inner loop, consisting of proposal, reweighting and optional resampling/mcmc steps.
    Args:
        rng: PRNGKeyArray
        smc_problem: SMCProblem, contains smc functions eg reweighting and markov kernel functions
        t_new: next time step
        t_prev: previous time step
        samples: Array, previous samples
        log_weights: Array, previous log weights
        num_particles: int, number of particles
        ess_threshold: float, threshold for resampling
        num_mcmc_steps: int, number of mcmc steps to perform
        mcmc_step_size: float, mcmc step size
        density_state: int, density state
    Returns:
        Dict: containing
            "samples_new": samples at new time step,
            "log_weights_new": log weights for new particles,
            "log_normaliser_increment": log normaliser increment, forms an estimate of log{Z_tnew/Z_tprev},
            "acceptance_ratio": average acceptance ratio of MCMC steps,
            "resampled": bool, indicates whether resampling was preformed
        Int: updated density state
    """
    # ====== Move ======
    rng, rng_ = jax.random.split(rng)
    proposal, density_state = smc_problem.markov_kernel_apply(
        x_prev=samples, t_new=t_new, t_prev=t_prev, density_state=density_state
    )
    samples_just_before_resampling = proposal.sample(rng_, num_particles)

    # ====== Reweight ======
    lw_incr, density_state = smc_problem.reweighter(
        x_new=samples_just_before_resampling,
        x_prev=samples,
        t_new=t_new,
        t_prev=t_prev,
        density_state=density_state,
    )
    log_weights_just_before_resampling = jax.nn.log_softmax(log_weights + lw_incr)
    log_normaliser_increment = log_sum_exp(jax.nn.log_softmax(log_weights) + lw_incr)

    # ====== Resample and MCMC======

    rng, rng1, rng2 = jax.random.split(rng, 3)
    resample_result = optionally_resample(
        rng=rng1,
        samples=samples_just_before_resampling,
        log_weights=log_weights_just_before_resampling,
        ess_threshold=ess_threshold,
    )

    @check_shapes("samples: [b, d]")
    def MCMC_steps(rng: Key, samples: Array, density_state: int):
        MCMC_kernel = smc_problem.get_MCMC_kernel(t_new, mcmc_step_size)
        keys = jax.random.split(rng, num_mcmc_steps)
        (samples, density_state), acceptance_rates = jax.lax.scan(
            MCMC_kernel, (samples, density_state), keys
        )
        acceptance_rate = jnp.mean(acceptance_rates)
        return samples, acceptance_rate, density_state

    N = samples_just_before_resampling.shape[0]
    mcmc_steps = lambda x: MCMC_steps(*x)
    no_mcmc_steps = lambda x: (x[1], 1.0, x[2])
    samples_new, accept_ratio, density_state = jax.lax.cond(
        (essl(log_weights_just_before_resampling) < ess_threshold * N),
        mcmc_steps,
        no_mcmc_steps,
        (rng2, resample_result["samples"], density_state),
    )
    log_weights_new = resample_result["lw"]

    return {
        "samples_new": samples_new,
        "log_weights_new": log_weights_new,
        "log_normaliser_increment": log_normaliser_increment,
        "acceptance_ratio": accept_ratio,
        "resampled": resample_result["resampled"],
    }, density_state


def get_short_inner_loop(
    smc_problem: SMCProblem,
    num_particles: int,
    ess_threshold: float,
    num_mcmc_steps: int,
    mcmc_step_size_scheduler: Callable,
):
    """Wraps the inner loop with partial fillment of arguments"""

    @check_shapes("t_new: []", "t_prev: []", "samples: [b, d]", "log_weights: [b]")
    def short_inner_loop(
        rng: Key,
        t_new: Array,
        t_prev: Array,
        samples: Array,
        log_weights: Array,
        density_state: int,
    ):
        return inner_loop(
            rng=rng,
            smc_problem=smc_problem,
            t_new=t_new,
            t_prev=t_prev,
            samples=samples,
            log_weights=log_weights,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            num_mcmc_steps=num_mcmc_steps,
            mcmc_step_size=mcmc_step_size_scheduler(t_new),
            density_state=density_state,
        )

    return short_inner_loop


def outer_loop_smc(
    rng: Key,
    smc_problem: SMCProblem,
    num_particles: int,
    ess_threshold: float,
    num_mcmc_steps: int,
    mcmc_step_size_scheduler: Callable,
    density_state: int,
    progress_bar: bool = False,
) -> tp.Dict:
    """
    Runs PDDS with various diagnostics logged.

    Returns
        Dict:
            "samples": samples from target,
            "log_weights": log weights for samples,
            "log_normalising_constant": logZ estimate,
            "ess_log": log of ESS at each algorithm step,
            "acceptance_log": log of MCMC acceptance rate (1.0 if no MCMC steps performed) at each step of algorithm,
            "logZ_incr_log": log of logZ increment at each algorithm step (should be close to 1.0 when the potential function is exact),
            "initial_ess": initial ESS (can help diagnose whether forward SDE has converged),
            "num_resample_steps": number of total resampling steps required, less is better.
        Int: updated density state
    Warnings
    The returned samples are weighted and a resampling should be called before plotting.
    """
    ess_log = np.zeros((smc_problem.num_steps + 1))
    acceptance_log = np.zeros((smc_problem.num_steps + 1))
    logZ_incr_log = np.zeros((smc_problem.num_steps + 1))
    num_resample_steps = 0

    rng, rng_ = jax.random.split(rng)
    x = smc_problem.initial_distribution.sample(rng_, num_particles)
    lw_unnorm, density_state = smc_problem.initial_reweighter(x, density_state)
    lw = jax.nn.log_softmax(lw_unnorm)
    logZ = log_sum_exp(lw_unnorm) - jnp.log(num_particles)
    initial_ess = essl(lw)
    rng, rng_ = jax.random.split(rng)
    initial_resample = optionally_resample(rng_, lw, x, ess_threshold)
    if initial_resample["resampled"]:
        num_resample_steps += 1
    x = initial_resample["samples"]
    lw = initial_resample["lw"]

    ess = essl(lw)
    ess_log[0] = ess
    acceptance_log[0] = 1.0
    logZ_incr_log[0] = 0.0

    inner_loop_jit = jax.jit(
        get_short_inner_loop(
            smc_problem=smc_problem,
            num_particles=num_particles,
            ess_threshold=ess_threshold,
            num_mcmc_steps=num_mcmc_steps,
            mcmc_step_size_scheduler=mcmc_step_size_scheduler,
        )
    )

    ts = jnp.linspace(0.0, smc_problem.tf, smc_problem.num_steps + 1)
    t1 = jnp.flip(ts[:-1])
    t2 = jnp.flip(ts[1:])

    keys = jax.random.split(rng, smc_problem.num_steps)

    for i, (t_new, t_prev) in tqdm(enumerate(zip(t1, t2)), disable=(not progress_bar)):
        rng_ = keys[i]
        inner_loop_result, density_state = inner_loop_jit(
            rng_,
            t_new=t_new,
            t_prev=t_prev,
            samples=x,
            log_weights=lw,
            density_state=density_state,
        )
        x = inner_loop_result["samples_new"]
        lw = inner_loop_result["log_weights_new"]
        logZ_incr_log[i + 1] = inner_loop_result["log_normaliser_increment"]
        logZ += inner_loop_result["log_normaliser_increment"]
        ess = essl(lw)
        ess_log[i + 1] = ess
        if jnp.isnan(inner_loop_result["acceptance_ratio"]):
            acceptance_log[i + 1] = 1.0
        else:
            acceptance_log[i + 1] = inner_loop_result["acceptance_ratio"]
        if inner_loop_result["resampled"]:
            num_resample_steps += 1

    return {
        "samples": x,
        "log_weights": lw,
        "log_normalising_constant": logZ,
        "ess_log": ess_log,
        "acceptance_log": acceptance_log,
        "logZ_incr_log": logZ_incr_log,
        "initial_ess": initial_ess,
        "num_resample_steps": num_resample_steps,
    }, density_state


def fast_outer_loop_smc(
    rng: Key,
    smc_problem: SMCProblem,
    num_particles: int,
    ess_threshold: float,
    num_mcmc_steps: int,
    mcmc_step_size_scheduler: Callable,
    density_state: int,
) -> tp.Dict:
    """Fast run of PDDS algorithm without diagnostics. Used for generating training samples.
    Returns:
        Dict:
            "samples": samples from target,
            "log_weights": log weights for samples,
            "log_normalising_constant": log normalising constant estimate,
        Int: updated density state.
    """
    num_steps = smc_problem.num_steps
    rng, rng_ = jax.random.split(rng)
    x = smc_problem.initial_distribution.sample(rng_, num_particles)
    lw_unnorm, density_state = smc_problem.initial_reweighter(x, density_state)
    lw = jax.nn.log_softmax(lw_unnorm)
    logZ = log_sum_exp(lw_unnorm) - jnp.log(num_particles)
    rng, rng_ = jax.random.split(rng)
    initial_resample = optionally_resample(rng_, lw, x, ess_threshold)
    x = initial_resample["samples"]
    lw = initial_resample["lw"]

    short_inner_loop = get_short_inner_loop(
        smc_problem=smc_problem,
        num_particles=num_particles,
        ess_threshold=ess_threshold,
        num_mcmc_steps=num_mcmc_steps,
        mcmc_step_size_scheduler=mcmc_step_size_scheduler,
    )

    keys = jax.random.split(rng, num_steps)

    def scan_step(passed_state, per_step_input):
        x, lw, density_state = passed_state
        t_new, t_prev, current_key = per_step_input
        inner_loop_result, density_state = short_inner_loop(
            current_key,
            t_new=t_new,
            t_prev=t_prev,
            samples=x,
            log_weights=lw,
            density_state=density_state,
        )
        new_passed_state = (
            inner_loop_result["samples_new"],
            inner_loop_result["log_weights_new"],
            density_state,
        )
        log_z_increment = inner_loop_result["log_normaliser_increment"]
        return new_passed_state, log_z_increment

    init_state = (x, lw, density_state)
    ts = jnp.linspace(0.0, smc_problem.tf, smc_problem.num_steps + 1)
    t1 = jnp.flip(ts[:-1])
    t2 = jnp.flip(ts[1:])
    per_step_inputs = (t1, t2, keys)
    final_state, log_normalizer_increments = jax.lax.scan(
        scan_step, init_state, per_step_inputs
    )

    logZ += jnp.sum(log_normalizer_increments)

    return {
        "samples": final_state[0],
        "log_weights": final_state[1],
        "log_normalising_constant": logZ,
    }, final_state[2]
