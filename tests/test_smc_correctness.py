from unittest import TestCase
import unittest

import typing as tp

import os

os.environ["JAX_ENABLE_X64"] = "True"

from jaxtyping import Array, Float as f

import jax
import jax.numpy as jnp

from jaxtyping import install_import_hook
from check_shapes import check_shapes

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.potentials import ApproxPotential, RatioPotential, BasePotential
from pdds.sde import LinearScheduler, SDE
from pdds.smc_problem import SMCProblem
from pdds.distributions import NormalDistributionWrapper
from pdds.utils.shaping import broadcast


def batch_scalar_prod(x_, v_):
    N_, d_ = x_.shape
    assert v_.shape == (d_,)
    return (x_ @ v_).reshape((N_,))


nsteps = 9
dim = 6
sample_size = 5
sigma = 0.72

# Need to create a potential function and approximation for the test - we choose a particularly wild one
rng = jax.random.PRNGKey(0)
rng, rng_ = jax.random.split(rng)
p1 = jax.random.uniform(rng_, (dim,))
rng, rng_ = jax.random.split(rng)
p2 = jax.random.uniform(rng_, (dim,))
rng, rng_ = jax.random.split(rng)
p3 = jax.random.uniform(rng_, (dim,))
rng, rng_ = jax.random.split(rng)
p4 = jax.random.uniform(rng_, (dim,))


def log_g0(x_):
    res = jnp.sin(batch_scalar_prod(x_, p1)) + jnp.log(
        jnp.abs(batch_scalar_prod(x_, p2))
    )
    return res


class PotentialInfo(ApproxPotential):
    def __init__(self, base_potential: BasePotential, dim: int, sigma: float):
        super().__init__(base_potential, dim)
        self.sigma = sigma

    @check_shapes("lbd: [b]", "x: [b, d]", "return[0]: [b]")
    def approx_log_gt(self, lbd: Array, x: Array, density_state: int) -> Array:
        def fun_0(args):
            x = args[0]
            density_state = args[3]
            return self.log_g0(x, density_state)

        def fun_other(args):
            samples = args[0]
            lbd = args[1]
            sigma = args[2]
            density_state = args[3]
            res = jnp.cos(
                batch_scalar_prod(samples, p3) + sigma + lbd
            ) + sigma**2 * jnp.log(
                jnp.abs((lbd + 0.1) * sigma * batch_scalar_prod(samples, p4))
            )
            return res, density_state

        return jax.lax.cond(
            lbd[0] == jnp.array(0.0),
            fun_0,
            fun_other,
            (x, lbd, self.sigma, density_state),
        )


class test_smc_correctness(TestCase):
    def test_smc_correctness_forwards_backwards_equal(self):
        # Instantiate SDE
        scheduler = LinearScheduler(t_0=0.0, t_f=1.0, beta_0=0.2, beta_f=9.3)
        sde = SDE(scheduler, sigma=sigma, dim=dim)

        target = NormalDistributionWrapper(1.23, 0.97, dim=dim)
        # Instantiate potential approximation
        log_g0 = RatioPotential(sigma, target)
        approx_potential = PotentialInfo(base_potential=log_g0, dim=dim, sigma=sigma)

        # Make SMC problem
        smc_problem = SMCProblem(sde, approx_potential, nsteps)

        rng = jax.random.PRNGKey(0)
        rng, rng_ = jax.random.split(rng)
        my_x = jax.random.uniform(key=rng_, shape=(nsteps + 1, sample_size, dim))

        t = jnp.linspace(0, smc_problem.tf, nsteps + 1)
        t1_rev = jnp.flip(t[:-1])
        t2_rev = jnp.flip(t[1:])
        t1 = t[:-1]
        t2 = t[1:]

        log_pdf_target = NormalDistributionWrapper(
            0.0, sigma, dim
        ).evaluate_log_density(my_x[0], 0)[0]
        log_pdf_target += log_g0._log_g0(my_x[0], 0)[0]
        for i, (t_prev, t_new) in enumerate(zip(t1, t2)):
            log_pdf_target += sde.forward_transition_dist(
                broadcast(t_new, my_x[i]), broadcast(t_prev, my_x[i]), my_x[i]
            ).evaluate_log_density(my_x[i + 1], 0)[0]

        log_pdf_target_rev = NormalDistributionWrapper(
            0.0, sigma, dim
        ).evaluate_log_density(my_x[nsteps], 0)[0]
        log_pdf_target_rev += log_g0._log_g0(my_x[0], 0)[0]
        for i, (t_new, t_prev) in enumerate(zip(t1_rev, t2_rev)):
            ind = nsteps - i
            log_pdf_target_rev += sde.reverse_transition_dist(
                broadcast(t_new, my_x[ind]), broadcast(t_prev, my_x[ind]), my_x[ind]
            ).evaluate_log_density(my_x[ind - 1], 0)[0]

        log_pdf_target_ref = smc_problem.initial_distribution.evaluate_log_density(
            my_x[nsteps], 0
        )[0]
        log_pdf_target_ref += smc_problem.initial_reweighter(
            x=my_x[nsteps], density_state=0
        )[0]
        for i, (t_new, t_prev) in enumerate(zip(t1_rev, t2_rev)):
            log_pdf_target_ref += smc_problem.markov_kernel_apply(
                my_x[nsteps - i], t_new, t_prev, 0
            )[0].evaluate_log_density(my_x[nsteps - i - 1], 0)[0]
            log_pdf_target_ref += smc_problem.reweighter(
                my_x[nsteps - i - 1], my_x[nsteps - i], t_new, t_prev, 0
            )[0]

        self.assertTrue(jnp.allclose(log_pdf_target_ref, log_pdf_target_rev))
        self.assertTrue(jnp.allclose(log_pdf_target_ref, log_pdf_target))

    def test_alternative_identity(self):
        # Instantiate SDE
        scheduler = LinearScheduler(t_0=0.0, t_f=1.0, beta_0=0.2, beta_f=9.3)
        sde = SDE(scheduler, sigma=sigma, dim=dim)

        target = NormalDistributionWrapper(1.23, 0.97, dim=dim)
        # Instantiate potential approximation
        log_g0 = RatioPotential(sigma, target)
        approx_potential = PotentialInfo(base_potential=log_g0, dim=dim, sigma=sigma)

        # Make SMC problem
        smc_problem = SMCProblem(sde, approx_potential, nsteps)

        rng = jax.random.PRNGKey(0)
        rng, rng_ = jax.random.split(rng)
        my_x = jax.random.uniform(key=rng_, shape=(nsteps + 1, sample_size, dim))

        t = jnp.linspace(0, smc_problem.tf, nsteps + 1)
        t1_rev = jnp.flip(t[:-1])
        t2_rev = jnp.flip(t[1:])

        for i, (t_new, t_prev) in enumerate(zip(t1_rev, t2_rev)):
            lw = smc_problem.reweighter(
                x_new=my_x[nsteps - i - 1],
                x_prev=my_x[nsteps - i],
                t_new=t_new,
                t_prev=t_prev,
                density_state=0,
            )[0]
            lw_a = smc_problem.tester_weight_A_(
                x_new=my_x[nsteps - i - 1],
                x_prev=my_x[nsteps - i],
                t_new=t_new,
                t_prev=t_prev,
                density_state=0,
            )[0]
            lw_b = smc_problem.tester_weight_B_(
                x_prev=my_x[nsteps - i], t_new=t_new, t_prev=t_prev, density_state=0
            )[0]
            lw = lw - (lw_a + lw_b)
            self.assertTrue(jnp.allclose(lw, 0))


if __name__ == "__main__":
    unittest.main()
