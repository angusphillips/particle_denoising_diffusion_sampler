from unittest import TestCase
import unittest

import typing as tp

import os

os.environ["JAX_ENABLE_X64"] = "True"

from jaxtyping import Array, Float as f

Samples = tp.Any
Values = tp.Any

import jax
import jax.numpy as jnp
import numpy as np

from jaxtyping import install_import_hook

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.smc_problem import SMCProblem
from pdds.distributions import (
    NormalDistributionWrapper,
    NormalDistribution,
)
from pdds.sde import LinearScheduler, SDE
from pdds.potentials import RatioPotential, NaivelyApproximatedPotential


class test_MCMC_steps(TestCase):
    def test_detailed_balance(self):
        # override parameters
        seed = 1
        sigma = 1.53
        dim = 12
        num_steps = 20
        num_particles = 10
        gamma = 0.6
        t0 = 0.0
        tf = 1.7

        # Instantiate target
        target_distribution = NormalDistributionWrapper(mean=5.0, scale=0.1, dim=dim)

        # Instantiate SDE

        scheduler = LinearScheduler(t_0=t0, t_f=tf, beta_0=0.12, beta_f=13)
        sde = SDE(scheduler, sigma, dim)

        # Instantiate potential approximation
        log_g0 = RatioPotential(sigma=sigma, target=target_distribution)
        approx_potential = NaivelyApproximatedPotential(base_potential=log_g0, dim=dim)

        # Make SMC problem
        smc_problem = SMCProblem(sde, approx_potential, num_steps)

        # Run SMC algorithm
        rng = jax.random.PRNGKey(seed)

        for t in np.linspace(t0, tf, num_steps):
            rng, rng1, rng2 = jax.random.split(rng, 3)
            my_x1 = jax.random.normal(rng1, shape=(num_particles, dim))
            my_x2 = jax.random.normal(rng2, shape=(num_particles, dim))

            log_acceptance_prob1 = (
                smc_problem._log_pi(t, my_x2, 0)[0]
                + smc_problem._get_MCMC_prop_dist(t, my_x2, gamma, 0)[
                    0
                ].evaluate_log_density(my_x1, 0)[0]
                - smc_problem._log_pi(t, my_x1, 0)[0]
                - smc_problem._get_MCMC_prop_dist(t, my_x1, gamma, 0)[
                    0
                ].evaluate_log_density(my_x2, 0)[0]
            )
            db1 = (
                smc_problem._log_pi(t, my_x1, 0)[0]
                + smc_problem._get_MCMC_prop_dist(t, my_x1, gamma, 0)[
                    0
                ].evaluate_log_density(my_x2, 0)[0]
                + jnp.minimum(
                    log_acceptance_prob1, jnp.zeros_like(log_acceptance_prob1)
                )
            )

            log_acceptance_prob2 = (
                smc_problem._log_pi(t, my_x1, 0)[0]
                + smc_problem._get_MCMC_prop_dist(t, my_x1, gamma, 0)[
                    0
                ].evaluate_log_density(my_x2, 0)[0]
                - smc_problem._log_pi(t, my_x2, 0)[0]
                - smc_problem._get_MCMC_prop_dist(t, my_x2, gamma, 0)[
                    0
                ].evaluate_log_density(my_x1, 0)[0]
            )
            db2 = (
                smc_problem._log_pi(t, my_x2, 0)[0]
                + smc_problem._get_MCMC_prop_dist(t, my_x2, gamma, 0)[
                    0
                ].evaluate_log_density(my_x1, 0)[0]
                + jnp.minimum(
                    log_acceptance_prob2, jnp.zeros_like(log_acceptance_prob2)
                )
            )

            self.assertTrue(jnp.allclose(db1, db2))


if __name__ == "__main__":
    unittest.main()
