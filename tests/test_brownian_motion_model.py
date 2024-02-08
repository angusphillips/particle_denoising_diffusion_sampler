# Import libraries
import os

os.environ["JAX_ENABLE_X64"] = "True"

import unittest
from unittest import TestCase

import jax
import jax.numpy as jnp

from jaxtyping import install_import_hook

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.distributions import BrownianMissingMiddleScales

from utils.distributions_for_testing import BrownianMissingMiddleScalesTestClass


class test_brownian_motion_model(TestCase):
    """Test that the two implementations of the Brownian motion
     missing middle unknown scales posterior independently
    give the same log density on a random set of x values.
    """

    def test_log_pdf(self):
        target = BrownianMissingMiddleScales()
        test_target = BrownianMissingMiddleScalesTestClass()

        key = jax.random.PRNGKey(42)
        my_x = jax.random.uniform(key, (10000, target.dim))

        log_densities = target.evaluate_log_density(my_x, 0)[0]
        log_densities_test = test_target.evaluate_log_density(my_x, 0)[0]

        self.assertTrue(jnp.allclose(log_densities, log_densities_test))

        key = jax.random.PRNGKey(42)
        my_x = jax.random.normal(key, (10000, target.dim))

        log_densities = target.evaluate_log_density(my_x, 0)[0]
        log_densities_test = test_target.evaluate_log_density(my_x, 0)[0]

        self.assertTrue(jnp.allclose(log_densities, log_densities_test))


if __name__ == "__main__":
    unittest.main()
