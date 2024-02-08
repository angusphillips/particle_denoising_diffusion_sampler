# Import libraries
import unittest
from unittest import TestCase

import jax
import jax.numpy as jnp

from jaxtyping import install_import_hook

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.distributions import BayesianLogisticRegression

from utils.distributions_for_testing import BayesianLogisticRegressionTestClass


class test_logistic_regression_densities(TestCase):
    """Test that the two implementations of the Bayesian logistic regression posterior independently
    give the same log density on a random set of x values.
    """

    def test_log_pdf(self):
        target = BayesianLogisticRegression(
            "/data/ziz/not-backed-up/anphilli/pdds/data/ionosphere_full.pkl"
        )
        test_target = BayesianLogisticRegressionTestClass(
            "/data/ziz/not-backed-up/anphilli/pdds/data/ionosphere_full.pkl"
        )

        self.assertEqual(target.dim, test_target.dim)

        key = jax.random.PRNGKey(42)
        my_x = jax.random.uniform(key, (10000, target.dim))

        log_densities = target.evaluate_log_density(my_x, 0)[0]
        log_densities_test = test_target.evaluate_log_density(my_x, 0)[0]

        self.assertTrue(jnp.allclose(log_densities, log_densities_test))


if __name__ == "__main__":
    unittest.main()
