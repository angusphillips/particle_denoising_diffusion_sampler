# Import libraries
import os

os.environ["JAX_ENABLE_X64"] = "True"

import unittest
from unittest import TestCase

import jax
import jax.numpy as jnp

from jaxtyping import install_import_hook

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.sde import CosineScheduler
from pdds.utils.shaping import broadcast


class test_cosine_scheduler(TestCase):
    def test_end_points(self):
        """Test that lambda_{0,0} = 0 and lambda_{0,T} = 1.0"""
        T = 3.14
        batch_size = 512
        cos_scheduler = CosineScheduler(0.0, T, 0.0)

        l1 = cos_scheduler.lambda_t(jnp.zeros((batch_size)), jnp.zeros((batch_size)))
        l2 = cos_scheduler.lambda_t0(jnp.zeros((batch_size)))

        l3 = cos_scheduler.lambda_t(T * jnp.ones((batch_size)), jnp.zeros((batch_size)))
        l4 = cos_scheduler.lambda_t0(T * jnp.ones((batch_size)))

        self.assertTrue(jnp.array_equal(l1, jnp.zeros((batch_size))))
        self.assertTrue(jnp.array_equal(l2, jnp.zeros((batch_size))))
        self.assertTrue(jnp.array_equal(l3, jnp.ones((batch_size))))
        self.assertTrue(jnp.array_equal(l4, jnp.ones((batch_size))))

    def test_consistency(self):
        """Test that lambda_t0(t) = lambda_t(t, 0)
        and lambda_t(t, t-1) = 1 - (1-lambda_t)/(1-lambda_t-1)"""
        T = 2.71828
        batch_size = 512
        cos_scheduler = CosineScheduler(0.0, T, 0.0)

        self.assertTrue(
            jnp.array_equal(
                cos_scheduler.lambda_t(
                    1.34 * jnp.ones((batch_size)), jnp.zeros((batch_size))
                ),
                cos_scheduler.lambda_t0(1.34 * jnp.ones((batch_size))),
            )
        )

        lbd = cos_scheduler.lambda_t(
            1.87 * jnp.ones((batch_size)), 1.32 * jnp.ones((batch_size))
        )
        lbd_ref = 1 - (1 - cos_scheduler.lambda_t0(1.87 * jnp.ones((batch_size)))) / (
            1 - cos_scheduler.lambda_t0(1.32 * jnp.ones((batch_size)))
        )
        self.assertTrue(
            jnp.allclose(lbd, lbd_ref)
        )  # not exact since we do the lambda_t calculation on the log scale


if __name__ == "__main__":
    unittest.main()
