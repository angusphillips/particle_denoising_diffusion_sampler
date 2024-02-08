from unittest import TestCase
import unittest

import jax
import jax.numpy as jnp

from jaxtyping import install_import_hook

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.resampling import resampler
from collections import Counter


class test_resampler(TestCase):
    def test_systematic_resampler(self):
        rng = jax.random.PRNGKey(0)
        rng, rng1 = jax.random.split(rng)

        # Define a discrete target distribution
        target_weights = jax.random.uniform(rng1, shape=(5,))
        target_weights /= target_weights.sum()
        target_log_weights = jnp.log(target_weights) + 2023
        samples = jnp.expand_dims(jnp.arange(5), -1)

        # perform many resamplings
        n_tests = 10000
        rngs = jax.random.split(rng, n_tests)
        resampled = jax.vmap(
            lambda rng: resampler(rng, samples=samples, log_weights=target_log_weights)[
                "samples"
            ]
        )(rngs)

        # Count the number of times each point occurs in each resampling
        resampled_list = resampled[:, :, 0].tolist()
        sample_counts = [Counter(s) for s in resampled_list]
        # Find the difference between the max and min occurances of each point in the resamplings
        delta_occurences = []
        for i in samples:
            occs = [c[int(i)] for c in sample_counts]
            delta_occurences.append(max(occs) - min(occs))

        # If the difference in occurences is more than 1 then we certainly have simple resampling so there is an error
        self.assertEqual(
            max(delta_occurences),
            1,
            msg="detected simple resampling when systematic was requested",
        )

        # Compute the discrete resampled distribution and test it is not far from the target distribution
        all_resampled = jnp.ravel(resampled).tolist()
        n_samples_total = len(all_resampled)
        all_counts = Counter(all_resampled)
        for n, i in enumerate(samples):
            sampled = all_counts[int(i)]
            target = target_weights[n] * n_samples_total
            tol = 0.1 * target
            self.assertAlmostEqual(target, sampled, delta=tol)


if __name__ == "__main__":
    unittest.main()
