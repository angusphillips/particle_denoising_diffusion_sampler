# %%
# Import libraries
import unittest
from unittest import TestCase

import jax
import jax.numpy as jnp
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

from jaxtyping import install_import_hook

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.distributions import FunnelDistribution

# %%
# Visual test of funnel distribution

# funnel_dist = FunnelDistribution()

# rng = jax.random.PRNGKey(42)
# rng, rng_ = jax.random.split(rng)
# samples = funnel_dist.sample(rng_, 10000)

# samples_df = pd.DataFrame(samples, columns=['x0', 'x1', 'x2', 'x3', 'x4', 'x5', 'x6', 'x7', 'x8', 'x9'])

# fig = plt.figure()
# ax = plt.gca()
# sns.kdeplot(samples_df, x='x0', ax=ax)
# plt.show()

# fig = plt.figure()
# ax = plt.gca()
# sns.scatterplot(samples_df, x='x0', y='x1', ax=ax)
# plt.show()

# fig = plt.figure()
# ax = plt.gca()
# sns.scatterplot(samples_df, x='x0', y='x2', ax=ax)
# plt.show()

# %%
