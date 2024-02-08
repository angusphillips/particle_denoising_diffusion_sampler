import abc
import pickle
from check_shapes import check_shapes

import jax
import jax.numpy as jnp
import jax.scipy.linalg as slinalg
from jax.scipy.special import logsumexp
import numpy as np
import numpyro
from jax.flatten_util import ravel_pytree
from pdds.utils.more_utils import load_data

from jaxtyping import Float as f, PRNGKeyArray, Array, install_import_hook
import typing as tp

from chex import assert_axis_dimension, assert_shape, assert_equal_shape

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.utils import cp_utils

Key = PRNGKeyArray


class Distribution(metaclass=abc.ABCMeta):
    def __init__(self, dim: int, is_target: bool):
        self.dim = dim
        self.is_target = (
            is_target  # marks whether or not to incrememt density_state on call
        )

    @abc.abstractmethod
    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        """Produce samples from the specified distribution.

        Args:
            key: Jax key
            num_samples: int, number of samples to draw.
        Returns:
            Array of shape (num_samples, dim) containing samples from the distribution.
        """

    @abc.abstractmethod
    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        """Evaluate the log likelihood of the specified distribution.

        Args:
            x: Samples.
            density_state: int, tracks number of density evaluations.
        Returns:
            Array of shape (batch_size,) containing values of log densities.
            Int containing updated density state.
        """


class WhitenedDistributionWrapper(Distribution):
    """Reparametrizes the target distribution based on the
    learnt variational reference measure."""

    @check_shapes("vi_means: [d]", "vi_scales: [d]")
    def __init__(
        self,
        target: Distribution,
        vi_means: Array,
        vi_scales: Array,
        is_target: bool = False,
    ):
        super().__init__(dim=target.dim, is_target=is_target)
        self.target = target
        self.vi_means = vi_means
        self.vi_scales = vi_scales
        # for GaussianRatioAnalyticPotential (only works when dim=1)
        if target.dim == 1:
            self._mean = vi_means[0]
            self._scale = vi_scales[0]

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        transformed_x = self.vi_means + x * self.vi_scales
        out, density_state = self.target.evaluate_log_density(
            transformed_x, density_state
        )
        out += jnp.log(jnp.prod(self.vi_scales))
        return out, density_state

    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        original_samples = self.target.sample(key, num_samples)
        return (original_samples - self.vi_means) / self.vi_scales


class NormalDistribution(Distribution):
    """Multivariate normal distribution with shared diagonal covariance only. Scale
    is a scalar value giving the shared scale for the diagonal covariance."""

    @check_shapes("mean: [b, d]")
    def __init__(
        self,
        mean: Array,
        scale: tp.Union[float, f[Array, ""]],
        dim: int = 1,
        is_target: bool = False,
    ):
        super().__init__(dim, is_target)
        assert_axis_dimension(mean, 1, dim)
        self._mean = mean
        self._scale = scale

    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        batched_sample_shape = (num_samples,) + (self.dim,)
        self._cov_matrix = self._scale**2 * jnp.eye(self.dim)
        samples = jax.random.multivariate_normal(
            key=key, mean=self._mean, cov=self._cov_matrix, shape=(num_samples,)
        )
        assert_shape(samples, batched_sample_shape)
        return samples

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        out = jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=self._mean, cov=self._scale**2
        )
        density_state += self.is_target * x.shape[0]
        return out, density_state


class BatchedNormalDistribution(Distribution):
    """Same as NormalDistribution above except that the scale is also batched as well as the mean, i.e.
    scale is a [b] dimensional vector with the unique shared scale for each sample in the batch.
    """

    @check_shapes("means: [b, d]", "scales: [b]")
    def __init__(
        self, means: Array, scales: Array, dim: int = 1, is_target: bool = False
    ):
        super().__init__(dim, is_target)
        assert_axis_dimension(means, 1, dim)
        self._mean = means
        self._scale = scales

    @check_shapes("return: [..., d]")
    def sample(self, key: Key, num_samples: tp.Tuple) -> Array:
        batched_sample_shape = (*num_samples,) + (self.dim,)
        self._cov_matrix = self._scale[..., None, None] ** 2 * jnp.tile(
            jnp.eye(self.dim)[None, ...], (*num_samples, 1, 1)
        )
        samples = jax.random.multivariate_normal(
            key=key, mean=self._mean, cov=self._cov_matrix, shape=(*num_samples,)
        )
        assert_shape(samples, batched_sample_shape)
        return samples

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        self._cov_matrix = self._scale[..., None, None] ** 2 * jnp.tile(
            jnp.eye(self.dim)[None, ...], (x.shape[0], 1, 1)
        )
        out = jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=self._mean, cov=self._cov_matrix
        )
        density_state += self.is_target * x.shape[0]
        return out, density_state


class MeanFieldNormalDistribution(Distribution):
    """Multivariate normal distribution with diagonal covariance (non-isotropic). Scales
    is a vector value giving the scales which go on the diagonal of the covariance."""

    @check_shapes("mean: [d]", "scales: [d]")
    def __init__(
        self, mean: Array, scales: Array, dim: int = 1, is_target: bool = False
    ):
        super().__init__(dim, is_target)
        assert_axis_dimension(mean, 0, dim)
        assert_axis_dimension(scales, 0, dim)
        self._mean = mean
        self._scales = scales

    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        batched_sample_shape = (num_samples,) + (self.dim,)
        self._cov_matrix = jnp.diag(self._scales**2)
        samples = jax.random.multivariate_normal(
            key=key, mean=self._mean, cov=self._cov_matrix, shape=(num_samples,)
        )
        assert_shape(samples, batched_sample_shape)
        return samples

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        out = jax.scipy.stats.multivariate_normal.logpdf(
            x, mean=self._mean, cov=jnp.diag(self._scales**2)
        )
        density_state += self.is_target * x.shape[0]
        return out, density_state


def NormalDistributionWrapper(
    mean: float, scale: float, dim: int = 1, is_target: bool = False
) -> Distribution:
    """Wraps the NormalDistribution class for easy initialisation from Hydra configs."""
    means = mean * jnp.ones((1, dim))
    return NormalDistribution(means, scale, dim, is_target)


class ChallengingTwoDimensionalMixture(Distribution):
    """A challenging mixture of Gaussians in two dimensions.
    From annealed_flow_transport codebase.
    """

    def __init__(self, dim: int = 2, is_target: bool = False):
        super().__init__(dim, is_target)
        self.n_components = 6
        self.mean_a = jnp.array([3.0, 0.0])
        self.mean_b = jnp.array([-2.5, 0.0])
        self.mean_c = jnp.array([2.0, 3.0])
        self.means = jnp.stack((self.mean_a, self.mean_b, self.mean_c), axis=0)
        self.cov_a = jnp.array([[0.7, 0.0], [0.0, 0.05]])
        self.cov_b = jnp.array([[0.7, 0.0], [0.0, 0.05]])
        self.cov_c = jnp.array([[1.0, 0.95], [0.95, 1.0]])
        self.covs = jnp.stack((self.cov_a, self.cov_b, self.cov_c), axis=0)
        self.all_means = jnp.array(
            [[3.0, 0.0], [0.0, 3.0], [-2.5, 0.0], [0.0, -2.5], [2.0, 3.0], [3.0, 2.0]]
        )
        self.all_covs = jnp.array(
            [
                [[0.7, 0.0], [0.0, 0.05]],
                [[0.05, 0.0], [0.0, 0.7]],
                [[0.7, 0.0], [0.0, 0.05]],
                [[0.05, 0.0], [0.0, 0.7]],
                [[1.0, 0.95], [0.95, 1.0]],
                [[1.0, 0.95], [0.95, 1.0]],
            ]
        )

    @check_shapes("x: [d]", "return: []")
    def raw_log_density(self, x: Array) -> Array:
        """A raw log density that we will then symmetrize."""
        log_weights = jnp.log(jnp.array([1.0 / 3, 1.0 / 3.0, 1.0 / 3.0]))
        l = jnp.linalg.cholesky(self.covs)
        y = slinalg.solve_triangular(l, x[None, :] - self.means, lower=True, trans=0)
        mahalanobis_term = -1 / 2 * jnp.einsum("...i,...i->...", y, y)
        n = self.means.shape[-1]
        normalizing_term = -n / 2 * np.log(2 * np.pi) - jnp.log(
            l.diagonal(axis1=-2, axis2=-1)
        ).sum(axis=1)
        individual_log_pdfs = mahalanobis_term + normalizing_term
        mixture_weighted_pdfs = individual_log_pdfs + log_weights
        return logsumexp(mixture_weighted_pdfs)

    @check_shapes("x: [d]", "return: []")
    def make_2d_invariant(self, log_density, x: Array) -> Array:
        density_a = log_density(x)
        density_b = log_density(np.flip(x))
        return jnp.logaddexp(density_a, density_b) - jnp.log(2)

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        density_func = lambda x: self.make_2d_invariant(self.raw_log_density, x)
        out = jax.vmap(density_func)(x)
        density_state += self.is_target * x.shape[0]
        return out, density_state

    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        batched_sample_shape = (num_samples,) + (self.dim,)
        subkey1, subkey2 = jax.random.split(key)
        components = jax.random.choice(
            subkey1, a=int(self.n_components), shape=(num_samples,)
        ).astype(int)
        means = self.all_means[components]
        covs = self.all_covs[components]
        samples = jax.random.multivariate_normal(
            key=subkey2, mean=means, cov=covs, shape=(num_samples,)
        )
        assert_shape(samples, batched_sample_shape)
        return samples


# TODO write my own and test
class FunnelDistribution(Distribution):
    """The funnel distribution from https://arxiv.org/abs/physics/0009028.
    evaluate_log_density implementation from AFT.
    """

    def __init__(self, dim: int = 10, is_target: bool = False):
        super().__init__(dim, is_target)

    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        key, subkey1, subkey2 = jax.random.split(key, 3)
        shape1 = (num_samples,) + (1,)
        shape2 = (num_samples,) + (self.dim - 1,)
        x1 = 3 * jax.random.normal(subkey1, shape=shape1)
        scale_rest = jnp.sqrt(jnp.exp(x1))
        scale_rest = jnp.tile(scale_rest, (1, self.dim - 1))
        x2_n = scale_rest * jax.random.normal(subkey2, shape=shape2)
        samples = jnp.concatenate([x1, x2_n], axis=-1)
        return samples

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        def unbatched(x):
            v = x[0]
            log_density_v = jax.scipy.stats.norm.logpdf(v, loc=0.0, scale=3.0)
            variance_other = jnp.exp(v)
            other_dim = self.dim - 1
            cov_other = jnp.eye(other_dim) * variance_other
            mean_other = jnp.zeros(other_dim)
            log_density_other = jax.scipy.stats.multivariate_normal.logpdf(
                x[1:], mean=mean_other, cov=cov_other
            )
            assert_equal_shape([log_density_v, log_density_other])
            return log_density_v + log_density_other

        output = jax.vmap(unbatched)(x)
        density_state += self.is_target * x.shape[0]
        return output, density_state


class LogGaussianCoxPines(Distribution):
    """Log Gaussian Cox process posterior in 2D for pine saplings data.

    This follows Heng et al 2020 https://arxiv.org/abs/1708.08396 .

    config.file_path should point to a csv file of num_points columns
    and 2 rows containg the Finnish pines data.

    config.use_whitened is a boolean specifying whether or not to use a
    reparameterization in terms of the Cholesky decomposition of the prior.
    See Section G.4 of https://arxiv.org/abs/2102.07501 for more detail.
    The experiments in the paper have this set to False.

    num_dim should be the square of the lattice sites per dimension.
    So for a 40 x 40 grid num_dim should be 1600.

    Implementation from https://github.com/deepmind/annealed_flow_transport/tree/master
    """

    def __init__(
        self,
        file_path: str,
        use_whitened: bool,
        dim: int = 1600,
        is_target: bool = False,
    ):
        super().__init__(dim, is_target=is_target)

        # Discretization is as in Controlled Sequential Monte Carlo
        # by Heng et al 2017 https://arxiv.org/abs/1708.08396
        self._num_latents = dim
        self._num_grid_per_dim = int(np.sqrt(dim))

        bin_counts = jnp.array(
            cp_utils.get_bin_counts(
                self.get_pines_points(file_path), self._num_grid_per_dim
            )
        )

        self._flat_bin_counts = jnp.reshape(bin_counts, (self._num_latents))

        # This normalizes by the number of elements in the grid
        self._poisson_a = 1.0 / self._num_latents
        # Parameters for LGCP are as estimated in Moller et al, 1998
        # "Log Gaussian Cox processes" and are also used in Heng et al.

        self._signal_variance = 1.91
        self._beta = 1.0 / 33

        self._bin_vals = cp_utils.get_bin_vals(self._num_grid_per_dim)

        def short_kernel_func(x, y):
            return cp_utils.kernel_func(
                x, y, self._signal_variance, self._num_grid_per_dim, self._beta
            )

        self._gram_matrix = cp_utils.gram(short_kernel_func, self._bin_vals)
        self._cholesky_gram = jnp.linalg.cholesky(self._gram_matrix)
        self._white_gaussian_log_normalizer = (
            -0.5 * self._num_latents * jnp.log(2.0 * jnp.pi)
        )

        half_log_det_gram = jnp.sum(jnp.log(jnp.abs(jnp.diag(self._cholesky_gram))))
        self._unwhitened_gaussian_log_normalizer = (
            -0.5 * self._num_latents * jnp.log(2.0 * jnp.pi) - half_log_det_gram
        )
        # The mean function is a constant with value mu_zero.
        self._mu_zero = jnp.log(126.0) - 0.5 * self._signal_variance

        if use_whitened:
            self._posterior_log_density = self.whitened_posterior_log_density
        else:
            self._posterior_log_density = self.unwhitened_posterior_log_density

    def get_pines_points(self, file_path):
        """Get the pines data points."""
        with open(file_path, mode="rt") as input_file:
            # with open(file_path, "rt") as input_file:
            b = np.genfromtxt(input_file, delimiter=",", skip_header=1, usecols=(1, 2))
        return b

    def whitened_posterior_log_density(self, white: Array) -> Array:
        quadratic_term = -0.5 * jnp.sum(white**2)
        prior_log_density = self._white_gaussian_log_normalizer + quadratic_term
        latent_function = cp_utils.get_latents_from_white(
            white, self._mu_zero, self._cholesky_gram
        )
        log_likelihood = cp_utils.poisson_process_log_likelihood(
            latent_function, self._poisson_a, self._flat_bin_counts
        )
        return prior_log_density + log_likelihood

    def unwhitened_posterior_log_density(self, latents: Array) -> Array:
        white = cp_utils.get_white_from_latents(
            latents, self._mu_zero, self._cholesky_gram
        )
        prior_log_density = (
            -0.5 * jnp.sum(white * white) + self._unwhitened_gaussian_log_normalizer
        )
        log_likelihood = cp_utils.poisson_process_log_likelihood(
            latents, self._poisson_a, self._flat_bin_counts
        )
        return prior_log_density + log_likelihood

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        # import pdb; pdb.set_trace()
        if len(x.shape) == 1:
            density_state += self.is_target
            return self._posterior_log_density(x), density_state
        else:
            density_state += self.is_target * x.shape[0]
            return jax.vmap(self._posterior_log_density)(x), density_state

    def sample(self, key: Key, num_samples: int) -> Array:
        return NotImplementedError("LGCP target cannot be sampled")


class BayesianLogisticRegression(Distribution):
    """Evalute the unnormalised log posterior
    for a bayesian logistic regression model:
    theta \sim N(0, I)
    y | x, theta \sim Bernoulli(sigmoid(theta^T x))

    Implementation adapted from Denoising Diffusion Samplers
    https://arxiv.org/pdf/2302.13834.pdf
    """

    def __init__(self, file_path: str, is_target: bool = False):
        def model(y_obs):
            w = numpyro.sample(
                "weights", numpyro.distributions.Normal(np.zeros(dim), np.ones(dim))
            )
            logits = jnp.dot(x, w)
            with numpyro.plate("J", n_data):
                _ = numpyro.sample(
                    "y", numpyro.distributions.BernoulliLogits(logits), obs=y_obs
                )

        x, y_ = load_data(file_path)
        dim = x.shape[1]
        n_data = x.shape[0]
        model_args = (y_,)

        rng_key = jax.random.PRNGKey(1)
        model_param_info, potential_fn, _, _ = numpyro.infer.util.initialize_model(
            rng_key, model, model_args=model_args
        )
        params_flat, unflattener = ravel_pytree(model_param_info[0])

        self.log_prob_model = lambda z: -1.0 * potential_fn(unflattener(z))
        dim = params_flat.shape[0]

        super().__init__(dim, is_target)

    def sample(self, key: Key, num_samples: int):
        return NotImplementedError("Logistic regression target cannot be sampled")

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        out = jax.vmap(self.log_prob_model, in_axes=0)(x)
        density_state += self.is_target * x.shape[0]
        return out, density_state


class BrownianMissingMiddleScales(Distribution):
    """Evaluates log posterior density in the Brownian motion
    missing middle scales model.
    """

    def __init__(self, dim: int = 32, is_target: bool = False):
        self.observed_locs = np.array(
            [
                0.21592641,
                0.118771404,
                -0.07945447,
                0.037677474,
                -0.27885845,
                -0.1484156,
                -0.3250906,
                -0.22957903,
                -0.44110894,
                -0.09830782,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                np.nan,
                -0.8786016,
                -0.83736074,
                -0.7384849,
                -0.8939254,
                -0.7774566,
                -0.70238715,
                -0.87771565,
                -0.51853573,
                -0.6948214,
                -0.6202789,
            ]
        ).astype(dtype=np.float32)

        super().__init__(dim, is_target)

    @check_shapes("return: [b, d]")
    def sample(self, key: Key, num_samples: int) -> Array:
        return NotImplementedError("Brownian motion posterior cannot be sampled")

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        def unbatched(x_):
            log_jacobian_term = -jnp.log(1 + jnp.exp(-x_[0])) - jnp.log(
                1 + jnp.exp(-x_[1])
            )
            x_ = x_.at[0].set(jnp.log(1 + jnp.exp(x_[0])))
            x_ = x_.at[1].set(jnp.log(1 + jnp.exp(x_[1])))
            inn_noise_prior = jax.scipy.stats.norm.logpdf(
                jnp.log(x_[0]), loc=0.0, scale=2
            ) - jnp.log(x_[0])
            obs_noise_prior = jax.scipy.stats.norm.logpdf(
                jnp.log(x_[1]), loc=0.0, scale=2
            ) - jnp.log(x_[1])
            hidden_loc_0_prior = jax.scipy.stats.norm.logpdf(
                x_[2], loc=0.0, scale=x_[0]
            )
            hidden_loc_priors = hidden_loc_0_prior
            for i in range(29):
                hidden_loc_priors += jax.scipy.stats.norm.logpdf(
                    x_[i + 3], loc=x_[i + 2], scale=x_[0]
                )
            log_prior = inn_noise_prior + obs_noise_prior + hidden_loc_priors

            inds_not_nan = np.argwhere(~np.isnan(self.observed_locs)).flatten()
            log_lik = jax.vmap(
                lambda x, y: jax.scipy.stats.norm.logpdf(y, loc=x, scale=x_[1])
            )(x_[inds_not_nan + 2], self.observed_locs[inds_not_nan])

            log_posterior = log_prior + jnp.sum(log_lik)

            return log_posterior + log_jacobian_term

        if len(x.shape) == 1:
            density_state += self.is_target
            return unbatched(x), density_state
        else:
            density_state += self.is_target * x.shape[0]
            return jax.vmap(unbatched)(x), density_state
