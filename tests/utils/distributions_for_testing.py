import jax
import jax.numpy as jnp
import pickle
import inference_gym.using_jax as gym


from jaxtyping import PRNGKeyArray, Array
from check_shapes import check_shapes
import typing as tp

Key = PRNGKeyArray

# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.distributions import Distribution, NormalDistributionWrapper


class BayesianLogisticRegressionTestClass(Distribution):
    def __init__(self, file_path: str, is_target: bool = False):
        # load data
        with open(file_path, mode="rb") as f:
            u, y = pickle.load(f)

        # pre-processing
        y = (y + 1) // 2
        mean = jnp.mean(u, axis=0)
        std = jnp.std(u, axis=0)
        std = std.at[std == 0.0].set(1.0)
        u = (u - mean) / std
        # Add column for intercept term
        extra = jnp.ones((u.shape[0], 1))
        u = jnp.hstack([extra, u])
        dim = u.shape[1]

        super().__init__(dim, is_target)

        self.y = y
        self.u = u

        self.prior = NormalDistributionWrapper(0.0, 1.0, self.dim)

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        def unbatched(x_):
            def log_bernoulli(u_, y_):
                log_sigmoids = -jnp.log(1 + jnp.exp(-jnp.dot(u_, x_)))
                log_1_minus_sigmoids = -jnp.log(1 + jnp.exp(jnp.dot(u_, x_)))
                return y_ * log_sigmoids + (1 - y_) * log_1_minus_sigmoids

            log_lik_terms = jax.vmap(log_bernoulli)(self.u, self.y)
            log_posterior = (
                jnp.sum(log_lik_terms)
                + self.prior.evaluate_log_density(x_[None, ...], 0)[0]
            )
            return log_posterior[0]

        density_state += self.is_target * x.shape[0]
        return jax.vmap(unbatched)(x), density_state

    def sample(self, key: Key, num_samples: int):
        return NotImplementedError("Logistic regression target cannot be sampled")


class BrownianMissingMiddleScalesTestClass(Distribution):
    """This class implements the BrownianMissingMiddle from inf gym.

    We wrap it for compatibility purposes.

    Implementation adapted from Denoising Diffusion Samplers
    https://arxiv.org/pdf/2302.13834.pdf
    """

    def __init__(self, dim: int = 32, is_target: bool = False):
        super().__init__(dim, is_target)
        self.cls = gym.targets.BrownianMotionUnknownScalesMissingMiddleObservations
        self.gym_target = gym.targets.VectorModel(
            self.cls(), flatten_sample_transformations=True
        )

    @check_shapes("x: [d]", "return: []")
    def _posterior_log_density(self, x: Array) -> Array:
        y = self.gym_target.default_event_space_bijector(x)

        lnp = self.gym_target.unnormalized_log_prob(y)
        jac = self.gym_target.default_event_space_bijector.forward_log_det_jacobian(
            x, event_ndims=1
        )
        return lnp + jac

    @check_shapes("x: [b, d]", "return[0]: [b]")
    def evaluate_log_density(
        self, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        if len(x.shape) == 1:
            density_state += self.is_target
            return self._posterior_log_density(x), density_state
        else:
            density_state += self.is_target * x.shape[0]
            return jax.vmap(self._posterior_log_density)(x), density_state

    @check_shapes("return: [b, d]")
    def sample(self, key: PRNGKeyArray, num_samples: int) -> Array:
        return NotImplementedError("Brownian motion posterior cannot be sampled")
