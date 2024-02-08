import jax.numpy as jnp
import jax
import optax
import tqdm

from check_shapes import check_shapes
from jaxtyping import Array, PRNGKeyArray
import typing as tp
from hydra.utils import instantiate

Key = PRNGKeyArray

import haiku as hk

from pdds.distributions import MeanFieldNormalDistribution
from pdds.ml_tools.state import TrainingState

# evaluate_log_density state is ignored during variational inference training


class VariationalLogDensity(hk.Module):
    def __init__(self, dim, dtype, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.dtype = dtype

    @check_shapes("x: [b, d]", "return: [b]")
    def __call__(self, x: Array) -> Array:
        means = hk.get_parameter(
            "means", shape=[self.dim], dtype=self.dtype, init=jnp.zeros
        )
        scales = hk.get_parameter(
            "scales", shape=[self.dim], dtype=self.dtype, init=jnp.ones
        )
        mean_field_dist = MeanFieldNormalDistribution(means, scales, self.dim)
        return mean_field_dist.evaluate_log_density(x, 0)[0]


class VariationalSampler(hk.Module):
    def __init__(self, dim, dtype, name=None):
        super().__init__(name=name)
        self.dim = dim
        self.dtype = dtype

    @check_shapes("return: [b, d]")
    def __call__(self, rng: Key, num_particles: int):
        means = hk.get_parameter(
            "means", shape=[self.dim], dtype=self.dtype, init=jnp.zeros
        )
        scales = hk.get_parameter(
            "scales", shape=[self.dim], dtype=self.dtype, init=jnp.ones
        )
        mean_field_dist = MeanFieldNormalDistribution(means, scales, self.dim)
        return mean_field_dist.sample(rng, num_particles)


def get_variational_approx(cfg, rng, target_distribution):
    """Computes a mean-field variational approximation to the target distribution"""
    learning_rate_schedule = instantiate(cfg.vi_lr_schedule)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )

    @check_shapes("x: [b, d]", "return: [b]")
    def variational_log_density(x: Array) -> Array:
        vld = VariationalLogDensity(dim=cfg.dim, dtype=jnp.float32, name="Variational")
        return vld(x)

    @check_shapes("return: [b, d]")
    def variational_sampler(sampler_rng: Key, num_particles: int) -> Array:
        v_sampler = VariationalSampler(
            dim=cfg.dim, dtype=jnp.float32, name="Variational"
        )
        return v_sampler(sampler_rng, num_particles)

    var_log_density = hk.without_apply_rng(hk.transform(variational_log_density))
    var_sampler = hk.without_apply_rng(hk.transform(variational_sampler))

    def loss_fn(params, rng, n_particles=1000):
        X = var_sampler.apply(params=params, sampler_rng=rng, num_particles=n_particles)
        diff_log_pdf = (
            var_log_density.apply(params, X)
            - target_distribution.evaluate_log_density(X, 0)[0]
        )
        loss = jnp.mean(diff_log_pdf, axis=0)
        return loss

    @check_shapes("samples: [b, d]")
    def init(samples: Array, key: Key) -> TrainingState:
        initial_params = var_log_density.init(None, samples)
        initial_opt_state = optimizer.init(initial_params)
        return TrainingState(
            params=initial_params,
            params_ema=initial_params,
            opt_state=initial_opt_state,
            key=key,
            step=0,
        )

    @jax.jit
    def update_step(state: TrainingState) -> tp.Tuple[TrainingState, tp.Mapping]:
        new_key, loss_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn)
        loss_value, grads = loss_and_grad_fn(state.params, loss_key, 1000)
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * cfg.vi_optim.ema_rate
            + p * (1.0 - cfg.vi_optim.ema_rate),
            state.params_ema,
            new_params,
        )
        new_state = TrainingState(
            params=new_params,
            params_ema=new_params_ema,
            opt_state=new_opt_state,
            key=new_key,
            step=state.step + 1,
        )
        metrics = {"loss": loss_value, "step": state.step}
        return new_state, metrics

    samples = jnp.ones((cfg.vi_optim.batch_size, cfg.dim))
    state = init(samples=samples, key=rng)

    progress_bar = tqdm.tqdm(
        list(range(1, cfg.vi_optim.num_steps + 1)),
        miniters=1,
        disable=(not cfg.progress_bars),
    )
    for step in progress_bar:
        state, metrics = update_step(state)
        if jnp.isnan(metrics["loss"]).any():
            print("loss is nan")
            break
        metrics["lr"] = learning_rate_schedule(step)

        if step % 100 == 0:
            progress_bar.set_description(f"loss {metrics['loss']:.2f}")

    return state.params
