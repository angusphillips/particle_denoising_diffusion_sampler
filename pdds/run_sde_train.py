"""Run script for the trainable version of pdds"""
import os
import socket
import logging
import time

import jax
import jax.numpy as jnp

import haiku as hk
import numpy as np
import optax

import matplotlib.pyplot as plt

import tqdm

from functools import partial

from hydra.utils import instantiate, call
from omegaconf import OmegaConf
from omegaconf.dictconfig import DictConfig

from jaxtyping import PRNGKeyArray as Key, Array
import typing as tp
from check_shapes import check_shapes

from pdds.sde import SDE, dsm_loss, guidance_loss
from pdds.smc_problem import SMCProblem
from pdds.vi import get_variational_approx
from pdds.plotting import (
    detailed_sde_rollout_plot,
    generate_smc_diagnostic_plots,
    single_reverse_sde_rollout,
)
from pdds.potentials import (
    NNApproximatedPotential,
)
from pdds.utils.loggers_pl import LoggerCollection
from pdds.utils.shaping import broadcast
from pdds.utils.jax import (
    _get_key_iter,
    x_gradient_stateful_parametrised,
)
from pdds.utils.lr_schedules import loop_schedule
from pdds.ml_tools.state import (
    TrainingState,
    load_checkpoint,
    save_checkpoint,
)
from pdds import ml_tools
from pdds.smc_loops import outer_loop_smc, fast_outer_loop_smc
from pdds.distributions import (
    WhitenedDistributionWrapper,
)


def run(cfg: DictConfig):
    # resolve number of steps
    cfg.num_steps = cfg.base_steps * cfg.steps_mult

    # START-UP
    if cfg.make_logs:
        log = logging.getLogger(__name__)
        log.info("Starting up...")
        log.info(f"Jax devices: {jax.devices()}")
        run_path = os.getcwd()
        log.info(f"Run path: {run_path}")
        log.info(f"Hostname: {socket.gethostname()}")
        ckpt_path = os.path.join(run_path, cfg.ckpt_dir)
        os.makedirs(ckpt_path, exist_ok=True)
        loggers = [instantiate(logger_cfg) for logger_cfg in cfg.logging.values()]
        logger = LoggerCollection(loggers)
        logger.log_hyperparams(OmegaConf.to_container(cfg, resolve=True))
    else:
        log = None
        logger = None

    # INSTANTIATE OBJECTS

    # Instantiate key iterator
    key = jax.random.PRNGKey(cfg.seed)
    key_iter = _get_key_iter(key)

    # Instantiate target
    target_distribution = call(cfg.target)

    # Instantiate SDE
    scheduler = instantiate(cfg.scheduler)
    sde = SDE(scheduler, cfg.sigma, cfg.dim)

    # Learn variational approximation of reference distribution
    if cfg.use_vi_approx:
        log.info("Learning VI approximation")
        key, key_ = jax.random.split(key)
        vi_params = get_variational_approx(cfg, key_, target_distribution)
        target_distribution = WhitenedDistributionWrapper(
            target_distribution,
            vi_params["Variational"]["means"],
            vi_params["Variational"]["scales"],
        )

    # Instantiate potential classes
    log_g0 = instantiate(cfg.log_g0, target=target_distribution)
    uncorrected_approx_potential = instantiate(cfg.potential, base_potential=log_g0)

    mcmc_step_size_scheduler = instantiate(cfg.mcmc_step_size)

    # Instantiate neural network potential approximator
    @hk.without_apply_rng
    @hk.transform
    @check_shapes("lbd: [b]", "x: [b, d]")
    def nn_potential_approximator(lbd: Array, x: Array, density_state: int):
        std_trick = False
        std = None
        residual, density_state = uncorrected_approx_potential.approx_log_gt(
            lbd=lbd, x=x, density_state=density_state
        )

        net = instantiate(
            cfg.network,
            dim=cfg.dim,
        )

        out = net(lbd, x, residual)

        if std_trick:
            out = out / (std + 1e-3)

        return out, density_state

    # Instantiate approximate density and potential functions
    @jax.jit
    @check_shapes("lbd: [b]", "x: [b, d]", "return[0]: [b]")
    def log_pi(
        params, lbd: Array, x: Array, density_state: int
    ) -> tp.Tuple[Array, int]:
        reference_term, _ = sde.reference_dist.evaluate_log_density(
            x=x, density_state=0
        )
        nn_approx, density_state = nn_potential_approximator.apply(
            params, lbd, x, density_state
        )
        return nn_approx + reference_term, density_state

    grad_log_pi = jax.jit(x_gradient_stateful_parametrised(log_pi))

    @jax.jit
    @check_shapes("lbd: [b]", "x: [b, d]")
    def grad_log_g(params, lbd: Array, x: Array, density_state: int):
        return x_gradient_stateful_parametrised(nn_potential_approximator.apply)(
            params, lbd, x, density_state
        )

    # Define loss functions
    @check_shapes("samples: [b, d]")
    def dsm_loss_fn(params, samples: Array, key: Key, density_state: int):
        return dsm_loss(
            key,
            sde,
            partial(grad_log_pi, params),
            samples,
            density_state,
            likelihood_weight=cfg.optim.likelihood_weight,
            sample_lbd=cfg.optim.sample_lbd,
        )

    @check_shapes("samples: [b, d]")
    def guidance_loss_fn(params, samples: Array, key: Key, density_state: int):
        return guidance_loss(
            key,
            sde,
            partial(grad_log_g, params),
            samples,
            density_state,
            log_g0,
            cfg.optim.sample_lbd,
        )

    loss_fn = guidance_loss_fn if cfg.loss == "guidance" else dsm_loss_fn

    # Instantiate learning rate schedulers
    if (
        cfg.lr_schedule._target_ == "optax.warmup_cosine_decay_schedule"
    ):  # If using multiple iterations of potential approximation, reset the lr schedule for each iteration.
        learning_rate_schedule_unlooped = instantiate(
            cfg.lr_schedule,
            warmup_steps=min(1000, cfg.optim.refresh_model_every // 20),
            decay_steps=cfg.optim.refresh_model_every,
        )
        learning_rate_schedule = loop_schedule(
            schedule=learning_rate_schedule_unlooped, freq=cfg.optim.refresh_model_every
        )
    if cfg.lr_schedule._target_ == "optax.exponential_decay":
        learning_rate_schedule_unlooped = instantiate(cfg.lr_schedule)
        learning_rate_schedule = loop_schedule(
            schedule=learning_rate_schedule_unlooped, freq=cfg.optim.refresh_model_every
        )
    else:
        learning_rate_schedule = instantiate(cfg.lr_schedule)
    optimizer = optax.chain(
        optax.clip_by_global_norm(1.0),
        optax.scale_by_adam(),
        optax.scale_by_schedule(learning_rate_schedule),
        optax.scale(-1.0),
    )

    # define model update function
    @jax.jit
    @check_shapes("samples: [b, d]")
    def update_step(
        state: TrainingState, samples: Array, density_state: int
    ) -> tp.Tuple[TrainingState, int, tp.Mapping]:
        new_key, loss_key = jax.random.split(state.key)
        loss_and_grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
        (loss_value, density_state), grads = loss_and_grad_fn(
            state.params, samples, loss_key, density_state
        )
        updates, new_opt_state = optimizer.update(grads, state.opt_state)
        new_params = optax.apply_updates(state.params, updates)
        new_params_ema = jax.tree_util.tree_map(
            lambda p_ema, p: p_ema * cfg.optim.ema_rate
            + p * (1.0 - cfg.optim.ema_rate),
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
        return new_state, density_state, metrics

    # define haiku initialisation
    @check_shapes("samples: [b, d]")
    def init(samples: Array, key: Key) -> TrainingState:
        key, init_rng = jax.random.split(key)
        lbd = broadcast(jnp.array(1.0), samples)
        density_state = 0
        initial_params = nn_potential_approximator.init(
            init_rng, lbd, samples, density_state
        )
        initial_opt_state = optimizer.init(initial_params)
        return TrainingState(
            params=initial_params,
            params_ema=initial_params,
            opt_state=initial_opt_state,
            key=key,
            step=0,
        )

    # initialise haiku model
    initial_samples = sde.reference_dist.sample(
        jax.random.PRNGKey(cfg.seed), cfg.optim.batch_size
    )
    training_state = init(initial_samples, jax.random.PRNGKey(cfg.seed))
    if cfg.mode == "eval":  # if resume or evaluate
        training_state = load_checkpoint(training_state, ckpt_path, cfg.optim.num_steps)

    # log number of trainable parameters
    nb_params = sum(x.size for x in jax.tree_util.tree_leaves(training_state.params))
    log.info(f"Number of parameters: {nb_params}")
    logger.log_metrics({"nb_params": nb_params}, step=0)

    # Define plot callback function
    def plot(training_state, key: Key, step):
        log.info(f"Plotting at step {step}")

        plot_dict = {}
        key = jax.random.PRNGKey(cfg.seed)
        num_samples = 10_000

        if step != None:
            # Plot the SDE roll out and learned SDE path (only possible when the scheduler has beta_t available)
            if (
                "CosineScheduler" not in cfg.scheduler._target_
                and "DDSScheduler" not in cfg.scheduler._target_
            ):
                if cfg.has_ground_truth:
                    plot_dict["detailed_rollout"] = detailed_sde_rollout_plot(
                        key=key,
                        sde=sde,
                        target_distribution=target_distribution,
                        score=lambda lbd, x: grad_log_pi(
                            params=training_state.params_ema,
                            lbd=lbd,
                            x=x,
                            density_state=0,
                        )[0],
                        uncorrected_approx_potential=uncorrected_approx_potential,
                    )
                x_T_samples = sde.reference_dist.sample(key, num_samples)
                plot_dict["learned_sde_rollout"] = single_reverse_sde_rollout(
                    key=key,
                    sde=sde,
                    target_distribution=target_distribution
                    if cfg.has_ground_truth
                    else None,
                    score=lambda lbd, x: grad_log_pi(
                        params=training_state.params_ema, lbd=lbd, x=x, density_state=0
                    )[0],
                    initial_samples=x_T_samples,
                    t_f=cfg.t_f,
                    t_0=cfg.t_0,
                    n_steps=200,
                )

        if step == None:  # when step==None we evaluate the naive approximation
            corrected_approx_potential = uncorrected_approx_potential
        else:  # Otherwise get the most recent potential approximation
            corrected_approx_potential = NNApproximatedPotential(
                base_potential=log_g0,
                dim=cfg.dim,
                nn_potential_approximator=partial(
                    nn_potential_approximator.apply, params=training_state.params_ema
                ),
            )

        # Run PDDS and generate diagnostic plots
        smc_problem = SMCProblem(sde, corrected_approx_potential, cfg.num_steps)
        key, subkey = jax.random.split(key)
        smc_result, _ = outer_loop_smc(
            rng=subkey,
            smc_problem=smc_problem,
            num_particles=cfg.num_particles,
            ess_threshold=cfg.ess_threshold,
            num_mcmc_steps=cfg.num_mcmc_steps,
            mcmc_step_size_scheduler=mcmc_step_size_scheduler,
            density_state=0,
        )

        plot_dict_ = generate_smc_diagnostic_plots(
            results=smc_result,
            smc_problem=smc_problem,
            target_distribution=target_distribution,
            sde=sde,
            cfg=cfg,
        )

        plot_dict = {**plot_dict, **plot_dict_}

        return plot_dict

    # Define evaluation callback function - slow, only for evaluating at the end of each PDDS iteration
    def eval(training_state, key, step, density_state_training):
        log.info(f"Evaluating at step {step}")

        num_smc_iterations = cfg.num_smc_iters
        if step == None:  # If step==None evaluate the naive approximation
            corrected_approx_potential = uncorrected_approx_potential
        else:
            corrected_approx_potential = NNApproximatedPotential(
                base_potential=log_g0,
                dim=cfg.dim,
                nn_potential_approximator=partial(
                    nn_potential_approximator.apply, params=training_state.params_ema
                ),
            )

        # Fast jitted sampler
        smc_problem = SMCProblem(sde, corrected_approx_potential, cfg.num_steps)
        eval_sampler = jax.jit(
            partial(
                fast_outer_loop_smc,
                smc_problem=smc_problem,
                num_particles=int(cfg.num_particles),
                ess_threshold=cfg.ess_threshold,
                num_mcmc_steps=cfg.num_mcmc_steps,
                mcmc_step_size_scheduler=mcmc_step_size_scheduler,
                density_state=0,
            )
        )

        # Evaluate the normalising constant estimate using num_smc_iterations seeds.
        log_Z = np.zeros(num_smc_iterations)
        for i in tqdm.trange(num_smc_iterations, disable=(not cfg.progress_bars)):
            key, subkey = jax.random.split(key)
            smc_result, _ = eval_sampler(subkey)
            log_Z[i] = smc_result["log_normalising_constant"]

        return {
            "log_Z": np.mean(log_Z),
            "var_log_Z": np.var(log_Z),
            "density_calls": density_state_training,
        }

    # Fast evaluation function - can be called at every training iteration to show evolution of logZ during training.
    def eval_logZ(params, key, density_state_training, step):
        corrected_approx_potential = NNApproximatedPotential(
            base_potential=log_g0,
            dim=cfg.dim,
            nn_potential_approximator=partial(
                nn_potential_approximator.apply,
                params=params,
            ),
        )
        smc_problem = SMCProblem(sde, corrected_approx_potential, cfg.num_steps)
        sampler = partial(
            fast_outer_loop_smc,
            smc_problem=smc_problem,
            num_particles=int(cfg.num_particles),
            ess_threshold=cfg.ess_threshold,
            num_mcmc_steps=cfg.num_mcmc_steps,
            mcmc_step_size_scheduler=mcmc_step_size_scheduler,
        )
        keys = jax.random.split(key, cfg.num_smc_iters)
        log_Z = jax.vmap(
            lambda key: sampler(rng=key, density_state=0)[0]["log_normalising_constant"]
        )(keys)
        return {
            "running_log_Z": np.mean(log_Z),
            "running_var_log_Z": np.var(log_Z),
            "density_calls": density_state_training,
        }

    eval_logZ_jit = jax.jit(eval_logZ)

    # define callback functions
    def log_and_close_plots(step, t, **kwargs):
        plot_dict = plot(kwargs["training_state"], kwargs["key"], step)
        logger.log_plot("plots", plot_dict, step)
        plt.close("all")

    log_training_metrics = ml_tools.actions.PeriodicCallback(
        every_steps=1,
        callback_fn=lambda step, t, **kwargs: logger.log_metrics(
            kwargs["metrics"], step
        ),
    )
    save_model = ml_tools.actions.PeriodicCallback(
        every_steps=cfg.optim.refresh_model_every,
        callback_fn=lambda step, t, **kwargs: save_checkpoint(
            kwargs["training_state"], ckpt_path, step
        ),
    )
    log_evaluation_metrics = ml_tools.actions.PeriodicCallback(
        every_steps=cfg.optim.refresh_model_every,
        callback_fn=lambda step, t, **kwargs: logger.log_metrics(
            eval(
                kwargs["training_state"], kwargs["key"], step, kwargs["density_state"]
            ),
            step,
        ),
    )
    calc_logZ = ml_tools.actions.PeriodicCallback(
        every_steps=cfg.logZ_log_freq,
        callback_fn=lambda step, t, **kwargs: logger.log_metrics(
            eval_logZ_jit(
                kwargs["params"], kwargs["key"], kwargs["density_state"], step
            ),
            step,
        ),
    )
    log_plots = ml_tools.actions.PeriodicCallback(
        every_steps=cfg.optim.refresh_model_every,
        callback_fn=log_and_close_plots,
    )

    # Training loop
    if cfg.mode == "train":
        density_state_training = 0
        log.info("Training")

        # Plot and evaluate the naive approximation
        if cfg.plot_train:
            logger.log_plot(
                "naive", plot(training_state, jax.random.PRNGKey(cfg.seed), None), 0
            )
        if cfg.eval_train:
            logger.log_metrics(
                eval(
                    training_state,
                    jax.random.PRNGKey(cfg.seed),
                    None,
                    density_state_training,
                ),
                0,
            )

        refresh_batch_every = cfg.optim.refresh_batch_every
        refresh_model_every = cfg.optim.refresh_model_every

        # Initial sampler for training samples
        smc_problem = SMCProblem(sde, uncorrected_approx_potential, cfg.num_steps)
        training_sampler = jax.jit(
            partial(
                fast_outer_loop_smc,
                smc_problem=smc_problem,
                num_particles=cfg.optim.batch_size * refresh_batch_every,
                ess_threshold=cfg.ess_threshold,
                num_mcmc_steps=cfg.num_mcmc_steps,
                mcmc_step_size_scheduler=mcmc_step_size_scheduler,
            )
        )
        # initial jit compilation
        _, _ = training_sampler(rng=key, density_state=0)

        progress_bar = tqdm.tqdm(
            list(range(1, cfg.optim.num_steps + 1)),
            miniters=1,
            disable=(not cfg.progress_bars),
        )

        start_time = time.time()
        for step, key in zip(progress_bar, key_iter):
            # Generate samples for potential approximation training from PDDS with previous potential approximation
            if (
                step - 1
            ) % refresh_batch_every == 0:  # refresh samples after every 'epoch'
                jit_results, density_state_training = training_sampler(
                    rng=key, density_state=density_state_training
                )
                sample_batches = jit_results["samples"].reshape(
                    (refresh_batch_every, cfg.optim.batch_size, cfg.dim)
                )

                if jnp.any(jnp.isnan(sample_batches)):
                    log.warning("nan in sampled batches")
                    break

            samples = sample_batches[(step - 1) % refresh_batch_every]
            training_state, density_state_training, metrics = update_step(
                training_state, samples, density_state_training
            )

            if jnp.isnan(metrics["loss"]).any():
                log.warning("Loss is nan")
                break

            metrics["lr"] = learning_rate_schedule(training_state.step)

            # Call callback functions
            log_training_metrics(
                step=step,
                t=None,
                metrics=metrics,
            )
            save_model(
                step,
                t=None,
                metrics=metrics,
                training_state=training_state,
                key=key,
            )
            if cfg.plot_train:
                log_plots(
                    step=step,
                    t=None,
                    key=key,
                    training_state=training_state,
                )
            if cfg.eval_train:
                log_evaluation_metrics(
                    step=step,
                    t=None,
                    key=key,
                    training_state=training_state,
                    density_state=density_state_training,
                )
            if cfg.logZ_train:
                calc_logZ(
                    step=step,
                    t=None,
                    key=key,
                    params=training_state.params_ema,
                    density_state=density_state_training,
                )

            if step % 100 == 0:
                progress_bar.set_description(f"loss {metrics['loss']:.2f}")

            # Start next iteration of PDDS potential refinement
            if step % refresh_model_every == 0:
                corrected_approx_potential = NNApproximatedPotential(
                    base_potential=log_g0,
                    dim=cfg.dim,
                    nn_potential_approximator=partial(
                        nn_potential_approximator.apply,
                        params=training_state.params_ema,
                    ),
                )
                smc_problem = SMCProblem(sde, corrected_approx_potential, cfg.num_steps)
                training_sampler = jax.jit(
                    partial(
                        fast_outer_loop_smc,
                        smc_problem=smc_problem,
                        num_particles=cfg.optim.batch_size * refresh_batch_every,
                        ess_threshold=cfg.ess_threshold,
                        num_mcmc_steps=cfg.num_mcmc_steps,
                        mcmc_step_size_scheduler=mcmc_step_size_scheduler,
                    )
                )
                if cfg.optim.retrain_from_scratch:
                    training_state = init(initial_samples, jax.random.PRNGKey(cfg.seed))
        end_time = time.time()
        logger.log_metrics(
            {"training_time": end_time - start_time}, step=0
        )  # log training time, only a fair comparison if the callbacks are switched off.

    else:
        log_plots._cb_fn(
            cfg.optim.num_steps + 1, t=None, training_state=training_state, key=key
        )
        log_evaluation_metrics._cb_fn(
            cfg.optim.num_steps + 1,
            t=None,
            training_state=training_state,
            key=key,
            density_state=0,
        )
        logger.save()

    # Finish and exit
    if logger:
        logger.save()
        logger.finalize("success")

    return
