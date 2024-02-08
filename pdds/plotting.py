import jax
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import os
import jax.numpy as jnp
import pandas as pd

import diffrax as dfx

from omegaconf.dictconfig import DictConfig
from pdds.resampling import resampler
import typing as tp
from check_shapes import check_shapes

from jaxtyping import install_import_hook, PRNGKeyArray, Array

Key = PRNGKeyArray


# with install_import_hook("pdds", "typeguard.typechecked"):
from pdds.distributions import Distribution, NormalDistributionWrapper
from pdds.sde import SDE, sde_solve
from pdds.smc_problem import SMCProblem
from pdds.utils.jax import x_gradient
from pdds.utils.shaping import broadcast


def generate_smc_diagnostic_plots(
    results: tp.Dict,
    smc_problem: SMCProblem,
    target_distribution: Distribution,
    sde: SDE,
    cfg: DictConfig,
    show: bool = False,
    plot_sde_convergence: bool = False,
):
    if not show:
        plot_dict = {}
    key = jax.random.PRNGKey(cfg.seed)

    # sns.kdeplot becomes very slow with large data so we subsample when there are large number of particles
    if cfg.num_particles > 1e6:
        n_plot_samples = int(1e6)
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, a=int(cfg.num_particles), shape=(int(1e6),))
    else:
        n_plot_samples = int(cfg.num_particles)
        idx = jnp.arange(int(cfg.num_particles))

    key, subkey1, subkey2, subkey3 = jax.random.split(key, 4)
    reference_samples = smc_problem.initial_distribution.sample(
        subkey1, num_samples=n_plot_samples
    )
    if cfg.plot_target:
        target_samples = target_distribution.sample(subkey2, num_samples=n_plot_samples)
    final_samples = resampler(
        rng=subkey3, samples=results["samples"], log_weights=results["log_weights"]
    )["samples"]
    ts = jnp.linspace(0, smc_problem.tf, 5)[1:]
    for d in cfg.univariate_plot_dims:
        if cfg.plot_target and plot_sde_convergence:
            # Checking that the forward SDE converges.
            fig = plt.figure()
            ax = plt.gca()
            sns.kdeplot(x=target_samples[:, d], ax=ax, label="target_samples")
            for i in range(4):
                key, subkey = jax.random.split(key)
                marginal_samples = sde.forward_path_marginal_dist(
                    t=broadcast(ts[i], target_samples), x0=target_samples
                ).sample(subkey, n_plot_samples)
                sns.kdeplot(marginal_samples[:, d], ax=ax, label=f"t={ts[i]} samples")
            sns.kdeplot(reference_samples[:, d], ax=ax, label="reference_samples")
            plt.legend()
            if show:
                plt.show()
            else:
                plot_dict[f"sde_convergence_dim{d}"] = fig
            plt.close(fig)

        # Visualise target, reference and samples
        fig = plt.figure()
        ax = fig.gca()
        sns.kdeplot(final_samples[idx, d], ax=ax, label="pdds samples")
        if cfg.plot_target:
            sns.kdeplot(target_samples[:, d], ax=ax, label="target samples")
        plt.legend()
        if show:
            plt.show()
        else:
            plot_dict[f"samples_dim{d}"] = fig
        plt.close(fig)

    if len(cfg.bivariate_plot_dims) != 0:
        vars = [f"x{i}" for i in range(cfg.dim)]
        reference_samples_pd = pd.DataFrame(reference_samples, columns=vars)
        if cfg.plot_target:
            target_samples_pd = pd.DataFrame(target_samples, columns=vars)
        final_samples_pd = pd.DataFrame(final_samples, columns=vars)
        key, subkey = jax.random.split(key)
        idx = jax.random.choice(subkey, a=int(cfg.num_particles), shape=(int(1e2),))
        for [d1, d2] in cfg.bivariate_plot_dims:
            if cfg.plot_target and plot_sde_convergence:
                # Checking that the forward SDE converges.
                fig = plt.figure()
                ax = plt.gca()
                sns.scatterplot(
                    target_samples_pd.iloc[idx],
                    x=vars[d1],
                    y=vars[d2],
                    ax=ax,
                    label="target_samples",
                    alpha=0.3,
                )
                for i in range(4):
                    key, subkey = jax.random.split(key)
                    marginal_samples = sde.forward_path_marginal_dist(
                        t=broadcast(ts[i], target_samples), x0=target_samples
                    ).sample(subkey, n_plot_samples)
                    marginal_samples_pd = pd.DataFrame(marginal_samples, columns=vars)
                    sns.scatterplot(
                        marginal_samples_pd.iloc[idx],
                        x=vars[d1],
                        y=vars[d2],
                        ax=ax,
                        label=f"t={ts[i]} samples",
                        alpha=0.3,
                    )
                sns.scatterplot(
                    reference_samples_pd.iloc[idx],
                    x=vars[d1],
                    y=vars[d2],
                    ax=ax,
                    label="reference_samples",
                    alpha=0.3,
                )
                plt.legend()
                if show:
                    plt.show()
                else:
                    plot_dict[f"sde_convergence_dims{d1}x{d2}"] = fig
                plt.close(fig)

            # Visualise target, reference and samples
            fig = plt.figure()
            ax = fig.gca()
            sns.scatterplot(
                final_samples_pd.iloc[idx],
                x=vars[d1],
                y=vars[d2],
                ax=ax,
                label="pdds samples",
                alpha=0.3,
            )
            if cfg.plot_target:
                sns.scatterplot(
                    target_samples_pd.iloc[idx],
                    x=vars[d1],
                    y=vars[d2],
                    ax=ax,
                    label="target samples",
                    alpha=0.3,
                )
            plt.legend()
            if show:
                plt.show()
            else:
                plot_dict[f"samples_dims{d1}x{d2}"] = fig
            plt.close(fig)

    ess_log = results["ess_log"]
    fig = plt.figure()
    plt.plot(np.arange(len(ess_log)), ess_log, "-")
    plt.ylabel("ESS")
    plt.ylim((cfg.num_particles // 4, cfg.num_particles + 100))
    if show:
        plt.show()
    else:
        plot_dict["ess_log"] = fig
    plt.close(fig)

    lz_incr_log = results["logZ_incr_log"]
    fig = plt.figure()
    plt.plot(np.arange(len(lz_incr_log)), lz_incr_log, "-")
    plt.ylabel("logZ_incr")
    if show:
        plt.show()
    else:
        plot_dict["logZ_incr_log"] = fig
    plt.close(fig)

    accept_log = results["acceptance_log"]
    fig = plt.figure()
    ts = jnp.linspace(0.0, smc_problem.tf, smc_problem.num_steps + 1)
    ts = jnp.flip(ts)
    plt.plot(ts, accept_log, "-")
    plt.ylabel("acceptance_ratio")
    plt.xlabel("t")
    plt.gca().invert_xaxis()
    plt.ylim((0, 1.1))
    if show:
        plt.show()
    else:
        plot_dict["acceptance_log"] = fig
    plt.close(fig)

    return plot_dict


def compare_densities(
    samples, names, title="Specify title", path="./figs/", display_title=True
):
    """Plots KDE estimates of multiple densities

    Parameters:
    samples - list of arrays of samples to plot
    names - list of names for the labels
    title - title of plot
    path - path to save plot
    """
    fig = plt.figure()
    ax = plt.gca()
    for i, data in enumerate(samples):
        sns.kdeplot(data[:, 0], ax=ax, label=names[i])
    plt.legend()
    if display_title:
        plt.title(title)
    plt.savefig(os.path.join(path, title + ".png"), bbox_inches="tight")
    plt.close(fig)


def sde_rollout_plot(
    key, sde, target_distribution, score, uncorrected_approx_potential
):
    num_samples = 10_000

    ts = list(jnp.linspace(sde.scheduler.t_0, sde.scheduler.t_f, 5))
    ts_rev = list(reversed(ts))

    key, skey = jax.random.split(key)
    x_0_samples = target_distribution.sample(key, num_samples)
    key, skey = jax.random.split(key)
    x_T_samples = sde.reference_dist.sample(skey, num_samples)

    key, skey = jax.random.split(key)
    forward_sde_samples = sde_solve(
        sde=sde,
        grad_log_pi=score,
        x=x_0_samples,
        key=skey,
        prob_flow=False,
        num_steps=1000,
        solver=dfx.Euler(),
        rtol=None,
        atol=None,
        forward=True,
        ts=ts,
    )

    key, skey = jax.random.split(key)
    reverse_sde_samples = sde_solve(
        sde=sde,
        grad_log_pi=score,
        x=x_T_samples,
        key=skey,
        prob_flow=False,
        num_steps=1000,
        solver=dfx.Euler(),
        rtol=None,
        atol=None,
        forward=False,
        ts=ts_rev,
    )

    key, skey = jax.random.split(key)
    forward_analytic_samples = jax.vmap(
        lambda t: sde.forward_path_marginal_dist(
            broadcast(t, x_0_samples), x_0_samples
        ).sample(skey, num_samples)
    )(jnp.array(ts))

    key, skey = jax.random.split(key)
    uncorrected_log_pi = (
        lambda lbd, x: uncorrected_approx_potential.approx_log_gt(lbd=lbd, x=x)
        + sde.reference_dist.evaluate_log_density(x, 0)[0]
    )
    reverse_sde_naive_samples = sde_solve(
        sde=sde,
        grad_log_pi=x_gradient(uncorrected_log_pi),
        x=x_T_samples,
        key=skey,
        prob_flow=False,
        num_steps=1000,
        solver=dfx.Euler(),
        rtol=None,
        atol=None,
        forward=False,
        ts=ts_rev,
    )

    fig, axes = plt.subplots(
        1, len(ts), sharey=True, sharex=True, figsize=(3 * len(ts), 3)
    )
    # plt.setp(axes, xlim=(-6, 7))
    line = jnp.linspace(-5, 5)[:, None]
    for i, (ax, t, fwd_sde, rev_lrn, fwd_ana, rev_naive) in enumerate(
        zip(
            axes,
            ts,
            forward_sde_samples,
            reversed(reverse_sde_samples),
            forward_analytic_samples,
            reversed(reverse_sde_naive_samples),
        )
    ):
        # ax.plot(line[:, 0], jnp.exp(sde.forward_path_marginal_dist(t).evaluate_log_density(line)))
        sns.kdeplot(fwd_sde[:, 0], ax=ax, label="fwd sde", alpha=0.5)
        if i != 0:
            sns.kdeplot(fwd_ana[:, 0], ax=ax, label="fwd ana", alpha=0.5)
        sns.kdeplot(rev_naive[:, 0], ax=ax, label="rev nve", alpha=0.5)
        sns.kdeplot(rev_lrn[:, 0], ax=ax, label="rev lrn", alpha=0.5)
        if i == 0:
            sns.kdeplot(x_0_samples[:, 0], ax=ax, label="x0 sampl", alpha=0.5)
            # ax.plot(
            #     line[:, 0], jnp.exp(target_distribution.evaluate_log_density(line))
            # )
        if i == (len(ts) - 1):
            sns.kdeplot(x_T_samples[:, 0], ax=ax, label="xT sampl", alpha=0.5)
            # ax.plot(
            #     line[:, 0],
            #     jnp.exp(sde.reference_dist.evaluate_log_density(line)),
            # )

        ax.set_title(f"{t=:0.3f}")
        ax.legend(fontsize=6)

    return fig


def detailed_sde_rollout_plot(
    key, sde, target_distribution, score, uncorrected_approx_potential
):
    num_samples = 10_000

    ts = list(jnp.linspace(sde.scheduler.t_0, sde.scheduler.t_f, 10))
    ts_rev = list(reversed(ts))

    key, skey = jax.random.split(key)
    x_0_samples = target_distribution.sample(key, num_samples)
    key, skey = jax.random.split(key)
    x_T_samples = sde.reference_dist.sample(skey, num_samples)

    key, skey = jax.random.split(key)
    forward_sde_samples = sde_solve(
        sde=sde,
        grad_log_pi=score,
        x=x_0_samples,
        key=skey,
        prob_flow=False,
        num_steps=1000,
        solver=dfx.Euler(),
        rtol=None,
        atol=None,
        forward=True,
        ts=ts,
    )

    key, skey = jax.random.split(key)
    reverse_sde_samples = sde_solve(
        sde=sde,
        grad_log_pi=score,
        x=x_T_samples,
        key=skey,
        prob_flow=False,
        num_steps=1000,
        solver=dfx.Euler(),
        rtol=None,
        atol=None,
        forward=False,
        ts=ts_rev,
    )

    key, skey = jax.random.split(key)
    forward_analytic_samples = jax.vmap(
        lambda t: sde.forward_path_marginal_dist(
            broadcast(t, x_0_samples), x_0_samples
        ).sample(skey, num_samples)
    )(jnp.array(ts))

    key, skey = jax.random.split(key)
    uncorrected_log_pi = (
        lambda lbd, x: uncorrected_approx_potential.approx_log_gt(
            lbd=lbd, x=x, density_state=0
        )[0]
        + sde.reference_dist.evaluate_log_density(x, 0)[0]
    )
    reverse_sde_naive_samples = sde_solve(
        sde=sde,
        grad_log_pi=x_gradient(uncorrected_log_pi),
        x=x_T_samples,
        key=skey,
        prob_flow=False,
        num_steps=1000,
        solver=dfx.Euler(),
        rtol=None,
        atol=None,
        forward=False,
        ts=ts_rev,
    )

    fig, axes = plt.subplots(
        2, len(ts) // 2, sharey=False, sharex=False, figsize=(3 * len(ts), 6)
    )
    line = jnp.linspace(-5, 5)[:, None]
    for i, (ax, t, fwd_sde, rev_lrn, fwd_ana, rev_naive) in enumerate(
        zip(
            axes.flatten(),
            ts,
            forward_sde_samples,
            reversed(reverse_sde_samples),
            forward_analytic_samples,
            reversed(reverse_sde_naive_samples),
        )
    ):
        centre = jnp.mean(fwd_sde[:, 0])
        std = jnp.std(fwd_sde[:, 0])
        # ax.plot(line[:, 0], jnp.exp(sde.forward_path_marginal_dist(t).evaluate_log_density(line)))
        sns.kdeplot(fwd_sde[:, 0], ax=ax, label="fwd sde", alpha=0.5)
        sns.kdeplot(rev_naive[:, 0], ax=ax, label="rev nve", alpha=0.5)
        sns.kdeplot(rev_lrn[:, 0], ax=ax, label="rev lrn", alpha=0.5)
        if i != 0:
            sns.kdeplot(fwd_ana[:, 0], ax=ax, label="fwd ana", alpha=0.5)
        # if i == 0:
        #     sns.kdeplot(x_0_samples[:, 0], ax=ax, label="x0 sampl", alpha=0.5)
        # if i == (len(ts) - 1):
        #     sns.kdeplot(x_T_samples[:, 0], ax=ax, label="xT sampl", alpha=0.5)
        ax.set_xlim((centre - 5 * std, centre + 5 * std))
        ax.set_title(f"{t=:0.3f}")
        ax.legend(fontsize=6)

    return fig


@check_shapes("initial_samples: [b, d]")
def single_reverse_sde_rollout(
    key: Key,
    sde: SDE,
    target_distribution,
    score,
    initial_samples: Array,
    t_f,
    t_0,
    n_steps: int,
):
    rng, rng_ = jax.random.split(key)
    fig = plt.figure()
    ax = plt.gca()
    sns.kdeplot(initial_samples[:, 0], ax=ax, color="blue")
    x_prev = initial_samples
    num_particles = initial_samples.shape[0]
    delta = (t_f - t_0) / n_steps
    noise_dist = NormalDistributionWrapper(0.0, 1.0, 1)
    ts = jnp.linspace(0, t_f, n_steps + 1)
    t1 = jnp.flip(ts[:-1])
    t2 = jnp.flip(ts[1:])
    for i, (t_new, t_prev) in enumerate(zip(t1, t2)):
        t_prev = broadcast(t_prev, x_prev)
        rng, rng_ = jax.random.split(rng)
        eps = noise_dist.sample(rng_, num_particles)
        bt = sde.scheduler.beta_t(t_prev)
        lbd = sde.scheduler.lambda_t0(t_prev)
        x_new = (
            x_prev
            + delta
            * bt[..., None]
            * (x_prev + 2 * sde.sigma**2 * score(lbd=lbd, x=x_prev))
            + jnp.sqrt(2 * bt[..., None] * sde.sigma**2 * delta) * eps
        )
        if i % 10 == 0:
            sns.kdeplot(
                x_new[:, 0],
                ax=ax,
                color="black",
                alpha=0.3 + (sde.scheduler.t_f - float(t_new)) * 0.7,
            )
        x_prev = x_new
    sns.kdeplot(x_new[:, 0], ax=ax, color="black")
    if target_distribution:
        target = target_distribution.sample(rng, num_particles)
        sns.kdeplot(target[:, 0], ax=ax, color="red")

    return fig


def learned_v_analytic(
    key, sde, target_distribution, learned, analytic, plot_dim, title
):
    ts = list(jnp.linspace(sde.scheduler.t_0, sde.scheduler.t_f, 10))

    key, skey1, skey2 = jax.random.split(key, 3)
    x_0_samples = target_distribution.sample(skey1, 10000)
    forward_analytic_samples = jax.vmap(
        lambda t: sde.forward_path_marginal_dist(
            broadcast(t, x_0_samples), x_0_samples
        ).sample(skey2, 10000)
    )(jnp.array(ts)[1:])
    forward_analytic_samples = jnp.insert(
        forward_analytic_samples, 0, x_0_samples, axis=0
    )

    fig, axs = plt.subplots(nrows=2, ncols=5, figsize=(3 * len(ts), 6))
    plt.title(title)
    for t, ax, fwd_samples in zip(ts, axs.flatten(), forward_analytic_samples):
        centre = jnp.mean(fwd_samples[:, 0])
        std = jnp.std(fwd_samples[:, 0])
        lower = centre - 2 * std
        upper = centre + 2 * std
        xs = jnp.tile(jnp.linspace(lower, upper, 100)[:, None], (1, sde.dim))
        ax.set_xlim((lower, upper))
        t = broadcast(t, xs)
        lbd = sde.scheduler.lambda_t0(t)
        if len(learned(lbd=lbd, x=xs, density_state=0)[0].shape) == 2:
            ax.set_title(f"t={float(t[0]):.3f}")
            ax.plot(
                xs[:, plot_dim],
                learned(lbd=lbd, x=xs, density_state=0)[0][:, plot_dim],
                label=f"NN",
                alpha=0.5,
            )
            ax.plot(
                xs[:, plot_dim],
                analytic(lbd=lbd, x=xs, density_state=0)[0][:, plot_dim],
                label=f"True",
                alpha=0.5,
            )
        else:
            ax.set_title(f"t={float(t[0]):.3f}")
            ax.plot(
                xs[:, plot_dim],
                learned(lbd=lbd, x=xs, density_state=0)[0],
                label=f"NN",
                alpha=0.5,
            )
            ax.plot(
                xs[:, plot_dim],
                analytic(lbd=lbd, x=xs, density_state=0)[0],
                label=f"True",
                alpha=0.5,
            )
    plt.legend()
    return fig


def fun_t_line_plot(sde, fun1, plot_dim, title, fun2=None, xrange=None):
    if xrange is not None:
        xs = jnp.tile(jnp.linspace(xrange[0], xrange[1], 100)[:, None], (1, sde.dim))
    else:
        xs = jnp.tile(jnp.linspace(-20, 21, 100)[:, None], (1, sde.dim))
    if fun2 is not None:
        fig, axs = plt.subplots(nrows=1, ncols=2, sharey=True)
        plt.title(title)
        ts = list(jnp.linspace(sde.scheduler.t_0, sde.scheduler.t_f, 5))
        for t in ts:
            t = broadcast(t, xs)
            lbd = sde.scheduler.lambda_t0(t)
            if len(fun1(lbd=lbd, x=xs).shape) == 2:
                axs[0].plot(
                    xs[:, plot_dim],
                    fun1(lbd=lbd, x=xs)[:, plot_dim],
                    label=f"t={float(t[0])}",
                )
                axs[1].plot(
                    xs[:, plot_dim],
                    fun2(lbd=lbd, x=xs)[:, plot_dim],
                    label=f"t={float(t[0])}",
                )
            else:
                axs[0].plot(
                    xs[:, plot_dim],
                    fun1(lbd=lbd, x=xs),
                    label=f"t={float(t[0])}",
                )
                axs[1].plot(
                    xs[:, plot_dim],
                    fun2(lbd=lbd, x=xs),
                    label=f"t={float(t[0])}",
                )
        plt.legend()
    else:
        fig = plt.figure()
        plt.title(title)
        ts = list(jnp.linspace(sde.scheduler.t_0, sde.scheduler.t_f, 5))
        for t in ts:
            t = broadcast(t, xs)
            lbd = sde.scheduler.lambda_t0(t)
            if len(fun1(lbd=lbd, x=xs).shape) == 2:
                plt.plot(
                    xs[:, plot_dim],
                    fun1(lbd=lbd, x=xs)[:, plot_dim],
                    label=f"t={float(t[0])}",
                )
            else:
                plt.plot(
                    xs[:, plot_dim],
                    fun1(lbd=lbd, x=xs),
                    label=f"t={float(t[0])}",
                )
        plt.legend(fontsize=6)
    return fig
