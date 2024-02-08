import typing as tp
import jax.numpy as jnp


class LinearStepSizeScheduler:
    def __init__(self, step_times: tp.List, step_sizes: tp.List, t_f: float):
        self.step_times = step_times
        self.step_sizes = step_sizes
        self.t_f = t_f

    def __call__(self, t: float):
        return jnp.interp(t, jnp.array(self.step_times), jnp.array(self.step_sizes))
