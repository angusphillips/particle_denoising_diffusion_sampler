from typing import Callable, Union
from jaxtyping import Array

Scalar = Union[float, int]
Numeric = Union[Array, Scalar]
Schedule = Callable[[Numeric], Numeric]


class loop_schedule:
    def __init__(self, schedule: Schedule, freq: int):
        self.schedule = schedule
        self.freq = freq

    def __call__(self, count: Numeric):
        return self.schedule(count % self.freq)
