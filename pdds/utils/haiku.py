from typing import Any, Callable, NamedTuple


import haiku as hk


class TranfsormedWithGrad(NamedTuple):
    """Holds a triple of pure functions.

    Attributes:
    init: A pure function: ``params = init(rng, *a, **k)``
    apply: A pure function: ``out = apply(params, rng, *a, **k)``
    apply_grad: A pure function : ``out = apply(params, rng, *a, **k)
    """

    # Args: [Optional[PRNGKey], ...]
    init: Callable[..., hk.Params]

    # Args: [Params, Optional[PRNGKey], ...]
    apply: Callable[..., Any]

    # Args [Parmas, Optional[PRNGKey], ...]
    apply_grad: Callable[..., Any]


class SDETermsModel(NamedTuple):
    init: Callable[..., hk.Params]
    log_pt: Callable[..., Any]
    grad_log_pt: Callable[..., Any]
    log_gt: Callable[..., Any]
    grad_log_gt: Callable[..., Any]
    log_gt_tilde: Callable[..., Any]
    grad_log_gt_tilde: Callable[..., Any]
