from collections import Counter
from functools import wraps, partial
import matplotlib.pyplot as plt
import typing as tp
from hashlib import sha256
import numpy as np
import seaborn as sns

# noinspection PyProtectedMember
from scipy.stats._distn_infrastructure import rv_frozen  # not critical, only for typing
from scipy.stats import norm, random_correlation
import pickle
from os import path


def make_plot_function(f: tp.Callable) -> tp.Callable:
    """
    Decorator to add display and return controls on a plotting function.

    Parameters
    ----------
    f: function that takes as a keyword argument `ax` of type ``plt.Axes`` and draws on it. Besides that it might have other positional or keyword arguments. It is not expected to return anything

    Returns
    -------
    function `g` that has the same arguments as the original function `f`, plus a new keyword argument `show`. It works as follows:
    * if an `ax` is passed, it is drawn on by `f`. Otherwise if `ax` is None, a new ``plt.Axes`` object is created automatically
    * if `show` is true, the ``plt.Figure`` object associated with `ax` is displayed upon exit
    * `g` always return the ``plt.Figure`` object.
    """

    @wraps(f)
    def _f(*args, ax: plt.Axes = None, show: bool = True, **kwargs):
        if ax is None:
            fig, ax = plt.subplots()
        else:
            fig = ax.figure
        f(*args, **kwargs, ax=ax)
        if show:
            fig.show()
        return fig

    return _f


class temporary_numpy_seed:
    """
    Context handler to temporary set numpy seed to some value, then revert back to original setting once done. Is used nowhere in the programme except in the ``auto_colour`` function.
    """

    def __init__(self, seed):
        self.seed = seed

    def __enter__(self):
        self.old_state = np.random.get_state()
        np.random.seed(self.seed)
        return None

    def __exit__(self, exc_type, exc_val, exc_tb):
        # noinspection PyTypeChecker
        np.random.set_state(self.old_state)


def auto_colour(s: str, seed=None) -> tp.Tuple[float, float, float]:
    """
    Convert, in a reproducible fashion, a string into a RGB Tuple representing some colour. Helpful for automatic colouring in matplotlib.
    """
    b = s.encode()
    hasher = sha256()
    hasher.update(b)
    digest: tp.List[str] = list(str(int(hasher.hexdigest(), 16)))
    res = []
    with temporary_numpy_seed(seed):
        for i in range(3):
            three_digits = np.random.choice(digest, size=3, replace=True)
            three_digits = [int(d) * 10 ** (-j - 1) for j, d in enumerate(three_digits)]
            res.append(sum(three_digits))
    assert all([0 <= e <= 1 for e in res])
    # noinspection PyTypeChecker
    return tuple(res)


def reject_scalar(x):
    if isinstance(x, float):
        raise ValueError
    if isinstance(x, np.ndarray) and len(x.shape) == 0:
        raise ValueError


@make_plot_function
def compare_densities(
    data1: tp.Sequence[float], data2: tp.Sequence[float], ax: plt.Axes = None
):
    """
    Draw a plot to compare the densities of two datasets.

    :param data1, data2: datasets
    :return: nothing. A plot is displayed
    """
    sns.kdeplot(x=data1, ax=ax, label="1")
    sns.kdeplot(x=data2, ax=ax, label="2")
    ax.figure.legend()


def test_multivariate_gaussian(
    sample: tp.Sequence[tp.Sequence[float]],
    expected_mu,
    expected_Sigma,
    weight_dist: rv_frozen = norm(),
) -> None:
    """Display a diagnostic graph to verify the distribution of a sample which is expected to have been drawn from a multivariate Gaussian. At each run, the sample will be projected randomly on the real line using weights from `weight_dist`, then the estimated density of that projection is plotted alongside its expected density. As such, it is necessary to run the function multiple times to ensure the correctness of a sample.
    Parameters
    ----------
    expected_Sigma: array
        expected covariance matrix
    weight_dist: rv_frozen
        any Scipy one-dimensional distribution that admits a ``.rvs(size)`` call
    Returns
    -------
    None
        A plot is displayed.
    """
    sample = np.array(sample)
    N, d = sample.shape
    w = weight_dist.rvs(size=d)
    sample = (sample @ w).reshape((N,))
    expected_mu = float(np.dot(expected_mu, w))
    expected_Sigma = float(np.dot(expected_Sigma @ w, w))
    compare_densities(
        sample, norm.rvs(size=N, loc=expected_mu, scale=expected_Sigma**0.5)
    )


_Ty1 = tp.TypeVar("_Ty1")
_Ty2 = tp.TypeVar("_Ty2")
_Ty3 = tp.TypeVar("_Ty3")
_Ty4 = tp.TypeVar("_Ty4")


class ZipWithAssert:
    """Like zip, but raises AssertionError if iterables are not of the same length."""

    def __init__(self, *iterables: tp.Iterable):
        self.iterators = [iter(iterable) for iterable in iterables]

    def __iter__(self):
        return self

    def __next__(self):
        res = []
        for iterator in self.iterators:
            try:
                res.append(next(iterator))
            except StopIteration:
                pass
        if len(res) == 0:
            raise StopIteration
        elif len(res) == len(self.iterators):
            return tuple(res)
        else:
            raise AssertionError


@tp.overload
def zip_with_assert(
    i1: tp.Iterable[_Ty1], i2: tp.Iterable[_Ty2]
) -> tp.Iterable[tp.Tuple[_Ty1, _Ty2]]:
    ...


@tp.overload
def zip_with_assert(
    i1: tp.Iterable[_Ty1], i2: tp.Iterable[_Ty2], i3: tp.Iterable[_Ty3]
) -> tp.Iterable[tp.Tuple[_Ty1, _Ty2, _Ty3]]:
    ...


@tp.overload
def zip_with_assert(
    i1: tp.Iterable[_Ty1],
    i2: tp.Iterable[_Ty2],
    i3: tp.Iterable[_Ty3],
    i4: tp.Iterable[_Ty4],
) -> tp.Iterable[tp.Tuple[_Ty1, _Ty2, _Ty3, _Ty4]]:
    ...


def zip_with_assert(*args):
    """Like zip, but raises AssertionError if iterables are not of the same length."""
    return ZipWithAssert(*args)


# DEBUG = []


@make_plot_function
def test_asymptotics(
    x: np.ndarray,
    h: tp.Callable[[np.ndarray, np.ndarray], np.ndarray],
    mags: tp.Sequence[int],
    ax: plt.Axes = None,
):
    r"""Test whether a given function ``h(x,y)`` behaves as expected where `y` is a small perturbation of `x`. Useful for testing manual differentiation in the pre-JAX age, but can be used for other purposes as well.

    Given a fixed `x`, the function generates artificial `y` values for several inverse pertubation levels given by `mags`. It draws a line on the object `ax`, where the x axis is the inverse perturbation level and the y axis is the value of ``h(x,y)``.

    Following conventions, the function treats multiple `x` at the same time, so a vectorised `h` is expected and multiple lines will be drawn on `ax`.

    Parameters
    ----------
    x
        (N,d) numpy array representing N vectors in :math:`\mathbb R^d`
    h
        function taking two (N,d) arrays and returning an (N,) array
    mags
        different inverse pertubation levels for generating `y` from `x`. More specifically, if a certain `mag` is 5, then :math:`y \sim Uniform[x- 2^{-5}, x+2^{-5}]`
    """
    res = {}
    # DEBUG_y = {}
    for m in mags:
        y = x + np.random.uniform(-1, 1, x.shape) * 2 ** (-m)
        # DEBUG_y[m] = y
        res[m] = h(x, y)
    # DEBUG.append(x)
    # DEBUG.append(DEBUG_y)
    # DEBUG.append(res)
    for i in range(len(x)):
        ax.plot(list(mags), [arr[i] for arr in res.values()])


@make_plot_function
def test_gradient(
    x: np.ndarray,
    g: tp.Callable[[np.ndarray], np.ndarray],
    grad_g: tp.Callable[[np.ndarray], np.ndarray],
    mags: tp.Sequence[int],
    ax: plt.Axes,
):
    # tested
    r"""Test whether the gradient for a function `g` is inputted correctly. This is done using the ``test_asymptotic`` function.

    Parameters
    ----------
    x
        values of `x` to be tested for
    g
        function taking an (N,d) numpy array and return an (N,) numpy array
    grad_g
        function taking an (N,d) numpy array and return an (N,d) numpy array
    """

    def h(x_, y_):
        num = g(y_) - g(x_)
        denom_1 = grad_g(x_)
        denom_2 = y_ - x_
        denom = np.sum(denom_1 * denom_2, axis=1)
        return num / denom

    test_asymptotics(x=x, h=h, mags=mags, ax=ax)


def discrete_histogram(x: tp.Iterable[int]):
    """
    Create the discrete histogram of `x`, suitable for plotting.

    :param x: iterable of integers
    :return: two arrays u and v such that v[i] is the empirical probability mass function evaluated at u[i]
    """
    min_plot = min(x) - 1
    max_plot = max(x) + 2
    empirical_pmf = np.zeros(max_plot - min_plot)
    counts = Counter(x)
    for k, v in counts.items():
        empirical_pmf[k - min_plot] = v
    empirical_pmf /= empirical_pmf.sum()

    return range(min_plot, max_plot), empirical_pmf


@make_plot_function
def plot_histogram_discrete(
    x: tp.Iterable[int],
    exact_pmf: tp.Callable[[int], float] = None,
    ax: plt.Axes = None,
):
    # tested
    """
    Plot the histogram of a discrete sample and compare it with the exact probability mass function if available.

    :param x: the sample
    :param exact_pmf: exact probability mass function
    """
    min_plot = min(x) - 1
    max_plot = max(x) + 2
    empirical_pmf = discrete_histogram(x)[1]

    fig = ax.figure
    ax.scatter(range(min_plot, max_plot), empirical_pmf, label="empirical")
    if exact_pmf is not None:
        dx = range(min_plot, max_plot)
        dy = [exact_pmf(v) for v in dx]
        ax.scatter(dx, dy, label="theoretical")
    fig.legend()


def _proba_arr_to_pmf(j: int, proba_array: tp.Sequence[float]):
    if 0 <= j < len(proba_array):
        return proba_array[j]
    else:
        return 0


def proba_array_to_pmf(proba_array: tp.Sequence[float]) -> tp.Callable[[int], float]:
    """
    Convert a probability array into a probability mass function, intended to be fed into the ``plot_histogram_discrete`` function
    """
    # noinspection PyTypeChecker
    return partial(_proba_arr_to_pmf, proba_array=proba_array)


def random_mean_and_cov(
    d: int,
    range_mean: tp.Tuple[float, float] = (-1, 1),
    range_std: tp.Tuple[float, float] = (1, 2),
    seed=None,
) -> tp.Tuple[np.ndarray, np.ndarray]:
    # tested
    """
    Generate a random vector in `R^d` and a random `dxd` matrix to be used as the mean and the covariance matrix of a multivariate gaussian distribution. The correlation matrix is generated so that their eigenvalues are uniformly distributed in the simplex `{x_1 + ... + x_d = 1; x_i >=0}`.

    :param d: dimension
    :param range_mean: range of the uniform distribution on which the mean is drawed
    :param range_std: range of the uniform distribution on which the standard deviation is drawed
    :return: mean and cov
    """
    with temporary_numpy_seed(seed):  # should be replaced by the new key-based API
        if d == 1:
            corr = np.array([[1]])
        else:
            eigs = np.diff(sorted(np.r_[0, np.random.rand(d - 1), 1])) * d
            corr = random_correlation.rvs(eigs)
        mu = np.random.uniform(range_mean[0], range_mean[1], size=d)
        S = np.diag(np.random.uniform(range_std[0], range_std[1], size=d))
    return mu, S @ corr @ S.T


def batch_scalar_prod(x_, v_):
    r"""Calculate the scalar product of each vector of `x_` with `v_`"""
    N_, d_ = x_.shape
    assert v_.shape == (d_,)
    return (x_ @ v_).reshape((N_,))


def batch_quad_form(x, A):
    r"""Calculate the quadratic form :math:`v^T A v` for each vector v in x"""
    return np.sum((x @ A) * x, axis=1)


def pad_with_const(x):
    extra = np.ones((x.shape[0], 1))
    return np.hstack([extra, x])


def standardize_and_pad(x):
    mean = np.mean(x, axis=0)
    std = np.std(x, axis=0)
    std[std == 0] = 1.0
    x = (x - mean) / std
    return pad_with_const(x)


def load_data(path: str):
    with open(path, mode="rb") as f:
        x, y = pickle.load(f)
    y = (y + 1) // 2
    x = standardize_and_pad(x)
    return x, y
