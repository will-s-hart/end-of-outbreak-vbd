"""Reproduction-number model priors used by the renewal inference.

This module holds the R_t-model-specific building blocks — the default prior medians /
percentiles / autoregressive hyperparameters, the AR innovation-std helper, the smooth
clip used by the suitability model, and the two ``rep_no_vec_func`` builders that register
the PyMC random variables for each model. Both the public fitting functions in
``inference`` and the quasi-real-time scripts call these builders, so the inference module
stays focused on the general renewal inference.
"""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from endoutbreakvbd._types import FloatArray, IntArray
from endoutbreakvbd.utils import lognormal_params_from_median_percentile_2_5


@dataclass(frozen=True)
class Defaults:
    """Default prior medians, percentiles, and AR hyperparameters for model fitting."""

    rep_no_prior_median: float = 1.0
    rep_no_prior_percentile_2_5: float = 0.2
    log_rep_no_rho: float | list[float] = 0.975
    suitability_std: float = 0.05
    suitability_rho: float = 0.975
    rep_no_factor_prior_median: float = 1.0
    rep_no_factor_prior_percentile_2_5: float = 0.2
    log_rep_no_factor_rho: float = 0.975


DEFAULTS = Defaults()


def build_known_rep_no(
    *, rep_no_func: Callable[[IntArray], FloatArray]
) -> Callable[[int], Any]:
    """
    Build a ``rep_no_vec_func`` registering a fixed (known) reproduction number.

    The reproduction number is evaluated from ``rep_no_func`` over the whole inference
    horizon and registered as a deterministic (constant across draws), so the fit infers
    only the remaining random variables. This isolates, for example, latent-case
    (reporting) inference from reproduction-number estimation.

    Parameters
    ----------
    rep_no_func : Callable[[IntArray], FloatArray]
        Function returning the known reproduction number at each of a vector of times
        (days).

    Returns
    -------
    Callable[[int], Any]
        Function taking the inference horizon and returning the ``rep_no`` deterministic.
    """

    def rep_no_vec_func(t_stop: int) -> Any:
        return pm.Deterministic(
            "rep_no",
            pt.as_tensor_variable(
                np.asarray(rep_no_func(np.arange(t_stop)), dtype=float)
            ),
            dims=("time",),
        )

    return rep_no_vec_func


def build_ar_rep_no(
    *,
    prior_median: float | None = None,
    prior_percentile_2_5: float | None = None,
    rho: float | list[float] | None = None,
) -> Callable[[int], Any]:
    """
    Build a ``rep_no_vec_func`` registering an autoregressive log-reproduction-number
    process with a lognormal stationary prior.

    Parameters
    ----------
    prior_median : float, optional
        Median of the lognormal prior on the reproduction number. Defaults to
        ``DEFAULTS.rep_no_prior_median``.
    prior_percentile_2_5 : float, optional
        2.5th percentile of the lognormal prior on the reproduction number. Defaults to
        ``DEFAULTS.rep_no_prior_percentile_2_5``.
    rho : float or list[float], optional
        Autoregressive coefficient(s); a length-2 list specifies an AR(2) process.
        Defaults to ``DEFAULTS.log_rep_no_rho``.

    Returns
    -------
    Callable[[int], Any]
        Function taking the inference horizon and returning the ``rep_no`` deterministic.
    """
    prior_median = (
        DEFAULTS.rep_no_prior_median if prior_median is None else prior_median
    )
    prior_percentile_2_5 = (
        DEFAULTS.rep_no_prior_percentile_2_5
        if prior_percentile_2_5 is None
        else prior_percentile_2_5
    )
    rho = DEFAULTS.log_rep_no_rho if rho is None else rho
    rep_no_lognormal_params = lognormal_params_from_median_percentile_2_5(
        median=prior_median, percentile_2_5=prior_percentile_2_5
    )
    log_rep_no_innovation_std = _ar_innovation_std(
        stationary_std=rep_no_lognormal_params["sigma"], rho=rho
    )

    def rep_no_vec_func(_: int) -> Any:
        log_rep_no_deviation_vec = pm.AR(
            "log_rep_no_deviation",
            sigma=log_rep_no_innovation_std,
            rho=rho,
            dims=("time",),
            init_dist=pm.Normal.dist(mu=0, sigma=rep_no_lognormal_params["sigma"]),
        )
        rep_no_vec = pm.Deterministic(
            "rep_no",
            pm.math.exp(
                rep_no_lognormal_params["mu"] + cast(Any, log_rep_no_deviation_vec)
            ),
            dims=("time",),
        )
        return rep_no_vec

    return rep_no_vec_func


def build_suitability_rep_no(
    *,
    suitability_mean_vec: list[float] | FloatArray,
    suitability_std: float | None = None,
    suitability_rho: float | None = None,
    rep_no_factor_prior_median: float | None = None,
    rep_no_factor_prior_percentile_2_5: float | None = None,
    log_rep_no_factor_rho: float | None = None,
) -> Callable[[int], Any]:
    """
    Build a ``rep_no_vec_func`` decomposing the reproduction number into a climate
    suitability profile and a reproduction-number factor.

    Parameters
    ----------
    suitability_mean_vec : list[float] or FloatArray
        Prior mean suitability at each time.
    suitability_std : float, optional
        Stationary standard deviation of the suitability deviations. Defaults to
        ``DEFAULTS.suitability_std``.
    suitability_rho : float, optional
        Autoregressive coefficient for the suitability deviations. Defaults to
        ``DEFAULTS.suitability_rho``.
    rep_no_factor_prior_median : float, optional
        Median of the lognormal prior on the reproduction-number factor. Defaults to
        ``DEFAULTS.rep_no_factor_prior_median``.
    rep_no_factor_prior_percentile_2_5 : float, optional
        2.5th percentile of the lognormal prior on the reproduction-number factor.
        Defaults to ``DEFAULTS.rep_no_factor_prior_percentile_2_5``.
    log_rep_no_factor_rho : float, optional
        Autoregressive coefficient for the log reproduction-number factor. Defaults to
        ``DEFAULTS.log_rep_no_factor_rho``.

    Returns
    -------
    Callable[[int], Any]
        Function taking the inference horizon and returning the ``rep_no`` deterministic,
        also registering the ``suitability`` and ``rep_no_factor`` deterministics.
    """
    suitability_mean_vec = np.asarray(suitability_mean_vec)
    suitability_std = (
        DEFAULTS.suitability_std if suitability_std is None else suitability_std
    )
    suitability_rho = (
        DEFAULTS.suitability_rho if suitability_rho is None else suitability_rho
    )
    rep_no_factor_prior_median = (
        DEFAULTS.rep_no_factor_prior_median
        if rep_no_factor_prior_median is None
        else rep_no_factor_prior_median
    )
    rep_no_factor_prior_percentile_2_5 = (
        DEFAULTS.rep_no_factor_prior_percentile_2_5
        if rep_no_factor_prior_percentile_2_5 is None
        else rep_no_factor_prior_percentile_2_5
    )
    log_rep_no_factor_rho = (
        DEFAULTS.log_rep_no_factor_rho
        if log_rep_no_factor_rho is None
        else log_rep_no_factor_rho
    )
    rep_no_factor_lognormal_params = lognormal_params_from_median_percentile_2_5(
        median=rep_no_factor_prior_median,
        percentile_2_5=rep_no_factor_prior_percentile_2_5,
    )

    def rep_no_vec_func(t_stop: int) -> Any:
        if len(suitability_mean_vec) < t_stop:
            raise ValueError(
                f"suitability_mean_vec has length {len(suitability_mean_vec)} but R_t is "
                f"inferred to horizon {t_stop}; extend it (typically a serial interval "
                "past the incidence) so R_t can be projected forward."
            )
        log_rep_no_factor_deviation_vec = pm.AR(
            "log_rep_no_factor_deviation",
            sigma=_ar_innovation_std(
                stationary_std=rep_no_factor_lognormal_params["sigma"],
                rho=log_rep_no_factor_rho,
            ),
            rho=log_rep_no_factor_rho,
            dims=("time",),
            init_dist=pm.Normal.dist(
                mu=0, sigma=rep_no_factor_lognormal_params["sigma"]
            ),
        )
        rep_no_factor_vec = pm.Deterministic(
            "rep_no_factor",
            pm.math.exp(
                rep_no_factor_lognormal_params["mu"]
                + cast(Any, log_rep_no_factor_deviation_vec)
            ),
            dims=("time",),
        )
        suitability_deviation_vec = pm.AR(
            "suitability_deviation",
            sigma=suitability_std * np.sqrt(1 - suitability_rho**2),
            rho=suitability_rho,
            dims=("time",),
            init_dist=pm.Normal.dist(mu=0, sigma=suitability_std),
        )
        suitability_vec = pm.Deterministic(
            "suitability",
            _softclip(
                suitability_mean_vec[:t_stop] + cast(Any, suitability_deviation_vec),
                lower=1e-8,
                upper=1.0,
            ),
            dims=("time",),
        )
        rep_no_vec = pm.Deterministic(
            "rep_no", rep_no_factor_vec * suitability_vec, dims=("time",)
        )
        return rep_no_vec

    return rep_no_vec_func


def _ar_innovation_std(*, stationary_std: float, rho: float | list[float]) -> float:
    # Innovation std giving the target stationary std for an AR(1) or AR(2) process
    if not isinstance(rho, list):
        rho = [rho]
    if len(rho) == 1:
        return stationary_std * np.sqrt(1 - rho[0] ** 2)
    elif len(rho) == 2:
        return stationary_std * np.sqrt(
            1 - (rho[0] ** 2 * (1 + rho[1])) / (1 - rho[1]) - rho[1] ** 2
        )
    raise ValueError("Only AR(1) and AR(2) are supported")


def _softclip(x: Any, *, lower: float, upper: float, tau: float = 0.001) -> Any:
    # Smooth (softplus-based) clip of x to the interval [lower, upper]
    return (
        x
        + tau * pm.math.log1pexp((lower - x) / tau)
        - tau * pm.math.log1pexp((x - upper) / tau)
    )
