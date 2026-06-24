import warnings
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import arviz_base as azb
import numpy as np
import pymc as pm
import xarray as xr
from arviz_stats import ess, rhat
from tqdm import tqdm

from endoutbreakvbd._types import (
    FloatArray,
    IncidenceSeriesInput,
    IntArray,
    RepNoOutput,
    SerialIntervalInput,
)
from endoutbreakvbd.additional_case_prob import calc_additional_case_prob_analytical
from endoutbreakvbd.utils import (
    lognormal_params_from_median_percentile_2_5,
    rep_no_from_grid,
)


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


def _compute_check_diagnostics(
    posterior: xr.Dataset,
    sample_stats: xr.Dataset | None,
    *,
    raise_on_problems: bool = False,
    var_name: str = "rep_no",
    ess_min_threshold: float = 1000,
    rhat_max_threshold: float = 1.01,
) -> dict[str, float]:
    # Summarise convergence diagnostics for `var_name` and warn (or raise) on poor
    # sampling. Divergences are N/A (NaN) when sample_stats is None, as for the
    # quasi-real-time aggregate of many separate fits.
    rhat_vals = rhat(posterior, var_names=var_name)[var_name]
    ess_vals = ess(posterior, var_names=var_name)[var_name]
    n_diverging = (
        np.nan if sample_stats is None else float(sample_stats["diverging"].sum())
    )
    diagnostics = {
        "rhat_mean": rhat_vals.mean().item(),
        "rhat_median": rhat_vals.median().item(),
        "rhat_max": rhat_vals.max().item(),
        "ess_mean": ess_vals.mean().item(),
        "ess_median": ess_vals.median().item(),
        "ess_min": ess_vals.min().item(),
        "n_diverging": n_diverging,
    }
    problems = []
    if diagnostics["ess_min"] < ess_min_threshold:
        problems.append(f"min ESS {diagnostics['ess_min']:.1f} < {ess_min_threshold}")
    if diagnostics["rhat_max"] > rhat_max_threshold:
        problems.append(
            f"max R-hat {diagnostics['rhat_max']:.4f} > {rhat_max_threshold}"
        )
    if not np.isnan(n_diverging) and n_diverging > 0:
        problems.append(f"{int(n_diverging)} divergence(s)")
    if problems:
        message = "Poor sampling diagnostics: " + "; ".join(problems)
        if raise_on_problems:
            raise RuntimeError(message)
        warnings.warn(message, stacklevel=2)
    return diagnostics


def _fit_model(
    *,
    incidence_vec: IncidenceSeriesInput,
    serial_interval_dist_vec: SerialIntervalInput,
    rep_no_vec_func: Callable[[int], Any],
    quasi_real_time: bool,
    t_infer_to: int | None = None,
    step_func: Callable[[], Any] | None = None,
    thin: int = 1,
    rng: np.random.Generator | int | None = None,
    freeze_from_final_case: bool = False,
    compute_diagnostics: bool = False,
    raise_on_poor_diagnostics: bool = False,
    **kwargs_sample: Any,
) -> xr.Dataset:
    # Core model fit: build the PyMC model, sample, and derive R_t summaries and
    # additional-case probabilities.
    kwargs_sample = {
        "nuts_sampler": "nutpie",
        "nuts_sampler_kwargs": {"adaptation": "low_rank"},
        "draws": 1000,
        "tune": 1000,
        "chains": 4,
        **kwargs_sample,
    }
    if rng is not None:
        kwargs_sample = {**kwargs_sample, "random_seed": rng}
    t_data_to = len(incidence_vec)
    t_infer_to = t_infer_to or t_data_to
    if t_infer_to < t_data_to:
        incidence_vec = incidence_vec[:t_infer_to]
        t_data_to = t_infer_to

    if quasi_real_time:
        if freeze_from_final_case:
            raise NotImplementedError(
                "freeze_from_final_case is not supported with quasi_real_time=True"
            )
        if len(incidence_vec) < 2:
            raise ValueError(
                "quasi_real_time inference requires at least 2 time points"
            )
        posterior_list = []
        for t in tqdm(
            range(1, len(incidence_vec)),
            desc="Inferring R_t using data up to each time",
        ):
            ds_posterior_curr = _fit_model(
                incidence_vec=incidence_vec[:t],
                serial_interval_dist_vec=serial_interval_dist_vec,
                rep_no_vec_func=rep_no_vec_func,
                quasi_real_time=False,
                t_infer_to=np.minimum(t + len(serial_interval_dist_vec), t_infer_to),
                step_func=step_func,
                thin=thin,
                rng=rng,
                **{"quiet": True, "progressbar": False, **kwargs_sample},
            )
            posterior_list.append(
                ds_posterior_curr[
                    [
                        var
                        for var in ds_posterior_curr.data_vars
                        if "time" in ds_posterior_curr[var].dims
                    ]
                ].isel(time=([0, 1] if t == 1 else [t]))
            )
        posterior = xr.concat(posterior_list, dim="time")
        if compute_diagnostics:
            posterior.attrs["diagnostics"] = _compute_check_diagnostics(
                posterior, None, raise_on_problems=raise_on_poor_diagnostics
            )
        return posterior

    serial_interval_dist_vec_ext = np.concatenate(
        [
            serial_interval_dist_vec,
            np.zeros(np.maximum(t_data_to - 1 - len(serial_interval_dist_vec), 0)),
        ]
    )

    incidence_vec_local = np.zeros(t_data_to)
    incidence_vec_local[1:] = incidence_vec[1:]

    foi_vec = np.zeros(t_data_to)
    for t in range(1, t_data_to):
        foi_vec[t] = np.sum(incidence_vec[:t][::-1] * serial_interval_dist_vec_ext[:t])

    nonzero_foi_idx = foi_vec > 0
    if np.any(incidence_vec_local[~nonzero_foi_idx]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of infection is 0."
        )

    t_vec = np.arange(t_infer_to)

    with pm.Model(coords={"time": t_vec}):
        rep_no_vec = rep_no_vec_func(t_infer_to)

        expected_incidence_local = rep_no_vec[:t_data_to] * foi_vec

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[nonzero_foi_idx],
            observed=incidence_vec_local[nonzero_foi_idx],
        )
        if step_func is not None:
            kwargs_sample["step"] = step_func()

        trace = pm.sample(**kwargs_sample)
    sample_stats = trace.sample_stats if compute_diagnostics else None
    if thin > 1:
        trace = trace.isel(draw=slice(0, None, thin))
        trace = trace.assign_coords(draw=np.arange(len(trace.posterior.draw)))
    ds_posterior = azb.convert_to_dataset(trace.posterior)
    if freeze_from_final_case:
        # Hold R_t after the final case fixed at its final-case-day posterior samples
        t_final_case = int(np.nonzero(np.asarray(incidence_vec))[0][-1])
        rep_no_frozen = ds_posterior["rep_no"].isel(time=t_final_case)
        ds_posterior = ds_posterior.assign(
            rep_no=ds_posterior["rep_no"].where(
                ds_posterior["time"] <= t_final_case, rep_no_frozen
            )
        )
    # Extract summary stats
    ds_posterior = ds_posterior.assign(
        rep_no_mean=ds_posterior["rep_no"].mean(dim=["chain", "draw"]),
        rep_no_lower=ds_posterior["rep_no"]
        .quantile(0.025, dim=["chain", "draw"])
        .drop_vars("quantile"),
        rep_no_upper=ds_posterior["rep_no"]
        .quantile(0.975, dim=["chain", "draw"])
        .drop_vars("quantile"),
    )
    # Compute daily probability of additional cases
    rep_no_mat = ds_posterior["rep_no"].transpose("time", ...).values

    def rep_no_post_func(t: int | IntArray) -> RepNoOutput:
        return rep_no_from_grid(t, rep_no_grid=rep_no_mat, periodic=False)

    prob_vec = calc_additional_case_prob_analytical(
        incidence_vec=incidence_vec,
        rep_no_func=rep_no_post_func,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=t_vec,
    )
    ds_posterior = ds_posterior.assign(additional_case_prob=(("time",), prob_vec))
    if compute_diagnostics:
        ds_posterior.attrs["diagnostics"] = _compute_check_diagnostics(
            ds_posterior, sample_stats, raise_on_problems=raise_on_poor_diagnostics
        )
    return ds_posterior


def fit_autoregressive_model(
    *,
    incidence_vec: IncidenceSeriesInput,
    serial_interval_dist_vec: SerialIntervalInput,
    prior_median: float | None = None,
    prior_percentile_2_5: float | None = None,
    rho: float | list[float] | None = None,
    quasi_real_time: bool = False,
    freeze_from_final_case: bool = False,
    compute_diagnostics: bool = True,
    raise_on_poor_diagnostics: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Fit the autoregressive renewal model to an incidence time series, inferring a
    time-varying reproduction number.

    Parameters
    ----------
    incidence_vec : IncidenceSeriesInput
        Observed incidence time series.
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    prior_median : float, optional
        Median of the lognormal prior on the reproduction number. Defaults to
        ``DEFAULTS.rep_no_prior_median``.
    prior_percentile_2_5 : float, optional
        2.5th percentile of the lognormal prior on the reproduction number. Defaults to
        ``DEFAULTS.rep_no_prior_percentile_2_5``.
    rho : float or list[float], optional
        Autoregressive coefficient(s); a length-2 list specifies an AR(2) process.
        Defaults to ``DEFAULTS.log_rep_no_rho``.
    quasi_real_time : bool
        If True, refit using only the data available up to each successive time point.
    freeze_from_final_case : bool
        If True, hold the reproduction number fixed at its final-case-day posterior
        samples after the final observed case.
    compute_diagnostics : bool
        If True, compute convergence diagnostics, attach them to the returned
        dataset's ``attrs["diagnostics"]``, and warn on poor sampling.
    raise_on_poor_diagnostics : bool
        If True, raise (instead of warning) when sampling diagnostics are poor.
    **kwargs : Any
        Additional keyword arguments forwarded to the sampler.

    Returns
    -------
    xr.Dataset
        Posterior dataset including the inferred reproduction numbers, summary
        statistics, and the daily probability of additional cases.
    """
    prior_median = prior_median or DEFAULTS.rep_no_prior_median
    prior_percentile_2_5 = prior_percentile_2_5 or DEFAULTS.rep_no_prior_percentile_2_5
    rho = rho or DEFAULTS.log_rep_no_rho
    lognormal_params = lognormal_params_from_median_percentile_2_5(
        median=prior_median, percentile_2_5=prior_percentile_2_5
    )
    log_rep_no_sigma_innov = _ar_innovation_std(
        stationary_std=lognormal_params["sigma"], rho=rho
    )

    def rep_no_vec_func(_: int) -> Any:
        log_rep_no_deviation_vec = pm.AR(
            "log_rep_no_deviation",
            sigma=log_rep_no_sigma_innov,
            rho=rho,
            dims=("time",),
            init_dist=pm.Normal.dist(mu=0, sigma=lognormal_params["sigma"]),
        )
        rep_no_vec = pm.Deterministic(
            "rep_no",
            pm.math.exp(lognormal_params["mu"] + cast(Any, log_rep_no_deviation_vec)),
            dims=("time",),
        )
        return rep_no_vec

    return _fit_model(
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        quasi_real_time=quasi_real_time,
        freeze_from_final_case=freeze_from_final_case,
        compute_diagnostics=compute_diagnostics,
        raise_on_poor_diagnostics=raise_on_poor_diagnostics,
        **kwargs,
    )


def fit_suitability_model(
    *,
    incidence_vec: IncidenceSeriesInput,
    serial_interval_dist_vec: SerialIntervalInput,
    suitability_mean_vec: list[float] | FloatArray,
    suitability_std: float | None = None,
    suitability_rho: float | None = None,
    rep_no_factor_prior_median: float | None = None,
    rep_no_factor_prior_percentile_2_5: float | None = None,
    log_rep_no_factor_rho: float | None = None,
    quasi_real_time: bool = False,
    compute_diagnostics: bool = True,
    raise_on_poor_diagnostics: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Fit the climate-suitability renewal model, decomposing the reproduction number into a
    suitability profile and a reproduction-number factor.

    Parameters
    ----------
    incidence_vec : IncidenceSeriesInput
        Observed incidence time series.
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
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
    quasi_real_time : bool
        If True, refit using only the data available up to each successive time point.
    compute_diagnostics : bool
        If True, compute convergence diagnostics, attach them to the returned
        dataset's ``attrs["diagnostics"]``, and warn on poor sampling.
    raise_on_poor_diagnostics : bool
        If True, raise (instead of warning) when sampling diagnostics are poor.
    **kwargs : Any
        Additional keyword arguments forwarded to the sampler.

    Returns
    -------
    xr.Dataset
        Posterior dataset including the inferred reproduction numbers, suitability,
        reproduction-number factor, summary statistics, and the daily probability of
        additional cases.
    """
    suitability_mean_vec = np.asarray(suitability_mean_vec)
    suitability_std = suitability_std or DEFAULTS.suitability_std
    suitability_rho = suitability_rho or DEFAULTS.suitability_rho
    rep_no_factor_prior_median = (
        rep_no_factor_prior_median or DEFAULTS.rep_no_factor_prior_median
    )
    rep_no_factor_prior_percentile_2_5 = (
        rep_no_factor_prior_percentile_2_5
        or DEFAULTS.rep_no_factor_prior_percentile_2_5
    )
    log_rep_no_factor_rho = log_rep_no_factor_rho or DEFAULTS.log_rep_no_factor_rho
    rep_no_factor_lognormal_params = lognormal_params_from_median_percentile_2_5(
        median=rep_no_factor_prior_median,
        percentile_2_5=rep_no_factor_prior_percentile_2_5,
    )

    def rep_no_vec_func(t_infer_to: int) -> Any:
        log_rep_no_factor_deviation_vec = pm.AR(
            "log_rep_no_factor_deviation",
            sigma=rep_no_factor_lognormal_params["sigma"]
            * np.sqrt(1 - log_rep_no_factor_rho**2),
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
                suitability_mean_vec[:t_infer_to]
                + cast(Any, suitability_deviation_vec),
                lower=1e-8,
                upper=1.0,
            ),
            dims=("time",),
        )
        rep_no_vec = pm.Deterministic(
            "rep_no", rep_no_factor_vec * suitability_vec, dims=("time",)
        )
        return rep_no_vec

    ds_posterior = _fit_model(
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        quasi_real_time=quasi_real_time,
        compute_diagnostics=compute_diagnostics,
        raise_on_poor_diagnostics=raise_on_poor_diagnostics,
        **kwargs,
    )
    # Extract further summary stats
    ds_posterior = ds_posterior.assign(
        suitability_mean=ds_posterior["suitability"].mean(dim=["chain", "draw"]),
        suitability_lower=ds_posterior["suitability"]
        .quantile(0.025, dim=["chain", "draw"])
        .drop_vars("quantile"),
        suitability_upper=ds_posterior["suitability"]
        .quantile(0.975, dim=["chain", "draw"])
        .drop_vars("quantile"),
        rep_no_factor_mean=ds_posterior["rep_no_factor"].mean(dim=["chain", "draw"]),
        rep_no_factor_lower=ds_posterior["rep_no_factor"]
        .quantile(0.025, dim=["chain", "draw"])
        .drop_vars("quantile"),
        rep_no_factor_upper=ds_posterior["rep_no_factor"]
        .quantile(0.975, dim=["chain", "draw"])
        .drop_vars("quantile"),
    )
    return ds_posterior


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
