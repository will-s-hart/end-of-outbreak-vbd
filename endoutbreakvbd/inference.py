import atexit
import os
import shutil
import warnings
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import arviz_base as azb
import numpy as np
import pymc as pm
import pytensor
import pytensor.tensor as pt
import xarray as xr
from arviz_stats import ess, rhat
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

from endoutbreakvbd._types import (
    FloatArray,
    IncidenceSeriesInput,
    IntArray,
    RepNoOutput,
    SerialIntervalInput,
)
from endoutbreakvbd.additional_case_prob import calc_additional_case_prob_analytical
from endoutbreakvbd.rep_no_models import (
    build_ar_rep_no,
    build_known_rep_no,
    build_suitability_rep_no,
)
from endoutbreakvbd.utils import renewal_convolution_matrix, rep_no_from_grid


def fit_autoregressive_model(
    *,
    incidence_vec: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    prior_median: float | None = None,
    prior_percentile_2_5: float | None = None,
    rho: float | list[float] | None = None,
    quasi_real_time: bool = False,
    t_calc: int | IntArray | None = None,
    reporting_prob: float | None = None,
    delay_cdf: FloatArray | None = None,
    freeze_from_final_case: bool = False,
    parallel: bool = True,
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
        ``incidence_vec`` may then also be a *sequence* of series (one per calculation
        time), paired with an explicit ``t_calc`` of matching length.
    t_calc : int or IntArray, optional
        Calculation time(s) (day) at which to report the additional-case probability;
        defaults to every day of the series. Required (and matched one-to-one) when
        ``incidence_vec`` is a sequence of series.
    reporting_prob : float, optional
        Case-reporting probability. If given, the under-reporting offshoot is fit with a
        latent true-case vector instead of the full-reporting model.
    delay_cdf : FloatArray, optional
        Onset-to-report delay CDF; combined with ``reporting_prob`` it adds right-truncation
        (nowcasting) to the under-reporting model.
    freeze_from_final_case : bool
        If True, hold the reproduction number fixed at its final-case-day posterior
        samples after the final observed case.
    parallel : bool
        If True (and ``quasi_real_time=True``), fit the per-snapshot models across processes
        with joblib. No effect on a single (non-quasi-real-time) fit.
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
    rep_no_vec_func = build_ar_rep_no(
        prior_median=prior_median,
        prior_percentile_2_5=prior_percentile_2_5,
        rho=rho,
    )
    return _fit_model(
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        quasi_real_time=quasi_real_time,
        t_calc=t_calc,
        reporting_prob=reporting_prob,
        delay_cdf=delay_cdf,
        freeze_from_final_case=freeze_from_final_case,
        parallel=parallel,
        compute_diagnostics=compute_diagnostics,
        raise_on_poor_diagnostics=raise_on_poor_diagnostics,
        **kwargs,
    )


def fit_suitability_model(
    *,
    incidence_vec: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    suitability_mean_vec: list[float] | FloatArray,
    suitability_std: float | None = None,
    suitability_rho: float | None = None,
    rep_no_factor_prior_median: float | None = None,
    rep_no_factor_prior_percentile_2_5: float | None = None,
    log_rep_no_factor_rho: float | None = None,
    quasi_real_time: bool = False,
    t_calc: int | IntArray | None = None,
    reporting_prob: float | None = None,
    delay_cdf: FloatArray | None = None,
    parallel: bool = True,
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
        Prior mean suitability at each time. For the under-reporting offshoot it must extend
        ``serial_interval`` days beyond the incidence so R_t can be projected forward.
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
        ``incidence_vec`` may then also be a *sequence* of series (one per calculation
        time), paired with an explicit ``t_calc`` of matching length.
    t_calc : int or IntArray, optional
        Calculation time(s) (day) at which to report the additional-case probability;
        defaults to every day of the series. Required (and matched one-to-one) when
        ``incidence_vec`` is a sequence of series.
    reporting_prob : float, optional
        Case-reporting probability. If given, the under-reporting offshoot is fit with a
        latent true-case vector instead of the full-reporting model.
    delay_cdf : FloatArray, optional
        Onset-to-report delay CDF; combined with ``reporting_prob`` it adds right-truncation
        (nowcasting) to the under-reporting model.
    parallel : bool
        If True (and ``quasi_real_time=True``), fit the per-snapshot models across processes
        with joblib. No effect on a single (non-quasi-real-time) fit.
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
    rep_no_vec_func = build_suitability_rep_no(
        suitability_mean_vec=suitability_mean_vec,
        suitability_std=suitability_std,
        suitability_rho=suitability_rho,
        rep_no_factor_prior_median=rep_no_factor_prior_median,
        rep_no_factor_prior_percentile_2_5=rep_no_factor_prior_percentile_2_5,
        log_rep_no_factor_rho=log_rep_no_factor_rho,
    )
    ds_posterior = _fit_model(
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        quasi_real_time=quasi_real_time,
        t_calc=t_calc,
        reporting_prob=reporting_prob,
        delay_cdf=delay_cdf,
        parallel=parallel,
        compute_diagnostics=compute_diagnostics,
        raise_on_poor_diagnostics=raise_on_poor_diagnostics,
        **kwargs,
    )
    # Extract further summary stats
    ds_posterior = ds_posterior.assign(
        {
            f"{var}_{stat}": _posterior_summary(ds_posterior[var], stat)
            for var in ("suitability", "rep_no_factor")
            for stat in ("mean", "lower", "upper")
        }
    )
    return ds_posterior


def fit_known_rep_no_model(
    *,
    incidence_vec: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    rep_no_func: Callable[[IntArray], FloatArray],
    quasi_real_time: bool = False,
    t_calc: int | IntArray | None = None,
    reporting_prob: float | None = None,
    delay_cdf: FloatArray | None = None,
    parallel: bool = True,
    compute_diagnostics: bool = True,
    raise_on_poor_diagnostics: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    """
    Fit the renewal model with the reproduction number fixed to a known function of time.

    The reproduction number is held at ``rep_no_func`` (constant across draws) rather than
    inferred, so only the remaining random variables are estimated. Combined with a
    ``reporting_prob`` this isolates the under-reporting (latent-case) inference from
    reproduction-number estimation.

    Parameters
    ----------
    incidence_vec : IncidenceSeriesInput
        Observed incidence time series.
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    rep_no_func : Callable[[IntArray], FloatArray]
        Function returning the known reproduction number at each of a vector of times
        (days). Evaluated internally over the full inference horizon.
    quasi_real_time : bool
        If True, refit using only the data available up to each successive time point.
        ``incidence_vec`` may then also be a *sequence* of series (one per calculation
        time), paired with an explicit ``t_calc`` of matching length.
    t_calc : int or IntArray, optional
        Calculation time(s) (day) at which to report the additional-case probability;
        defaults to every day of the series. Required (and matched one-to-one) when
        ``incidence_vec`` is a sequence of series.
    reporting_prob : float, optional
        Case-reporting probability. If given, the under-reporting offshoot is fit with a
        latent true-case vector instead of the full-reporting model.
    delay_cdf : FloatArray, optional
        Onset-to-report delay CDF; combined with ``reporting_prob`` it adds right-truncation
        (nowcasting) to the under-reporting model.
    parallel : bool
        If True (and ``quasi_real_time=True``), fit the per-snapshot models across processes
        with joblib. No effect on a single (non-quasi-real-time) fit.
    compute_diagnostics : bool
        If True, compute convergence diagnostics, attach them to the returned dataset's
        ``attrs["diagnostics"]``, and warn on poor sampling. Reproduction-number diagnostics
        are degenerate here (it is fixed), but any latent variables are still checked.
    raise_on_poor_diagnostics : bool
        If True, raise (instead of warning) when sampling diagnostics are poor.
    **kwargs : Any
        Additional keyword arguments forwarded to the sampler.

    Returns
    -------
    xr.Dataset
        Posterior dataset including the fixed reproduction number, summary statistics, and
        the daily probability of additional cases.
    """
    rep_no_vec_func = build_known_rep_no(rep_no_func=rep_no_func)
    return _fit_model(
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        quasi_real_time=quasi_real_time,
        t_calc=t_calc,
        reporting_prob=reporting_prob,
        delay_cdf=delay_cdf,
        parallel=parallel,
        compute_diagnostics=compute_diagnostics,
        raise_on_poor_diagnostics=raise_on_poor_diagnostics,
        **kwargs,
    )


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
    # Subset to the single variable first: arviz's var-name resolution mis-handles an
    # exact name when the dataset also carries derived vars that share it as a prefix
    # (e.g. `cases` alongside `cases_mean`) together with time-only summaries.
    posterior_var = posterior[[var_name]]
    rhat_vals = rhat(posterior_var, var_names=var_name)[var_name]
    ess_vals = ess(posterior_var, var_names=var_name)[var_name]
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


def _is_incidence_sequence(incidence_vec: Any) -> bool:
    # Distinguish a *sequence of incidence series* (one series per calculation time) from a
    # *single* incidence series. A single series is a 1-D array or a flat list of counts; a
    # sequence is a list/tuple whose elements are themselves array-like, or a 2-D array. Keying
    # off the container type alone would misread a single series passed as a plain list (e.g.
    # ``[1, 2, 3]``) as several one-element series.
    if isinstance(incidence_vec, np.ndarray):
        return incidence_vec.ndim >= 2
    if isinstance(incidence_vec, (list, tuple)):
        return len(incidence_vec) > 0 and all(np.ndim(v) >= 1 for v in incidence_vec)
    return False


def _fit_model(
    *,
    incidence_vec: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    rep_no_vec_func: Callable[[int], Any],
    quasi_real_time: bool,
    t_calc: int | IntArray | None = None,
    step_func: Callable[[], Any] | None = None,
    thin: int = 1,
    rng: np.random.Generator | int | None = None,
    reporting_prob: float | None = None,
    delay_cdf: FloatArray | None = None,
    freeze_from_final_case: bool = False,
    parallel: bool = True,
    compute_diagnostics: bool = False,
    raise_on_poor_diagnostics: bool = False,
    **kwargs_sample: Any,
) -> xr.Dataset:
    # Core model fit. Reproduction numbers are inferred over a horizon derived from the
    # calculation times `t_calc` (the days at which the additional-case probability is
    # reported); the returned dataset is sliced to those times. A scalar `reporting_prob`
    # (optionally with a `delay_cdf` for right-truncation) dispatches to the under-reporting
    # offshoot; otherwise the full-reporting renewal model is fit.
    #
    # Two distinct day indices are in play and are deliberately independent: the reporting
    # "as-of" day (how much delayed reporting has accrued) is fixed by the last day of
    # `incidence_vec`, whereas `t_calc` is the day from which the additional-case probability is
    # projected. The under-reporting nowcast caller pairs an end-of-day-`d` snapshot with
    # `t_calc = d + 1` (a start-of-next-day decision); nothing here requires the two to coincide.
    if quasi_real_time:
        if freeze_from_final_case:
            raise NotImplementedError(
                "freeze_from_final_case is not supported with quasi_real_time=True"
            )
        return _fit_model_qrt(
            incidence_vec=incidence_vec,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rep_no_vec_func=rep_no_vec_func,
            t_calc=t_calc,
            reporting_prob=reporting_prob,
            delay_cdf=delay_cdf,
            step_func=step_func,
            thin=thin,
            rng=rng,
            parallel=parallel,
            compute_diagnostics=compute_diagnostics,
            raise_on_poor_diagnostics=raise_on_poor_diagnostics,
            **kwargs_sample,
        )

    if _is_incidence_sequence(incidence_vec):
        raise ValueError("a sequence of incidence series requires quasi_real_time=True")
    incidence_vec = np.asarray(incidence_vec)
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec)
    t_data_to = len(incidence_vec)
    t_calc = np.atleast_1d(
        np.arange(t_data_to) if t_calc is None else np.asarray(t_calc)
    ).astype(int)
    underreporting_fit = reporting_prob is not None
    t_infer_rep_no_to = _reproduction_number_horizon(
        incidence_vec=incidence_vec,
        serial_interval_max=len(serial_interval_dist_vec),
        t_calc=t_calc,
        underreporting_fit=underreporting_fit,
    )

    # Resolve the sampler kwargs once, here, so there is a single place to change them:
    # nutpie NUTS (with low_rank mass-matrix adaptation) is used only on the full-reporting
    # path. The under-reporting offshoot carries a discrete latent, which nutpie cannot sample,
    # so it falls back to PyMC's native compound NUTS + Metropolis (assigned automatically by
    # pm.sample); low_rank adaptation is a nutpie feature and is unavailable there.
    if underreporting_fit:
        kwargs_sample = {"draws": 2000, "tune": 2000, "chains": 4, **kwargs_sample}
    else:
        kwargs_sample = {
            "nuts_sampler": "nutpie",
            "nuts": {"adaptation": "low_rank"},
            "draws": 1000,
            "tune": 1000,
            "chains": 4,
            **kwargs_sample,
        }
    if rng is not None:
        kwargs_sample = {**kwargs_sample, "random_seed": rng}

    # Constructing the PyMC model is the only path-specific step; everything below is shared.
    if underreporting_fit:
        if freeze_from_final_case:
            raise NotImplementedError(
                "freeze_from_final_case is not supported with reporting_prob"
            )
        model = _build_underreporting_model(
            incidence_vec=incidence_vec,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rep_no_vec_func=rep_no_vec_func,
            reporting_prob=reporting_prob,
            delay_cdf=delay_cdf,
            t_infer_rep_no_to=t_infer_rep_no_to,
        )
    else:
        model = _build_full_reporting_model(
            incidence_vec=incidence_vec,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rep_no_vec_func=rep_no_vec_func,
            t_infer_rep_no_to=t_infer_rep_no_to,
        )

    with model:
        # The offshoot's discrete latent gets a Metropolis step (and the continuous R_t block
        # NUTS) assigned automatically by pm.sample, so only the full-reporting step_func is
        # attached explicitly here.
        if step_func is not None:
            kwargs_sample = {"step": step_func(), **kwargs_sample}
        trace = pm.sample(**kwargs_sample)

    sample_stats = (
        trace.sample_stats
        if compute_diagnostics and "diverging" in trace.sample_stats.data_vars
        else None
    )
    if thin > 1:
        trace = trace.isel(draw=slice(0, None, thin))
        trace = trace.assign_coords(draw=np.arange(len(trace.posterior.draw)))
    ds_posterior = azb.convert_to_dataset(trace.posterior)

    if freeze_from_final_case:
        # Hold R_t after the final case fixed at its final-case-day posterior samples
        t_final_case = _final_case_time(incidence_vec)
        rep_no_frozen = ds_posterior["rep_no"].isel(time=t_final_case)
        ds_posterior = ds_posterior.assign(
            rep_no=ds_posterior["rep_no"].where(
                ds_posterior["time"] <= t_final_case, rep_no_frozen
            )
        )

    # Posterior summaries for R_t (and, for the offshoot, the latent true cases).
    summary_vars = ["rep_no", "cases"] if underreporting_fit else ["rep_no"]
    ds_posterior = ds_posterior.assign(
        {
            f"{var}_{stat}": _posterior_summary(ds_posterior[var], stat)
            for var in summary_vars
            for stat in ("mean", "lower", "upper")
        }
    )

    # Additional-case probability at the calculation times. For the offshoot each draw
    # supplies its own latent true cases, aligned with its R_t; for full reporting the fixed
    # incidence is pushed through the posterior R_t (sample dimensions averaged over).
    rep_no_grid = ds_posterior["rep_no"].transpose("time", ...).values

    def rep_no_post_func(t: int | IntArray) -> RepNoOutput:
        return rep_no_from_grid(t, rep_no_grid=rep_no_grid, periodic=False)

    if underreporting_fit:
        # Each draw supplies its own latent true cases, aligned (across chain/draw) with its R_t.
        incidence_for_prob = np.rint(
            ds_posterior["cases"].transpose("time", ...).values
        ).astype(int)
    else:
        incidence_for_prob = incidence_vec

    # Keep the per-sample probabilities (additional_dims="broadcast") so we can report the
    # posterior mean and a credible interval, not just the mean.
    prob_samples = np.asarray(
        calc_additional_case_prob_analytical(
            incidence_vec=incidence_for_prob,
            rep_no_func=rep_no_post_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=t_calc,
            additional_dims="broadcast",
        )
    ).reshape(len(t_calc), -1)
    ds_posterior = ds_posterior.isel(time=t_calc)
    ds_posterior = ds_posterior.assign(
        additional_case_prob=(("time",), prob_samples.mean(axis=1)),
        additional_case_prob_lower=(
            ("time",),
            np.quantile(prob_samples, 0.025, axis=1),
        ),
        additional_case_prob_upper=(
            ("time",),
            np.quantile(prob_samples, 0.975, axis=1),
        ),
    )

    if compute_diagnostics:
        diagnostics = _compute_check_diagnostics(
            ds_posterior, sample_stats, raise_on_problems=raise_on_poor_diagnostics
        )
        if underreporting_fit:
            # The latent true-case vector has a degenerate post-outbreak tail, so warn
            # (never raise) on its diagnostics; report the median mixing summaries.
            cases_diag = _compute_check_diagnostics(
                ds_posterior, None, var_name="cases", raise_on_problems=False
            )
            diagnostics.update(
                {
                    f"cases_{k}": cases_diag[k]
                    for k in ("rhat_max", "ess_min", "ess_median")
                }
            )
        ds_posterior.attrs["diagnostics"] = diagnostics
    return ds_posterior


def _posterior_summary(data_array: xr.DataArray, stat: str) -> xr.DataArray:
    # Posterior mean / 2.5% / 97.5% summary over the chain and draw dimensions.
    if stat == "mean":
        return data_array.mean(dim=["chain", "draw"])
    quantile = 0.025 if stat == "lower" else 0.975
    return data_array.quantile(quantile, dim=["chain", "draw"]).drop_vars("quantile")


def _final_case_time(incidence_vec: IntArray) -> int:
    # Outbreak day of the last observed case (0 if there are none).
    nonzero_incidence_idx = np.nonzero(incidence_vec)[0]
    return int(nonzero_incidence_idx[-1]) if nonzero_incidence_idx.size else 0


def _reproduction_number_horizon(
    *,
    incidence_vec: IntArray,
    serial_interval_max: int,
    t_calc: IntArray,
    underreporting_fit: bool,
) -> int:
    # Horizon to which R_t must be inferred so the additional-case projection (which needs R_t
    # up to a source case plus the serial interval) is covered — never short of the latest
    # calculation time. The two paths differ in which case is the last possible source:
    #   full reporting: the last *observed* case, so project a serial interval past it;
    #   under-reporting: the last *true* (latent) case can sit anywhere in the observed window,
    #     so project a serial interval past *all* data (carrying any seasonal R_t decline).
    t_data_to = len(incidence_vec)
    if underreporting_fit:
        base = t_data_to + serial_interval_max
    else:
        base = _final_case_time(incidence_vec) + serial_interval_max + 1
    return max(t_data_to, base, int(np.max(t_calc)) + 1)


def _build_full_reporting_model(
    *,
    incidence_vec: IntArray,
    serial_interval_dist_vec: FloatArray,
    rep_no_vec_func: Callable[[int], Any],
    t_infer_rep_no_to: int,
) -> pm.Model:
    # Build the full-reporting renewal model (renewal force of infection as a precomputed
    # convolution, observed incidence via a Poisson likelihood). R_t is inferred to
    # `t_infer_rep_no_to` (see `_reproduction_number_horizon`).
    t_data_to = len(incidence_vec)

    incidence_vec_local = np.zeros(t_data_to)
    incidence_vec_local[1:] = incidence_vec[1:]

    foi_vec = (
        renewal_convolution_matrix(serial_interval_dist_vec, t_data_to) @ incidence_vec
    )

    nonzero_foi_idx = foi_vec > 0
    if np.any(incidence_vec_local[~nonzero_foi_idx]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of infection is 0."
        )

    model = pm.Model(coords={"time": np.arange(t_infer_rep_no_to)})
    with model:
        rep_no_vec = rep_no_vec_func(t_infer_rep_no_to)
        expected_incidence_local = rep_no_vec[:t_data_to] * foi_vec
        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[nonzero_foi_idx],
            observed=incidence_vec_local[nonzero_foi_idx],
        )
    return model


def _reporting_prob_vec(
    incidence_vec: IntArray, reporting_prob: float, delay_cdf: FloatArray | None
) -> FloatArray:
    # Per-day effective reporting probability over onset days 0..(t_data_to - 1), as seen from an
    # "as-of" day equal to the last day of this incidence snapshot (t_data_to - 1) — i.e. the
    # snapshot encodes reporting known by the end of that day. The as-of day is set purely by the
    # length of `incidence_vec` and is independent of `t_calc` (the day the additional-case
    # probability is later projected). Without a delay CDF the probability is a constant
    # `reporting_prob` (pure under-reporting). With one it is
    # `reporting_prob * P(delay <= as_of_day - onset_day)`, so recent onset days (small available
    # delay) are truncated toward zero (right-truncation / nowcasting) while old onset days plateau
    # at `reporting_prob`.
    t_data_to = len(incidence_vec)
    if delay_cdf is None:
        return np.full(t_data_to, float(reporting_prob))
    delay_cdf = np.asarray(delay_cdf, dtype=float)
    as_of_day = t_data_to - 1
    available_delay = as_of_day - np.arange(t_data_to)
    return (
        float(reporting_prob)
        * delay_cdf[np.clip(available_delay, 0, len(delay_cdf) - 1)]
    )


def _build_underreporting_model(
    *,
    incidence_vec: IntArray,
    serial_interval_dist_vec: FloatArray,
    rep_no_vec_func: Callable[[int], Any],
    reporting_prob: float,
    delay_cdf: FloatArray | None,
    t_infer_rep_no_to: int,
) -> pm.Model:
    # Build the fixed-index Poisson-thinning under-reporting model. With per-day reporting
    # probability pi_t, the true cases N_t by symptom-onset date follow the renewal process and
    # are thinned into reported and unreported counts:
    #     c_t ~ Poisson(pi_t * mu_t),  N_t ~ Poisson(mu_t),  mu_t = R_t * FOI_t(N).
    # The latent unreported cases U = N - c carry the self-referential renewal density via a
    # single pm.CustomDist ("unobserved"); the reported cases are a clean top-level pm.Poisson
    # ("obs"). The first reported case(s) are the fixed index (no hidden day-0 infections), so
    # only t >= 1 carries latent cases. R_t is inferred to `t_infer_rep_no_to` (see
    # `_reproduction_number_horizon`). The discrete latent gets a Metropolis step from pm.sample
    # automatically (NUTS handles the continuous R_t block), so no step is attached by the caller.
    observed_vec = np.asarray(incidence_vec, dtype=int)
    t_data_to = len(observed_vec)
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec, dtype=float)
    n_pad = t_infer_rep_no_to - t_data_to

    # Floor the per-day reporting probability above zero so (1 - pi) / pi stays finite and pi * mu
    # is non-degenerate in the likelihoods below.
    reporting_prob_vec = np.clip(
        _reporting_prob_vec(observed_vec, reporting_prob, delay_cdf), 1e-6, 1.0
    )

    index_incidence = float(max(int(observed_vec[0]), 1))
    conv_mat = renewal_convolution_matrix(serial_interval_dist_vec, t_data_to)
    observed_after_index = observed_vec[1:].astype(float)
    reporting_prob_after_index = reporting_prob_vec[1:]
    index_col = pt.as_tensor_variable([index_incidence])

    model = pm.Model(
        coords={
            "time": np.arange(t_infer_rep_no_to),
            "gen_time": np.arange(1, t_data_to),
        }
    )
    with model:
        rep_no_vec = rep_no_vec_func(t_infer_rep_no_to)
        rep_no_data = rep_no_vec[:t_data_to]

        def _logp(value: Any, rep_no_data: Any) -> Any:
            cases = pt.concatenate([index_col, observed_after_index + value])
            mu = rep_no_data * pt.dot(conv_mat, cases)
            # +1e-12 keeps the Poisson mu > 0 where FOI is 0 (day 0 / no sources) or pi == 1.
            return pm.logp(
                pm.Poisson.dist(mu=(1 - reporting_prob_after_index) * mu[1:] + 1e-12),
                value,
            )

        def _dist(rep_no_data: Any, size: Any) -> Any:
            # Initial values / prior-predictive only (the density is overridden by _logp), so this
            # does not change the target posterior. Moment-match the latent unreported cases to the
            # reported count, U ~ Poisson(reported * (1 - pi) / pi); the floor only keeps mu > 0 for
            # the draw (`size` is unused — the latent length is fixed via `shape`).
            return pm.Poisson.dist(
                mu=np.maximum(
                    observed_after_index
                    * (1 - reporting_prob_after_index)
                    / reporting_prob_after_index,
                    1e-3,
                ),
                shape=t_data_to - 1,
            )

        latent_rv = pm.CustomDist(
            "unobserved",
            rep_no_data,
            logp=_logp,
            dist=_dist,
            dtype="int64",
            dims=("gen_time",),
        )
        cases_data = pt.concatenate([index_col, observed_after_index + latent_rv])
        pm.Deterministic(
            "cases",
            pt.concatenate([cases_data, pt.zeros((n_pad,))]) if n_pad else cases_data,
            dims=("time",),
        )
        mu_vec = rep_no_data * pt.dot(conv_mat, cases_data)
        pm.Poisson(
            "obs",
            mu=reporting_prob_after_index * mu_vec[1:] + 1e-12,
            observed=observed_vec[1:],
        )
    return model


def _fit_model_qrt(
    *,
    incidence_vec: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    rep_no_vec_func: Callable[[int], Any],
    t_calc: int | IntArray | None = None,
    reporting_prob: float | None = None,
    delay_cdf: FloatArray | None = None,
    step_func: Callable[[], Any] | None = None,
    thin: int = 1,
    rng: np.random.Generator | int | None = None,
    parallel: bool = True,
    compute_diagnostics: bool = False,
    raise_on_poor_diagnostics: bool = False,
    **kwargs_sample: Any,
) -> xr.Dataset:
    # Quasi-real-time loop: refit using only the data available at each calculation time and
    # keep that time's slice. Two input shapes:
    #   - a single incidence series -> for calc time t, fit the data up to day t-1
    #     (`incidence_vec[:t]`) and report the probability at day t (full reporting);
    #   - a sequence of series (one per calculation time) -> the right-truncated data known at
    #     each snapshot (the under-reporting nowcast). Each series' reporting as-of day is its own
    #     last day (see `_reporting_prob_vec`); the matching `t_calc` is supplied separately and
    #     need not equal it (the nowcast passes `t_calc = as_of_day + 1`).
    # The per-snapshot fits are independent, so with `parallel` they are run across processes via
    # joblib (mirroring calc_additional_case_prob_simulation); each fit then samples chains
    # sequentially (cores=1) so joblib, not pm.sample, owns the parallelism.
    sequence_mode = _is_incidence_sequence(incidence_vec)
    extra_kwargs: dict[str, Any] = {"progressbar": False}
    if reporting_prob is None:
        extra_kwargs["quiet"] = True

    if sequence_mode:
        if t_calc is None:
            raise ValueError(
                "t_calc (the calculation times) is required for sequence-of-series input"
            )
        incidence_list = [np.asarray(v) for v in incidence_vec]
        calc_times = np.atleast_1d(np.asarray(t_calc)).astype(int)
        if len(incidence_list) != len(calc_times):
            raise ValueError(
                "incidence_vec sequence and t_calc must have the same length"
            )
        steps: list[tuple[IntArray, int | IntArray]] = [
            (incidence_list[i], int(calc_times[i])) for i in range(len(calc_times))
        ]
    else:
        if t_calc is not None:
            raise ValueError(
                "t_calc is determined automatically for single-series quasi-real-time "
                "inference; pass a sequence of series for explicit calculation times"
            )
        incidence_vec = np.asarray(incidence_vec)
        if len(incidence_vec) < 2:
            raise ValueError(
                "quasi_real_time inference requires at least 2 time points"
            )
        steps = [
            (
                incidence_vec[:t],
                np.array([0, 1]) if t == 1 else t,
            )
            for t in range(1, len(incidence_vec))
        ]

    # One child RNG per fit so results depend on spawn order, not execution order (mirrors
    # calc_additional_case_prob_simulation); serial and parallel then agree exactly. A bare int
    # or None seed is replicated to preserve the previous shared-seed behaviour.
    if isinstance(rng, np.random.Generator):
        child_rngs: list[np.random.Generator | int | None] = list(rng.spawn(len(steps)))
    else:
        child_rngs = [rng] * len(steps)

    fit_kwargs_shared: dict[str, Any] = {
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "rep_no_vec_func": rep_no_vec_func,
        "quasi_real_time": False,
        "reporting_prob": reporting_prob,
        "delay_cdf": delay_cdf,
        "step_func": step_func,
        "thin": thin,
        **extra_kwargs,
        **kwargs_sample,
    }
    if parallel:
        # Chains sequential within a fit; joblib owns the cross-fit parallelism.
        fit_kwargs_shared = {"cores": 1, **fit_kwargs_shared}
    tasks = [
        (
            idx,
            {
                **fit_kwargs_shared,
                "incidence_vec": incidence_vec_step,
                "t_calc": t_calc_step,
                "rng": child_rng,
            },
        )
        for idx, ((incidence_vec_step, t_calc_step), child_rng) in enumerate(
            zip(steps, child_rngs, strict=True)
        )
    ]

    desc = "Inferring R_t using data up to each time"
    if parallel:
        # inner_max_num_threads pins the numeric threadpools in each worker so a single-core fit
        # doesn't spawn its own pool on top of joblib's process parallelism (loky is required for
        # inner_max_num_threads to take effect).
        with parallel_config(backend="loky", inner_max_num_threads=1):
            results = list(
                tqdm(
                    Parallel(
                        n_jobs=-1,
                        return_as="generator",
                        batch_size="auto",
                    )(delayed(_fit_model_qrt_step)(task, True) for task in tasks),
                    total=len(tasks),
                    desc=desc,
                )
            )
    else:
        results = list(
            tqdm(
                (_fit_model_qrt_step(task, False) for task in tasks),
                total=len(tasks),
                desc=desc,
            )
        )

    ds_posterior_list: list[xr.Dataset | None] = [None] * len(tasks)
    for idx, ds_subset in results:
        ds_posterior_list[idx] = ds_subset
    ds_posterior = xr.concat(ds_posterior_list, dim="time")
    if compute_diagnostics:
        ds_posterior.attrs["diagnostics"] = _compute_check_diagnostics(
            ds_posterior, None, raise_on_problems=raise_on_poor_diagnostics
        )
    return ds_posterior


# Guards the one-time per-worker environment setup below.
_WORKER_ENV_READY = False


def _prepare_qrt_worker() -> None:
    # Isolate this worker's PyTensor compile dir: the compile FileLock is taken on
    # `config.compiledir/.lock` (read dynamically per compilation), so sharing one dir across
    # workers would serialise the C-compilation phase and defeat the parallelism. Pin the numeric
    # threadpools too, belt-and-braces alongside joblib's inner_max_num_threads.
    global _WORKER_ENV_READY
    if _WORKER_ENV_READY:
        return
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    worker_compiledir = (
        Path(pytensor.config.base_compiledir) / f"qrt_worker_{os.getpid()}"
    )
    worker_compiledir.mkdir(parents=True, exist_ok=True)
    pytensor.config.compiledir = worker_compiledir
    # Remove the per-worker compile dir when the worker process exits so they don't accumulate
    # under base_compiledir across runs.
    atexit.register(shutil.rmtree, worker_compiledir, ignore_errors=True)
    _WORKER_ENV_READY = True


def _fit_model_qrt_step(
    task: tuple[int, dict[str, Any]], isolate: bool
) -> tuple[int, xr.Dataset]:
    # Fit a single quasi-real-time snapshot and keep only its time-indexed variables (the
    # chain/draw dims survive for the aggregate diagnostics). `isolate` is True when running in a
    # joblib worker process, where the PyTensor compile dir must be made per-process.
    idx, fit_kwargs = task
    if isolate:
        _prepare_qrt_worker()
    ds_posterior_curr = _fit_model(**fit_kwargs)
    ds_subset = ds_posterior_curr[
        [
            var
            for var in ds_posterior_curr.data_vars
            if "time" in ds_posterior_curr[var].dims
        ]
    ]
    return idx, cast(xr.Dataset, ds_subset)
