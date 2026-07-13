import warnings
from collections.abc import Callable, Sequence
from typing import Any

import arviz_base as azb
import numpy as np
import pymc as pm
import xarray as xr
from arviz_stats import ess, rhat

from endoutbreakvbd._inference_models import (
    _build_full_reporting_model,
    _build_underreporting_model,
)
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
from endoutbreakvbd.utils import rep_no_from_grid


def fit_autoregressive_model(
    *,
    incidence_vec: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    prior_median: float | None = None,
    prior_percentile_2_5: float | None = None,
    rho: float | list[float] | None = None,
    quasi_real_time: bool = False,
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

    For a single incidence series of length ``N``, the returned time axis spans
    days ``0..N``: day ``N`` is the projected decision day one day past the data.
    With a quasi-real-time sequence of snapshots, each length-``N`` snapshot
    contributes only its projected decision day ``N``.

    Parameters
    ----------
    incidence_vec : IncidenceSeriesInput
        Observed incidence time series. When ``reporting_prob`` is supplied, the series
        must start on the index-case day and its first value must be positive.
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
        ``incidence_vec`` may then also be a *sequence* of snapshots; each snapshot
        must start on outbreak day 0, and its calculation time is inferred from its
        length.
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

    For a single incidence series of length ``N``, the returned time axis spans
    days ``0..N``: day ``N`` is the projected decision day one day past the data.
    With a quasi-real-time sequence of snapshots, each length-``N`` snapshot
    contributes only its projected decision day ``N``.

    Parameters
    ----------
    incidence_vec : IncidenceSeriesInput
        Observed incidence time series. When ``reporting_prob`` is supplied, the series
        must start on the index-case day and its first value must be positive.
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    suitability_mean_vec : list[float] or FloatArray
        Prior mean suitability over the full reproduction-number inference horizon.
        For incidence length ``N`` and serial-interval length ``S``, this is at least
        ``N + 1`` for full reporting (and can extend to one serial interval past a
        later final case), and ``N + S`` for under-reporting. For a snapshot sequence,
        size it for the longest snapshot.
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
        ``incidence_vec`` may then also be a *sequence* of snapshots; each snapshot
        must start on outbreak day 0, and its calculation time is inferred from its
        length.
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

    For a single incidence series of length ``N``, the returned time axis spans
    days ``0..N``: day ``N`` is the projected decision day one day past the data.
    With a quasi-real-time sequence of snapshots, each length-``N`` snapshot
    contributes only its projected decision day ``N``.

    Parameters
    ----------
    incidence_vec : IncidenceSeriesInput
        Observed incidence time series. When ``reporting_prob`` is supplied, the series
        must start on the index-case day and its first value must be positive.
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    rep_no_func : Callable[[IntArray], FloatArray]
        Function returning the known reproduction number at each of a vector of times
        (days). Evaluated internally over the full inference horizon.
    quasi_real_time : bool
        If True, refit using only the data available up to each successive time point.
        ``incidence_vec`` may then also be a *sequence* of snapshots; each snapshot
        must start on outbreak day 0, and its calculation time is inferred from its
        length.
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
    # Core model fit. The additional-case probability is always reported over every day of the
    # incidence series plus one projected day past the last datapoint (`0 .. t_data_to`), the
    # final entry being the "probability of further cases given everything observed" — the
    # decision-relevant current-day risk. Reproduction numbers are inferred over a horizon that
    # covers this projection (see `_reproduction_number_horizon`) and the returned dataset is
    # sliced to those days. A scalar `reporting_prob` (optionally with a `delay_cdf` for
    # right-truncation) dispatches to the under-reporting offshoot; otherwise the full-reporting
    # renewal model is fit.
    underreporting_fit = reporting_prob is not None
    if underreporting_fit and step_func is not None:
        raise ValueError(
            "step_func is not supported with reporting_prob; under-reporting fits "
            "use PyMC's automatically assigned compound sampler"
        )
    if quasi_real_time:
        if freeze_from_final_case:
            raise NotImplementedError(
                "freeze_from_final_case is not supported with quasi_real_time=True"
            )
        # Deferred import: `_inference_qrt` imports `_fit_model` (it refits one snapshot per
        # calculation time), so a top-level import here would cycle.
        from endoutbreakvbd._inference_qrt import _fit_model_qrt

        return _fit_model_qrt(
            incidence_vec=incidence_vec,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rep_no_vec_func=rep_no_vec_func,
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
    # Report over every observed day plus one projected day past the data (the current-day risk).
    t_calc = np.arange(t_data_to + 1)
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
    # pm.sample); low_rank adaptation is a nutpie feature and is unavailable there. The
    # Metropolis-sampled latent true-case block mixes far more slowly than the NUTS reproduction
    # number, so the under-reporting path draws more (its ESS bottleneck is the latent block).
    if underreporting_fit:
        kwargs_sample = {"draws": 4000, "tune": 2000, "chains": 4, **kwargs_sample}
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
        # Under-reporting fits use PyMC's automatically assigned compound sampler and reject
        # step_func above, so an explicit step here is necessarily for full reporting.
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
