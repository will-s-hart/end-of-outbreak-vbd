"""Public renewal-model fitting API.

The three ``fit_*`` functions differ only in which reproduction-number prior they build (see
``rep_no_models``); they share the dispatcher below, which routes to the quasi-real-time loop
in ``_inference_qrt`` or to the single-snapshot engine in ``_inference_core``.
"""

from collections.abc import Callable, Sequence
from typing import Any

import numpy as np
import xarray as xr

from endoutbreakvbd._inference_core import (
    _fit_single_model,
    _is_incidence_sequence,
    _posterior_summary,
)
from endoutbreakvbd._inference_qrt import _fit_model_qrt
from endoutbreakvbd._types import (
    FloatArray,
    IncidenceSeriesInput,
    IntArray,
    SerialIntervalInput,
)
from endoutbreakvbd.rep_no_models import (
    build_ar_rep_no,
    build_known_rep_no,
    build_suitability_rep_no,
)


def fit_autoregressive_model(
    *,
    incidence: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    prior_median: float | None = None,
    prior_percentile_2_5: float | None = None,
    rho: float | list[float] | None = None,
    quasi_real_time: bool = False,
    reporting_prob: float | None = None,
    delay_cdf: FloatArray | None = None,
    freeze_from_final_case: bool = False,
    rng: np.random.Generator | int | None = None,
    thin: int = 1,
    step_func: Callable[[], Any] | None = None,
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

    For a non-quasi-real-time under-reporting fit, posterior ``incidence`` and its
    summaries use a separate ``data_time`` axis spanning days ``0..N-1``. They are
    omitted from quasi-real-time aggregates, whose snapshots have differing histories.

    Parameters
    ----------
    incidence : IncidenceSeriesInput or Sequence[IncidenceSeriesInput]
        Observed incidence time series, or (with ``quasi_real_time=True``) a sequence of
        snapshots. When ``reporting_prob`` is supplied, the series must start on the
        index-case day and its first value must be positive. With both ``quasi_real_time``
        and ``delay_cdf``, a historical snapshot sequence is required because a single final
        series cannot reconstruct earlier reporting states.
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
        ``incidence`` may then also be a *sequence* of snapshots; each snapshot
        must start on outbreak day 0, and its calculation time is inferred from its
        length. Supplying ``delay_cdf`` requires this snapshot form.
    reporting_prob : float, optional
        Case-reporting probability. If given, the under-reporting offshoot is fit with a
        latent true-case vector instead of the full-reporting model. This also **changes the
        sampler**: the discrete latent cannot be sampled by nutpie, so PyMC's compound
        NUTS + Metropolis is used, with more draws by default (``draws=4000``, ``tune=2000``
        rather than ``1000``/``1000``) because the latent block is the mixing bottleneck.
        Override via ``draws`` / ``tune`` / ``chains``.

        On a non-quasi-real-time fit the latent case history is inferred from the *whole*
        series, so the reported probability at day ``t`` is conditioned on data after ``t`` as
        well as before it — a retrospective quantity, not a real-time one. (The reproduction
        number is retrospective on both the full- and under-reporting paths.)
    delay_cdf : FloatArray, optional
        Onset-to-report delay CDF; requires ``reporting_prob`` and adds right-truncation
        (nowcasting) to the under-reporting model. After the fixed index case, a day with
        effective reporting probability zero must have zero reported cases and contributes no
        reported-case likelihood.
    freeze_from_final_case : bool
        If True, hold the reproduction number fixed at its final-case-day posterior
        samples after the final observed case.
    rng : np.random.Generator or int, optional
        Seed for the sampler. A ``Generator`` is consumed in place, so successive fits sharing
        one generator draw different seeds. Under ``quasi_real_time`` a child generator is
        spawned per snapshot, making results independent of execution order.
    thin : int
        Keep every ``thin``th posterior draw. Applied after sampling.
    step_func : Callable[[], Any], optional
        Zero-argument factory returning an explicit PyMC step method. Not supported together
        with ``reporting_prob`` (under-reporting fits rely on PyMC assigning the latent's
        Metropolis step automatically).
    parallel : bool
        If True (and ``quasi_real_time=True``), fit the per-snapshot models across processes
        with joblib. No effect on a single (non-quasi-real-time) fit.
    compute_diagnostics : bool
        If True, compute convergence diagnostics, attach them to the returned
        dataset's ``attrs["diagnostics"]``, and warn on poor sampling. Under
        ``quasi_real_time``, diagnostics aggregate every fitted reproduction-number value
        (and every fitted latent-incidence value for under-reporting) across all snapshots,
        before retaining only their decision-day outputs.
    raise_on_poor_diagnostics : bool
        If True, raise (instead of warning) when sampling diagnostics are poor. Only the
        reproduction-number diagnostics can raise; the discrete latent case block is
        warn-only, since it mixes slowly by construction.
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
        incidence=incidence,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        quasi_real_time=quasi_real_time,
        reporting_prob=reporting_prob,
        delay_cdf=delay_cdf,
        freeze_from_final_case=freeze_from_final_case,
        rng=rng,
        thin=thin,
        step_func=step_func,
        parallel=parallel,
        compute_diagnostics=compute_diagnostics,
        raise_on_poor_diagnostics=raise_on_poor_diagnostics,
        **kwargs,
    )


def fit_suitability_model(
    *,
    incidence: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
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
    rng: np.random.Generator | int | None = None,
    thin: int = 1,
    step_func: Callable[[], Any] | None = None,
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

    For a non-quasi-real-time under-reporting fit, posterior ``incidence`` and its
    summaries use a separate ``data_time`` axis spanning days ``0..N-1``. They are
    omitted from quasi-real-time aggregates, whose snapshots have differing histories.

    Parameters
    ----------
    incidence : IncidenceSeriesInput or Sequence[IncidenceSeriesInput]
        Observed incidence time series, or (with ``quasi_real_time=True``) a sequence of
        snapshots. When ``reporting_prob`` is supplied, the series must start on the
        index-case day and its first value must be positive. With both ``quasi_real_time``
        and ``delay_cdf``, a historical snapshot sequence is required because a single final
        series cannot reconstruct earlier reporting states.
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    suitability_mean_vec : list[float] or FloatArray
        Prior mean suitability over the full reproduction-number inference horizon. For
        incidence length ``N``, serial-interval length ``S`` and last non-zero incidence day
        ``t_final_case``, the required length is ``max(N + 1, t_final_case + S + 1)`` for full
        reporting, and ``N + S`` for under-reporting. Note the full-reporting requirement
        depends on *where* the last case sits, not just on ``N``: a series padded with trailing
        zeros needs less than one ending on a case. Too short a vector raises, reporting the
        required horizon. For a snapshot sequence, size it for the longest snapshot.
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
        ``incidence`` may then also be a *sequence* of snapshots; each snapshot
        must start on outbreak day 0, and its calculation time is inferred from its
        length. Supplying ``delay_cdf`` requires this snapshot form.
    reporting_prob : float, optional
        Case-reporting probability. If given, the under-reporting offshoot is fit with a
        latent true-case vector instead of the full-reporting model. This also **changes the
        sampler**: the discrete latent cannot be sampled by nutpie, so PyMC's compound
        NUTS + Metropolis is used, with more draws by default (``draws=4000``, ``tune=2000``
        rather than ``1000``/``1000``) because the latent block is the mixing bottleneck.
        Override via ``draws`` / ``tune`` / ``chains``.

        On a non-quasi-real-time fit the latent case history is inferred from the *whole*
        series, so the reported probability at day ``t`` is conditioned on data after ``t`` as
        well as before it — a retrospective quantity, not a real-time one. (The reproduction
        number is retrospective on both the full- and under-reporting paths.)
    delay_cdf : FloatArray, optional
        Onset-to-report delay CDF; requires ``reporting_prob`` and adds right-truncation
        (nowcasting) to the under-reporting model. After the fixed index case, a day with
        effective reporting probability zero must have zero reported cases and contributes no
        reported-case likelihood.
    rng : np.random.Generator or int, optional
        Seed for the sampler. A ``Generator`` is consumed in place, so successive fits sharing
        one generator draw different seeds. Under ``quasi_real_time`` a child generator is
        spawned per snapshot, making results independent of execution order.
    thin : int
        Keep every ``thin``th posterior draw. Applied after sampling.
    step_func : Callable[[], Any], optional
        Zero-argument factory returning an explicit PyMC step method. Not supported together
        with ``reporting_prob`` (under-reporting fits rely on PyMC assigning the latent's
        Metropolis step automatically).
    parallel : bool
        If True (and ``quasi_real_time=True``), fit the per-snapshot models across processes
        with joblib. No effect on a single (non-quasi-real-time) fit.
    compute_diagnostics : bool
        If True, compute convergence diagnostics, attach them to the returned
        dataset's ``attrs["diagnostics"]``, and warn on poor sampling. Under
        ``quasi_real_time``, diagnostics aggregate every fitted reproduction-number value
        (and every fitted latent-incidence value for under-reporting) across all snapshots,
        before retaining only their decision-day outputs.
    raise_on_poor_diagnostics : bool
        If True, raise (instead of warning) when sampling diagnostics are poor. Only the
        reproduction-number diagnostics can raise; the discrete latent case block is
        warn-only, since it mixes slowly by construction.
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
    posterior_ds = _fit_model(
        incidence=incidence,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        quasi_real_time=quasi_real_time,
        reporting_prob=reporting_prob,
        delay_cdf=delay_cdf,
        rng=rng,
        thin=thin,
        step_func=step_func,
        parallel=parallel,
        compute_diagnostics=compute_diagnostics,
        raise_on_poor_diagnostics=raise_on_poor_diagnostics,
        **kwargs,
    )
    # Extract further summary stats
    posterior_ds = posterior_ds.assign(
        {
            f"{var}_{stat}": _posterior_summary(posterior_ds[var], stat)
            for var in ("suitability", "rep_no_factor")
            for stat in ("mean", "lower", "upper")
        }
    )
    return posterior_ds


def fit_known_rep_no_model(
    *,
    incidence: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    rep_no_func: Callable[[IntArray], FloatArray],
    quasi_real_time: bool = False,
    reporting_prob: float | None = None,
    delay_cdf: FloatArray | None = None,
    rng: np.random.Generator | int | None = None,
    thin: int = 1,
    step_func: Callable[[], Any] | None = None,
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

    For a non-quasi-real-time under-reporting fit, posterior ``incidence`` and its
    summaries use a separate ``data_time`` axis spanning days ``0..N-1``. They are
    omitted from quasi-real-time aggregates, whose snapshots have differing histories.

    Parameters
    ----------
    incidence : IncidenceSeriesInput or Sequence[IncidenceSeriesInput]
        Observed incidence time series, or (with ``quasi_real_time=True``) a sequence of
        snapshots. When ``reporting_prob`` is supplied, the series must start on the
        index-case day and its first value must be positive. With both ``quasi_real_time``
        and ``delay_cdf``, a historical snapshot sequence is required because a single final
        series cannot reconstruct earlier reporting states.
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    rep_no_func : Callable[[IntArray], FloatArray]
        Function returning the known reproduction number at each of a vector of times
        (days). Evaluated internally over the full inference horizon.
    quasi_real_time : bool
        If True, refit using only the data available up to each successive time point.
        ``incidence`` may then also be a *sequence* of snapshots; each snapshot
        must start on outbreak day 0, and its calculation time is inferred from its
        length. Supplying ``delay_cdf`` requires this snapshot form.
    reporting_prob : float, optional
        Case-reporting probability. If given, the under-reporting offshoot is fit with a
        latent true-case vector instead of the full-reporting model. This also **changes the
        sampler**: the discrete latent cannot be sampled by nutpie, so PyMC's compound
        NUTS + Metropolis is used, with more draws by default (``draws=4000``, ``tune=2000``
        rather than ``1000``/``1000``) because the latent block is the mixing bottleneck.
        Override via ``draws`` / ``tune`` / ``chains``.

        On a non-quasi-real-time fit the latent case history is inferred from the *whole*
        series, so the reported probability at day ``t`` is conditioned on data after ``t`` as
        well as before it — a retrospective quantity, not a real-time one.
    delay_cdf : FloatArray, optional
        Onset-to-report delay CDF; requires ``reporting_prob`` and adds right-truncation
        (nowcasting) to the under-reporting model. After the fixed index case, a day with
        effective reporting probability zero must have zero reported cases and contributes no
        reported-case likelihood.
    rng : np.random.Generator or int, optional
        Seed for the sampler. A ``Generator`` is consumed in place, so successive fits sharing
        one generator draw different seeds. Under ``quasi_real_time`` a child generator is
        spawned per snapshot, making results independent of execution order.
    thin : int
        Keep every ``thin``th posterior draw. Applied after sampling.
    step_func : Callable[[], Any], optional
        Zero-argument factory returning an explicit PyMC step method. Not supported together
        with ``reporting_prob`` (under-reporting fits rely on PyMC assigning the latent's
        Metropolis step automatically).
    parallel : bool
        If True (and ``quasi_real_time=True``), fit the per-snapshot models across processes
        with joblib. No effect on a single (non-quasi-real-time) fit.
    compute_diagnostics : bool
        If True, compute convergence diagnostics, attach them to the returned dataset's
        ``attrs["diagnostics"]``, and warn on poor sampling. Under ``quasi_real_time``,
        diagnostics aggregate every fitted reproduction-number value (and every fitted
        latent-incidence value for under-reporting) across all snapshots, before retaining
        only their decision-day outputs.
    raise_on_poor_diagnostics : bool
        Has **no effect on this path**. The only variable it can raise on is the reproduction
        number, which is constant here, so its R-hat and ESS are degenerate (ESS simply equals
        the draw count) and the check can neither pass nor fail meaningfully. Latent-variable
        diagnostics are still computed and reported in ``attrs["diagnostics"]``, but are
        warn-only.
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
        incidence=incidence,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        quasi_real_time=quasi_real_time,
        reporting_prob=reporting_prob,
        delay_cdf=delay_cdf,
        rng=rng,
        thin=thin,
        step_func=step_func,
        parallel=parallel,
        compute_diagnostics=compute_diagnostics,
        raise_on_poor_diagnostics=raise_on_poor_diagnostics,
        **kwargs,
    )


def _fit_model(
    *,
    incidence: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
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
    # Reject the argument combinations neither fitting path can honour, then dispatch:
    # `quasi_real_time` refits one snapshot per calculation time (`_inference_qrt`), otherwise a
    # single snapshot is fit directly (`_inference_core`). Validating here rather than in the
    # engines keeps the checks in one place, since the quasi-real-time loop forwards these
    # arguments to the single-fit engine unchanged.
    if delay_cdf is not None:
        if reporting_prob is None:
            raise ValueError("delay_cdf requires reporting_prob")
        if quasi_real_time and not _is_incidence_sequence(incidence):
            raise ValueError(
                "delay_cdf with quasi_real_time=True requires a sequence of historical "
                "incidence snapshots; a single final series cannot reconstruct past "
                "reporting states"
            )
    if reporting_prob is not None and step_func is not None:
        raise ValueError(
            "step_func is not supported with reporting_prob; under-reporting fits "
            "use PyMC's automatically assigned compound sampler"
        )
    if freeze_from_final_case:
        if quasi_real_time:
            raise NotImplementedError(
                "freeze_from_final_case is not supported with quasi_real_time=True"
            )
        if reporting_prob is not None:
            raise NotImplementedError(
                "freeze_from_final_case is not supported with reporting_prob"
            )
    if quasi_real_time:
        return _fit_model_qrt(
            incidence=incidence,
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
    return _fit_single_model(
        incidence=incidence,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        step_func=step_func,
        thin=thin,
        rng=rng,
        reporting_prob=reporting_prob,
        delay_cdf=delay_cdf,
        freeze_from_final_case=freeze_from_final_case,
        compute_diagnostics=compute_diagnostics,
        raise_on_poor_diagnostics=raise_on_poor_diagnostics,
        **kwargs_sample,
    )
