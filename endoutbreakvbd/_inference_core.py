"""Single-fit engine shared by the public fitting functions and the quasi-real-time loop.

``inference`` holds the public API and dispatches on ``quasi_real_time``; this module holds
the one-snapshot fit those two paths have in common, together with the convergence-diagnostic
and input-shape helpers both need. Keeping the engine here (rather than in ``inference``) is
what lets ``_inference_qrt`` — which refits one snapshot per calculation time — import it
directly, instead of importing back into the public module.
"""

import warnings
from collections.abc import Callable, Sequence
from dataclasses import dataclass
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
from endoutbreakvbd.utils import rep_no_from_grid


@dataclass(frozen=True)
class _DiagnosticComponents:
    rhat_values: FloatArray
    ess_values: FloatArray
    n_diverging: float


@dataclass(frozen=True)
class _SingleFitResult:
    posterior_ds: xr.Dataset
    rep_no_diagnostics: _DiagnosticComponents | None
    incidence_diagnostics: _DiagnosticComponents | None


def _is_incidence_sequence(incidence: Any) -> bool:
    # Distinguish a *sequence of incidence series* (one series per calculation time) from a
    # *single* incidence series. A single series is a 1-D array or a flat list of counts; a
    # sequence is a list/tuple whose elements are themselves array-like, or a 2-D array. Keying
    # off the container type alone would misread a single series passed as a plain list (e.g.
    # ``[1, 2, 3]``) as several one-element series.
    if isinstance(incidence, np.ndarray):
        return incidence.ndim >= 2
    if isinstance(incidence, (list, tuple)):
        return len(incidence) > 0 and all(np.ndim(v) >= 1 for v in incidence)
    return False


def _fit_single_model(
    *,
    incidence: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    rep_no_vec_func: Callable[[int], Any],
    step_func: Callable[[], Any] | None = None,
    thin: int = 1,
    rng: np.random.Generator | int | None = None,
    reporting_prob: float | None = None,
    delay_cdf: FloatArray | None = None,
    freeze_from_final_case: bool = False,
    compute_diagnostics: bool = False,
    raise_on_poor_diagnostics: bool = False,
    **kwargs_sample: Any,
) -> xr.Dataset:
    # Public-module-facing wrapper: a single fit applies diagnostics immediately, whereas QRT
    # calls `_fit_single_model_result` directly and combines its raw components across snapshots.
    return _finalize_single_fit_result(
        _fit_single_model_result(
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
            **kwargs_sample,
        ),
        raise_on_poor_diagnostics=raise_on_poor_diagnostics,
    )


def _fit_single_model_result(
    *,
    incidence: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    rep_no_vec_func: Callable[[int], Any],
    step_func: Callable[[], Any] | None = None,
    thin: int = 1,
    rng: np.random.Generator | int | None = None,
    reporting_prob: float | None = None,
    delay_cdf: FloatArray | None = None,
    freeze_from_final_case: bool = False,
    compute_diagnostics: bool = False,
    **kwargs_sample: Any,
) -> _SingleFitResult:
    # Fit one incidence snapshot. The additional-case probability is always reported over every day
    # of the incidence series plus one projected day past the last datapoint (`0 .. t_data_stop`),
    # the final entry being the "probability of further cases given everything observed" — the
    # decision-relevant current-day risk. Reproduction numbers are inferred over a horizon that
    # covers this projection (see `_get_t_rep_no_stop`) and the returned time-indexed variables are
    # sliced to those days. Under-reporting incidence variables retain their distinct data-only
    # axis. A scalar `reporting_prob` (optionally with a `delay_cdf` for right-truncation)
    # dispatches to the under-reporting offshoot; otherwise the full-reporting renewal model is fit.
    underreporting_fit = reporting_prob is not None
    if _is_incidence_sequence(incidence):
        raise ValueError("a sequence of incidence series requires quasi_real_time=True")
    incidence_vec = np.asarray(incidence)
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec)
    t_data_stop = len(incidence_vec)
    # Report over every observed day plus one projected day past the data (the current-day risk).
    t_calc_vec = np.arange(t_data_stop + 1)
    t_rep_no_stop = _get_t_rep_no_stop(
        incidence_vec=incidence_vec,
        serial_interval_max=len(serial_interval_dist_vec),
        t_calc_vec=t_calc_vec,
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
        kwargs_sample = {"draws": 8000, "tune": 2000, "chains": 4, **kwargs_sample}
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
        model = _build_underreporting_model(
            incidence_vec=incidence_vec,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rep_no_vec_func=rep_no_vec_func,
            reporting_prob=reporting_prob,
            delay_cdf=delay_cdf,
            t_rep_no_stop=t_rep_no_stop,
        )
    else:
        model = _build_full_reporting_model(
            incidence_vec=incidence_vec,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rep_no_vec_func=rep_no_vec_func,
            t_rep_no_stop=t_rep_no_stop,
        )

    with model:
        # Under-reporting fits use PyMC's automatically assigned compound sampler and reject
        # step_func upstream, so an explicit step here is necessarily for full reporting.
        if step_func is not None:
            kwargs_sample = {"step": step_func(), **kwargs_sample}
        trace = pm.sample(**kwargs_sample)

    sample_stats_ds = (
        trace.sample_stats
        if compute_diagnostics and "diverging" in trace.sample_stats.data_vars
        else None
    )
    if thin > 1:
        trace = trace.isel(draw=slice(0, None, thin))
        trace = trace.assign_coords(draw=np.arange(len(trace.posterior.draw)))
    posterior_ds = azb.convert_to_dataset(trace.posterior)

    if freeze_from_final_case:
        # Hold R_t after the final case fixed at its final-case-day posterior samples
        t_final_case = _get_t_final_case(incidence_vec)
        rep_no_frozen_da = posterior_ds["rep_no"].isel(time=t_final_case)
        posterior_ds = posterior_ds.assign(
            rep_no=posterior_ds["rep_no"].where(
                posterior_ds["time"] <= t_final_case, rep_no_frozen_da
            )
        )

    # Posterior summaries for R_t (and, for the offshoot, the latent true incidence). The incidence
    # variables retain their data-only `data_time` axis; R_t and projected risk use `time`.
    summary_vars = ["rep_no", "incidence"] if underreporting_fit else ["rep_no"]
    posterior_ds = posterior_ds.assign(
        {
            f"{var}_{stat}": _posterior_summary(posterior_ds[var], stat)
            for var in summary_vars
            for stat in ("mean", "lower", "upper")
        }
    )
    posterior_diagnostic_ds = posterior_ds

    # Additional-case probability at the calculation times. For the offshoot each draw
    # supplies its own latent true incidence, aligned with its R_t; for full reporting the fixed
    # incidence is pushed through the posterior R_t (sample dimensions averaged over).
    rep_no_grid = posterior_ds["rep_no"].transpose("time", ...).values

    def rep_no_post_func(t: int | IntArray) -> RepNoOutput:
        return rep_no_from_grid(t, rep_no_grid=rep_no_grid, periodic=False)

    if underreporting_fit:
        # Each draw supplies its own latent true incidence, aligned (across chain/draw) with its
        # R_t. Incidence stops at the data boundary; the analytical calculation constructs its
        # own zero-valued future trajectory when integrating over possible additional cases.
        incidence_for_prob = np.rint(
            posterior_ds["incidence"].transpose("data_time", ...).values
        ).astype(int)
    else:
        incidence_for_prob = incidence_vec

    # Keep the per-sample probabilities (additional_dims="broadcast") so we can report the
    # posterior mean and a credible interval, not just the mean.
    prob_sample_mat = np.asarray(
        calc_additional_case_prob_analytical(
            incidence=incidence_for_prob,
            rep_no_func=rep_no_post_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=t_calc_vec,
            additional_dims="broadcast",
        )
    ).reshape(len(t_calc_vec), -1)
    posterior_ds = posterior_ds.isel(time=t_calc_vec)
    posterior_ds = posterior_ds.assign(
        additional_case_prob=(("time",), prob_sample_mat.mean(axis=1)),
        additional_case_prob_lower=(
            ("time",),
            np.quantile(prob_sample_mat, 0.025, axis=1),
        ),
        additional_case_prob_upper=(
            ("time",),
            np.quantile(prob_sample_mat, 0.975, axis=1),
        ),
    )

    rep_no_diagnostics = (
        _compute_diagnostic_components(posterior_diagnostic_ds, sample_stats_ds)
        if compute_diagnostics
        else None
    )
    incidence_diagnostics = (
        _compute_diagnostic_components(
            posterior_diagnostic_ds, None, var_name="incidence"
        )
        if compute_diagnostics and underreporting_fit
        else None
    )
    return _SingleFitResult(
        posterior_ds=posterior_ds,
        rep_no_diagnostics=rep_no_diagnostics,
        incidence_diagnostics=incidence_diagnostics,
    )


def _finalize_single_fit_result(
    result: _SingleFitResult, *, raise_on_poor_diagnostics: bool
) -> xr.Dataset:
    posterior_ds = result.posterior_ds
    if result.rep_no_diagnostics is None:
        return posterior_ds
    diagnostics = _summarize_and_check_diagnostics(
        result.rep_no_diagnostics,
        raise_on_problems=raise_on_poor_diagnostics,
    )
    if result.incidence_diagnostics is not None:
        # Apply the same caller-selected warning/raising policy to the discrete
        # Metropolis-sampled true-case block, and report its mixing summaries separately.
        incidence_diagnostics = _summarize_and_check_diagnostics(
            result.incidence_diagnostics,
            raise_on_problems=raise_on_poor_diagnostics,
        )
        diagnostics.update(
            {
                f"incidence_{key}": incidence_diagnostics[key]
                for key in ("rhat_max", "ess_min", "ess_median")
            }
        )
    posterior_ds.attrs["diagnostics"] = diagnostics
    return posterior_ds


def _posterior_summary(data_array: xr.DataArray, stat: str) -> xr.DataArray:
    # Posterior mean / 2.5% / 97.5% summary over the chain and draw dimensions.
    if stat == "mean":
        return data_array.mean(dim=["chain", "draw"])
    quantile = 0.025 if stat == "lower" else 0.975
    return data_array.quantile(quantile, dim=["chain", "draw"]).drop_vars("quantile")


def _get_t_final_case(incidence_vec: IntArray) -> int:
    # Outbreak day of the last observed case (0 if there are none).
    positive_incidence_idx_vec = np.nonzero(incidence_vec)[0]
    return int(positive_incidence_idx_vec[-1]) if positive_incidence_idx_vec.size else 0


def _get_t_rep_no_stop(
    *,
    incidence_vec: IntArray,
    serial_interval_max: int,
    t_calc_vec: IntArray,
    underreporting_fit: bool,
) -> int:
    # Horizon to which R_t must be inferred so the additional-case projection (which needs R_t
    # up to a source case plus the serial interval) is covered — never short of the latest
    # calculation time. The two paths differ in which case is the last possible source:
    #   full reporting: the last *observed* case, so project a serial interval past it;
    #   under-reporting: the last *true* (latent) case can sit anywhere in the observed window,
    #     so project a serial interval past *all* data (carrying any seasonal R_t decline).
    t_data_stop = len(incidence_vec)
    if underreporting_fit:
        base = t_data_stop + serial_interval_max
    else:
        base = _get_t_final_case(incidence_vec) + serial_interval_max + 1
    return max(t_data_stop, base, int(np.max(t_calc_vec)) + 1)


def _compute_and_check_diagnostics(
    posterior_ds: xr.Dataset,
    sample_stats_ds: xr.Dataset | None,
    *,
    raise_on_problems: bool = False,
    var_name: str = "rep_no",
    ess_min_threshold: float = 1000,
    rhat_max_threshold: float = 1.01,
) -> dict[str, float]:
    # Convenience wrapper retained for a single supplied posterior dataset.
    return _summarize_and_check_diagnostics(
        _compute_diagnostic_components(
            posterior_ds, sample_stats_ds, var_name=var_name
        ),
        raise_on_problems=raise_on_problems,
        ess_min_threshold=ess_min_threshold,
        rhat_max_threshold=rhat_max_threshold,
    )


def _compute_diagnostic_components(
    posterior_ds: xr.Dataset,
    sample_stats_ds: xr.Dataset | None,
    *,
    var_name: str = "rep_no",
) -> _DiagnosticComponents:
    # Subset to the single variable first: passing the full dataset raises a spurious
    # KeyError ("var names ... are not present") in arviz_stats 1.2.0 when the dataset also
    # carries derived variables sharing `var_name` as a prefix (e.g. `incidence` alongside
    # `incidence_mean`). `filter_vars=None` is already arviz's exact-match mode and is the
    # one that fails; "like"/"regex" raise outright, so the single-variable subset is the fix.
    posterior_var_ds = posterior_ds[[var_name]]
    rhat_da = rhat(posterior_var_ds, var_names=var_name)[var_name]
    ess_da = ess(posterior_var_ds, var_names=var_name)[var_name]
    n_diverging = (
        np.nan if sample_stats_ds is None else float(sample_stats_ds["diverging"].sum())
    )
    return _DiagnosticComponents(
        rhat_values=np.asarray(rhat_da, dtype=float).ravel(),
        ess_values=np.asarray(ess_da, dtype=float).ravel(),
        n_diverging=n_diverging,
    )


def _combine_diagnostic_components(
    components: Sequence[_DiagnosticComponents],
) -> _DiagnosticComponents:
    if not components:
        raise ValueError("at least one set of diagnostic components is required")
    n_diverging_values = np.array(
        [component.n_diverging for component in components], dtype=float
    )
    n_diverging = (
        float(n_diverging_values.sum())
        if np.all(np.isfinite(n_diverging_values))
        else np.nan
    )
    return _DiagnosticComponents(
        rhat_values=np.concatenate(
            [component.rhat_values for component in components]
        ),
        ess_values=np.concatenate([component.ess_values for component in components]),
        n_diverging=n_diverging,
    )


def _summarize_and_check_diagnostics(
    components: _DiagnosticComponents,
    *,
    raise_on_problems: bool = False,
    ess_min_threshold: float = 1000,
    rhat_max_threshold: float = 1.01,
) -> dict[str, float]:
    # Summarise convergence diagnostics and apply the warning/raising policy once. Keeping
    # extraction separate lets QRT combine every independently fitted snapshot first.
    n_diverging = components.n_diverging
    diagnostics = {
        "rhat_mean": _diagnostic_stat(components.rhat_values, "mean"),
        "rhat_median": _diagnostic_stat(components.rhat_values, "median"),
        "rhat_max": _diagnostic_stat(components.rhat_values, "max"),
        "ess_mean": _diagnostic_stat(components.ess_values, "mean"),
        "ess_median": _diagnostic_stat(components.ess_values, "median"),
        "ess_min": _diagnostic_stat(components.ess_values, "min"),
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


def _diagnostic_stat(values: FloatArray, stat: str) -> float:
    non_nan_values = values[~np.isnan(values)]
    if non_nan_values.size == 0:
        return np.nan
    if stat == "mean":
        return float(np.mean(non_nan_values))
    if stat == "median":
        return float(np.median(non_nan_values))
    if stat == "max":
        return float(np.max(non_nan_values))
    if stat == "min":
        return float(np.min(non_nan_values))
    raise ValueError(f"unknown diagnostic statistic: {stat}")
