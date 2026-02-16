from collections.abc import Callable
from dataclasses import dataclass
from typing import Any, cast

import arviz_base as azb
import numpy as np
import pymc as pm
import xarray as xr
from tqdm import tqdm

from endoutbreakvbd.types import FloatArray, GenTimeInput, IncidenceSeriesInput
from endoutbreakvbd.utils import lognormal_params_from_median_percentile_2_5


@dataclass(frozen=True)
class Defaults:
    rep_no_prior_median: float = 1.0
    rep_no_prior_percentile_2_5: float = 0.2
    rep_no_rho: float = 0.975
    suitability_std: float = 0.05
    suitability_rho: float = 0.975
    # rep_no_factor_prior_median: float = 2.5
    # rep_no_factor_prior_percentile_2_5: float = 1.25
    rep_no_factor_prior_median: float = 1.0
    rep_no_factor_prior_percentile_2_5: float = 0.2
    log_rep_no_factor_rho: float = 0.975


DEFAULTS = Defaults()


def _fit_model(
    *,
    incidence_vec: IncidenceSeriesInput,
    gen_time_dist_vec: GenTimeInput,
    rep_no_vec_func: Callable[[int], Any],
    quasi_real_time: bool,
    step_func: Callable[[], Any] | None = None,
    thin: int = 1,
    rng: np.random.Generator | int | None = None,
    **kwargs_sample: Any,
) -> xr.Dataset:
    if quasi_real_time:
        posterior_list = []
        for t in tqdm(
            range(len(incidence_vec)), desc="Inferring R_t using data up to each time"
        ):
            posterior_curr = _fit_model(
                incidence_vec=incidence_vec[: t + 1],
                gen_time_dist_vec=gen_time_dist_vec,
                rep_no_vec_func=rep_no_vec_func,
                quasi_real_time=False,
                step_func=step_func,
                thin=thin,
                rng=rng,
                **{"quiet": True, "progressbar": False, **kwargs_sample},
            )
            posterior_list.append(
                posterior_curr[
                    [
                        var
                        for var in posterior_curr.data_vars
                        if "time" in posterior_curr[var].dims
                    ]
                ].isel(time=[t])
            )
        posterior = xr.concat(posterior_list, dim="time")
        return posterior

    if rng is not None:
        kwargs_sample = {**kwargs_sample, "random_seed": rng}
    t_stop = len(incidence_vec)
    gen_time_dist_vec = np.concatenate(
        [
            gen_time_dist_vec,
            np.zeros(np.maximum(t_stop - 1 - len(gen_time_dist_vec), 0)),
        ]
    )

    incidence_vec_local = np.zeros(t_stop)
    incidence_vec_local[1:] = incidence_vec[1:]

    foi_vec = np.zeros(t_stop)
    for t in range(1, t_stop):
        foi_vec[t] = np.sum(incidence_vec[:t][::-1] * gen_time_dist_vec[:t])

    nonzero_foi_idx = foi_vec > 0
    if np.any(incidence_vec_local[~nonzero_foi_idx]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of infection is 0."
        )

    coords = {
        "time": np.arange(t_stop),
    }

    with pm.Model(coords=coords):
        rep_no_vec = rep_no_vec_func(t_stop)

        expected_incidence_local = rep_no_vec * foi_vec

        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[nonzero_foi_idx],
            observed=incidence_vec_local[nonzero_foi_idx],
        )
        if step_func is not None:
            kwargs_sample["step"] = step_func()

        trace = pm.sample(**kwargs_sample)
    if thin > 1:
        trace = trace.isel(draw=slice(0, None, thin))
        trace = trace.assign_coords(draw=np.arange(len(trace.posterior.draw)))
    return azb.convert_to_dataset(trace.posterior)


def fit_autoregressive_model(
    *,
    incidence_vec: IncidenceSeriesInput,
    gen_time_dist_vec: GenTimeInput,
    prior_median: float | None = None,
    prior_percentile_2_5: float | None = None,
    rho: float | None = None,
    quasi_real_time: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
    prior_median = prior_median or DEFAULTS.rep_no_prior_median
    prior_percentile_2_5 = prior_percentile_2_5 or DEFAULTS.rep_no_prior_percentile_2_5
    rho = rho or DEFAULTS.rep_no_rho
    lognormal_params = lognormal_params_from_median_percentile_2_5(
        median=prior_median, percentile_2_5=prior_percentile_2_5
    )

    def rep_no_vec_func(_: int) -> Any:
        log_rep_no_deviation_vec = pm.AR(
            "log_rep_no_deviation",
            sigma=lognormal_params["sigma"] * np.sqrt(1 - rho**2),
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
        gen_time_dist_vec=gen_time_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        quasi_real_time=quasi_real_time,
        **kwargs,
    )


def fit_suitability_model(
    *,
    incidence_vec: IncidenceSeriesInput,
    gen_time_dist_vec: GenTimeInput,
    suitability_mean_vec: list[float] | FloatArray,
    suitability_std: float | None = None,
    suitability_rho: float | None = None,
    rep_no_factor_prior_median: float | None = None,
    rep_no_factor_prior_percentile_2_5: float | None = None,
    log_rep_no_factor_rho: float | None = None,
    quasi_real_time: bool = False,
    **kwargs: Any,
) -> xr.Dataset:
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

    suitability_mean_vec_arr = np.asarray(suitability_mean_vec)

    def rep_no_vec_func(t_stop: int) -> Any:
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
        suitability_deviations = pm.AR(
            "suitability_deviation",
            sigma=suitability_std * np.sqrt(1 - suitability_rho**2),
            rho=suitability_rho,
            dims=("time",),
            init_dist=pm.Normal.dist(mu=0, sigma=suitability_std),
        )
        suitability_vec = pm.Deterministic(
            "suitability",
            pm.math.clip(  # need to truncate mean suitability at t_stop in QRT mode
                suitability_mean_vec_arr[:t_stop] + cast(Any, suitability_deviations),
                1e-8,
                1,
            ),
            dims=("time",),
        )
        rep_no_vec = pm.Deterministic(
            "rep_no", rep_no_factor_vec * suitability_vec, dims=("time",)
        )
        return rep_no_vec

    return _fit_model(
        incidence_vec=incidence_vec,
        gen_time_dist_vec=gen_time_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        quasi_real_time=quasi_real_time,
        **kwargs,
    )
