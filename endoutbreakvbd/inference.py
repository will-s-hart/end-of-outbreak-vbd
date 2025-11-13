import numpy as np
import pymc as pm
import pytensor.tensor as pt
import scipy.stats


def _lognormal_params_from_median_percentile_2_5(median, percentile_2_5):
    mu = np.log(median)
    sigma = (mu - np.log(percentile_2_5)) / scipy.stats.norm.ppf(0.975)
    return {"mu": mu, "sigma": sigma}


DEFAULT_PRIORS = {
    "rep_no_max": (
        pm.LogNormal,
        _lognormal_params_from_median_percentile_2_5(2, 0.1),
    ),
    "rep_no_amplitude": (
        pm.LogNormal,
        _lognormal_params_from_median_percentile_2_5(2, 0.1),
    ),
    "doy_max": (
        pm.TruncatedNormal,
        {"mu": 215, "sigma": 1, "lower": 1, "upper": 366},
    ),
    "rep_no_start": (
        pm.LogNormal,
        _lognormal_params_from_median_percentile_2_5(1, 0.2),
    ),
}


def _fit_model(
    *,
    incidence_vec,
    gen_time_dist_vec,
    rep_no_vec_func,
    step_func=None,
    thin=1,
    **kwargs_sample,
):
    t_stop = len(incidence_vec)
    gen_time_dist_vec = np.concatenate(
        [gen_time_dist_vec, np.zeros(t_stop - 1 - len(gen_time_dist_vec))]
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

    with pm.Model() as model:
        rep_no_vec = rep_no_vec_func()

        expected_incidence_local = rep_no_vec * foi_vec

        likelihood = pm.Poisson(
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
    return trace


def _extract_priors(var_names, priors=None):
    priors = {**DEFAULT_PRIORS, **(priors or {})}
    prior_list = []
    for var_name in var_names:
        prior_func, prior_kwargs = priors[var_name]
        prior_list.append(prior_func(var_name, **prior_kwargs))
    return tuple(prior_list)


def fit_random_walk_model(
    *,
    incidence_vec,
    gen_time_dist_vec,
    random_walk_std=0.05,
    priors=None,
    **kwargs,
):
    t_stop = len(incidence_vec)

    def rep_no_vec_func():
        (rep_no_start,) = _extract_priors(["rep_no_start"], priors=priors)

        log_rep_no_jumps = pm.Normal(
            "rep_no_jumps", mu=0, sigma=random_walk_std, shape=t_stop - 1
        )

        rep_no_vec = pm.Deterministic(
            "rep_no_vec",
            pm.math.exp(
                pm.math.cumsum(
                    pm.math.concatenate(
                        [[pm.math.log(rep_no_start)], log_rep_no_jumps], axis=0
                    )
                )
            ),
        )
        return rep_no_vec

    return _fit_model(
        incidence_vec=incidence_vec,
        gen_time_dist_vec=gen_time_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        **kwargs,
    )


def fit_periodic_model(
    *,
    incidence_vec,
    doy_start,
    gen_time_dist_vec,
    priors=None,
    **kwargs,
):
    priors = priors or {}
    t_stop = len(incidence_vec)
    doy_vec = (doy_start + np.arange(t_stop)) % 365

    def rep_no_vec_func():
        rep_no_max, rep_no_amplitude, doy_max = _extract_priors(
            ["rep_no_max", "rep_no_amplitude", "doy_max"], priors=priors
        )

        rep_no_vec = pm.Deterministic(
            "rep_no_vec",
            pm.math.maximum(
                rep_no_max
                - rep_no_amplitude
                + rep_no_amplitude * pm.math.cos(2 * np.pi * (doy_vec - doy_max) / 365),
                0,
            ),
        )
        return rep_no_vec

    return _fit_model(
        incidence_vec=incidence_vec,
        gen_time_dist_vec=gen_time_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        **kwargs,
    )


def fit_weather_model(
    *,
    incidence_vec,
    temperature_vec,
    gen_time_dist_vec,
    priors=None,
    **kwargs,
):
    priors = priors or {}

    def rep_no_vec_func():
        (rep_no_max,) = _extract_priors(["rep_no_max"], priors=priors)
        c_0 = pm.Normal("c_0", mu=-9, sigma=1)
        c_1 = pm.Normal("c_1", mu=0.3, sigma=0.1)
        # c_0 = pm.Uniform("c_0", lower=-10, upper=0)
        # c_1 = pm.Uniform("c_1", lower=0, upper=1)
        rep_no_vec = pm.Deterministic(
            "rep_no_vec",
            rep_no_max / (1 + pm.math.exp(-(c_0 + c_1 * temperature_vec))),
        )
        return rep_no_vec

    return _fit_model(
        incidence_vec=incidence_vec,
        gen_time_dist_vec=gen_time_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        **kwargs,
    )


def fit_suitability_model(
    *,
    incidence_vec,
    suitability_mean_vec,
    suitability_std,
    gen_time_dist_vec,
    random_walk_std=0.05,
    **kwargs,
):
    t_stop = len(incidence_vec)

    def rep_no_vec_func():
        log_rep_no_factor_start = pm.Normal("log_rep_no_factor_start", mu=0, sigma=1)
        log_rep_no_factor_jumps = pm.Normal(
            "rep_no_jumps", mu=0, sigma=random_walk_std, shape=t_stop - 1
        )
        rep_no_factor_vec = pm.Deterministic(
            "rep_no_factor_vec",
            pm.math.exp(
                pm.math.cumsum(
                    pm.math.concatenate(
                        [[log_rep_no_factor_start], log_rep_no_factor_jumps], axis=0
                    )
                )
            ),
        )
        suitability_vec_ext = pm.Normal(
            "suitability_ext",
            mu=suitability_mean_vec,
            sigma=suitability_std,
            shape=t_stop,
        )
        suitability_vec = pm.Deterministic(
            "suitability_vec",
            pm.math.clip(suitability_vec_ext, 1e-8, 1),
        )
        rep_no_vec = pm.Deterministic("rep_no_vec", rep_no_factor_vec * suitability_vec)
        return rep_no_vec

    return _fit_model(
        incidence_vec=incidence_vec,
        gen_time_dist_vec=gen_time_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        **kwargs,
    )
