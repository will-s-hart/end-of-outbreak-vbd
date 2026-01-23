import numpy as np
import pymc as pm
import scipy.stats


def _lognormal_params_from_median_percentile_2_5(median, percentile_2_5):
    mu = np.log(median)
    sigma = (mu - np.log(percentile_2_5)) / scipy.stats.norm.ppf(0.975)
    return {"mu": mu, "sigma": sigma}


DEFAULT_PRIORS = {
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
    rng=None,
    **kwargs_sample,
):
    if rng is not None:
        kwargs_sample = {**kwargs_sample, "random_seed": rng}
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

    with pm.Model():
        rep_no_vec = rep_no_vec_func()

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


def fit_autoregressive_model(
    *,
    incidence_vec,
    gen_time_dist_vec,
    prior_median=1.0,
    prior_percentile_2_5=0.33,
    rho=0.975,
    **kwargs,
):
    t_stop = len(incidence_vec)

    def rep_no_vec_func():
        lognormal_params = _lognormal_params_from_median_percentile_2_5(
            prior_median, prior_percentile_2_5
        )

        log_rep_no_vec = pm.AR(
            "log_rep_no_vec",
            sigma=lognormal_params["sigma"] * np.sqrt(1 - rho**2),
            rho=rho,
            shape=t_stop,
            init_dist=pm.Normal.dist(
                mu=lognormal_params["mu"], sigma=lognormal_params["sigma"]
            ),
        )
        rep_no_vec = pm.Deterministic(
            "rep_no_vec",
            pm.math.exp(log_rep_no_vec),
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
    gen_time_dist_vec,
    suitability_mean_vec,
    suitability_std=0.05,
    suitability_rho=0.975,
    rep_no_factor_model="autoregressive",
    random_walk_std=0.05,
    rep_no_factor_prior_median=2.0,
    rep_no_factor_prior_percentile_2_5=0.5,
    log_rep_no_factor_rho=0.975,
    **kwargs,
):
    t_stop = len(incidence_vec)

    def rep_no_vec_func():
        rep_no_factor_lognormal_params = _lognormal_params_from_median_percentile_2_5(
            rep_no_factor_prior_median, rep_no_factor_prior_percentile_2_5
        )
        if rep_no_factor_model == "autoregressive":
            log_rep_no_factor_vec = pm.AR(
                "log_rep_no_factor_vec",
                sigma=rep_no_factor_lognormal_params["sigma"]
                * np.sqrt(1 - log_rep_no_factor_rho**2),
                rho=log_rep_no_factor_rho,
                shape=t_stop,
                init_dist=pm.Normal.dist(
                    mu=rep_no_factor_lognormal_params["mu"],
                    sigma=rep_no_factor_lognormal_params["sigma"],
                ),
            )
            rep_no_factor_vec = pm.Deterministic(
                "rep_no_factor_vec", pm.math.exp(log_rep_no_factor_vec)
            )
        elif rep_no_factor_model == "random_walk":
            log_rep_no_factor_start = pm.Normal(
                "log_rep_no_factor_start",
                mu=rep_no_factor_lognormal_params["mu"],
                sigma=rep_no_factor_lognormal_params["sigma"],
            )
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
        else:
            raise ValueError(
                "rep_no_factor_model must be one of 'autoregressive' or 'random_walk'"
            )
        suitability_deviations = pm.AR(
            "suitability_ext",
            sigma=suitability_std * np.sqrt(1 - suitability_rho**2),
            rho=suitability_rho,
            shape=t_stop,
            init_dist=pm.Normal.dist(mu=0, sigma=suitability_std),
        )
        suitability_vec = pm.Deterministic(
            "suitability_vec",
            pm.math.clip(suitability_mean_vec + suitability_deviations, 1e-8, 1),
        )
        rep_no_vec = pm.Deterministic("rep_no_vec", rep_no_factor_vec * suitability_vec)
        return rep_no_vec

    return _fit_model(
        incidence_vec=incidence_vec,
        gen_time_dist_vec=gen_time_dist_vec,
        rep_no_vec_func=rep_no_vec_func,
        **kwargs,
    )
