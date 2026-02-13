import functools
import pathlib

import numpy as np
import pandas as pd
import scipy.stats

from endoutbreakvbd.inference import DEFAULTS
from endoutbreakvbd.utils import rep_no_from_grid


def get_inputs_weather_suitability_data():
    results_dir = pathlib.Path(__file__).parents[1] / "results/weather_suitability_data"
    results_paths = {
        "all": results_dir / "all.csv",
        "2017": results_dir / "2017.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures/weather_suitability_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "temperature": fig_dir / "temperature.svg",
        "suitability": fig_dir / "suitability.svg",
    }

    return {
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def get_inputs_sim_study():
    gen_time_dist_vec = _get_gen_time_dist()

    df_suitability = _get_2017_suitability_data()
    suitability_grid = df_suitability["suitability_smoothed"].to_numpy()

    rep_no_factor = 2
    rep_no_grid = rep_no_factor * suitability_grid

    rep_no_func_doy = functools.partial(
        rep_no_from_grid, rep_no_grid=rep_no_grid, periodic=True, doy_start=0
    )
    rep_no_from_doy_start = functools.partial(
        rep_no_from_grid, rep_no_grid=rep_no_grid, periodic=True
    )

    example_outbreak_doy_start_vals = (
        np.nonzero(rep_no_grid > 1.2)[0][0] + 1,
        np.nonzero(rep_no_grid > 1.2)[0][-1] + 1,
    )
    example_outbreak_incidence_vec = [1]
    example_outbreak_perc_risk_threshold_vals = (1, 2.5, 5)

    many_outbreak_n_sims = 100000
    many_outbreak_outbreak_size_threshold = 2
    many_outbreak_perc_risk_threshold = 5

    results_dir = pathlib.Path(__file__).parents[1] / "results/sim_study"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_paths = {
        "example_outbreak_risk": results_dir / "example_outbreak_risk.csv",
        "example_outbreak_declaration": results_dir
        / "example_outbreak_declaration.csv",
        "many_outbreak": results_dir / "many_outbreak.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures/sim_study"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "rep_no": fig_dir / "rep_no.svg",
        "example_outbreak_risk": fig_dir / "example_outbreak_risk.svg",
        "example_outbreak_declaration": fig_dir / "example_outbreak_declaration.svg",
        "many_outbreak": fig_dir / "many_outbreak.svg",
    }

    return {
        "gen_time_dist_vec": gen_time_dist_vec,
        "rep_no_factor": rep_no_factor,
        "rep_no_func_doy": rep_no_func_doy,
        "rep_no_from_doy_start": rep_no_from_doy_start,
        "example_outbreak_doy_start_vals": example_outbreak_doy_start_vals,
        "example_outbreak_incidence_vec": example_outbreak_incidence_vec,
        "example_outbreak_perc_risk_threshold_vals": example_outbreak_perc_risk_threshold_vals,
        "many_outbreak_n_sims": many_outbreak_n_sims,
        "many_outbreak_outbreak_size_threshold": many_outbreak_outbreak_size_threshold,
        "many_outbreak_perc_risk_threshold": many_outbreak_perc_risk_threshold,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def get_inputs_inference_test(quasi_real_time=False):
    gen_time_dist_vec = _get_gen_time_dist()

    df_suitability = _get_2017_suitability_data()
    suitability_mean_grid = df_suitability["suitability_smoothed"].to_numpy()

    suitability_model_params = {
        "suitability_std": DEFAULTS.suitability_std,
        "suitability_rho": DEFAULTS.suitability_rho,
        "rep_no_factor_prior_median": DEFAULTS.rep_no_factor_prior_median,
        "rep_no_factor_prior_percentile_2_5": DEFAULTS.rep_no_factor_prior_percentile_2_5,
        "log_rep_no_factor_rho": DEFAULTS.log_rep_no_factor_rho,
    }

    doy_start = 152

    analysis_label = "inference_test" + ("_qrt" if quasi_real_time else "")
    results_dir = pathlib.Path(__file__).parents[1] / "results" / analysis_label
    results_dir.mkdir(parents=True, exist_ok=True)
    results_paths = {
        "outbreak_data": results_dir / "outbreak_data.csv",
        "autoregressive": results_dir / "autoregressive.csv",
        "autoregressive_diagnostics": results_dir / "autoregressive_diagnostics.csv",
        "suitability": results_dir / "suitability.csv",
        "suitability_diagnostics": results_dir / "suitability_diagnostics.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures" / analysis_label
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "rep_no": fig_dir / "rep_no.svg",
        "risk": fig_dir / "risk.svg",
        "declaration": fig_dir / "declaration.svg",
        "suitability": fig_dir / "suitability.svg",
        "scaling_factor": fig_dir / "scaling_factor.svg",
    }

    return {
        "gen_time_dist_vec": gen_time_dist_vec,
        "suitability_mean_grid": suitability_mean_grid,
        "suitability_model_params": suitability_model_params,
        "doy_start": doy_start,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def get_inputs_lazio_outbreak(quasi_real_time=False):
    gen_time_dist_vec = _get_gen_time_dist()

    df_data = _get_lazio_outbreak_data()
    doy_start = df_data["doy"].to_numpy()[0]
    incidence_vec = np.append(
        df_data["cases"].to_numpy(), np.zeros(len(gen_time_dist_vec) + 1, dtype=int)
    )
    doy_vec = (np.arange(doy_start, doy_start + len(incidence_vec)) - 1) % 365 + 1

    df_suitability = _get_2017_suitability_data()
    suitability_mean_vec = (
        df_suitability["suitability_smoothed"]
        .loc[df_suitability["doy"].isin(doy_vec)]
        .to_numpy()
    )

    time_last_case = np.nonzero(incidence_vec)[0][-1]
    doy_last_case = doy_start + time_last_case
    doy_blood_resumed_rome = pd.Timestamp("2017-11-17").dayofyear
    doy_blood_resumed_anzio = pd.Timestamp("2017-12-01").dayofyear
    existing_declarations = {
        "blood_resumed_rome": {
            "doy": doy_blood_resumed_rome,
            "outbreak_day": doy_blood_resumed_rome - doy_start,
            "days_from_last_case": doy_blood_resumed_rome - doy_last_case,
        },
        "blood_resumed_anzio": {
            "doy": doy_blood_resumed_anzio,
            "outbreak_day": doy_blood_resumed_anzio - doy_start,
            "days_from_last_case": doy_blood_resumed_anzio - doy_last_case,
        },
        "45_day_rule": {
            "doy": doy_last_case + 45,
            "outbreak_day": time_last_case + 45,
            "days_from_last_case": 45,
        },
    }

    analysis_label = "lazio_outbreak" + ("_qrt" if quasi_real_time else "")
    results_dir = pathlib.Path(__file__).parents[1] / "results" / analysis_label
    results_dir.mkdir(parents=True, exist_ok=True)
    results_paths = {
        "autoregressive": results_dir / "autoregressive.csv",
        "autoregressive_diagnostics": results_dir / "autoregressive_diagnostics.csv",
        "suitability": results_dir / "suitability.csv",
        "suitability_diagnostics": results_dir / "suitability_diagnostics.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures" / analysis_label
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "gen_time_dist": fig_dir / "gen_time_dist.svg",
        "rep_no": fig_dir / "rep_no.svg",
        "risk": fig_dir / "risk.svg",
        "declaration": fig_dir / "declaration.svg",
        "suitability": fig_dir / "suitability.svg",
        "scaling_factor": fig_dir / "scaling_factor.svg",
    }

    return {
        "gen_time_dist_vec": gen_time_dist_vec,
        "doy_vec": doy_vec,
        "incidence_vec": incidence_vec,
        "suitability_mean_vec": suitability_mean_vec,
        "existing_declarations": existing_declarations,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def _get_lazio_outbreak_data():
    df = pd.read_csv(
        pathlib.Path(__file__).parents[1] / "data/lazio_chik_2017.csv",
        index_col="onset_date",
        parse_dates=True,
    )
    df["doy"] = df.index.day_of_year
    return df


def _get_2017_suitability_data():
    df = pd.read_csv(
        pathlib.Path(__file__).parents[1] / "results/weather_suitability_data/2017.csv",
        index_col="date",
        parse_dates=True,
    )
    df["doy"] = df.index.day_of_year
    return df


def _get_gen_time_dist():
    gen_time_max = 40
    gen_time_dist_cont = scipy.stats.gamma(a=8.53, scale=1.46)
    gen_time_dist_vec = _discretise_cori(
        dist_cont=gen_time_dist_cont, max_val=gen_time_max
    )
    return gen_time_dist_vec


def _discretise_cori(
    *, dist_cont: scipy.stats.rv_continuous, max_val: int, allow_zero: bool = False
):
    """
    Function for discretising a continuous distribution using the method
    described in https://doi.org/10.1093/aje/kwt133 (web appendix 11).
    """

    def _integrand_func(x, y):
        # To get probability mass function at time x, need to integrate this expression
        # with respect to y between y=x-1 and and y=x+1
        return (1 - abs(x - y)) * dist_cont.pdf(y)

    if max_val < 0:
        raise ValueError("max_val must be non-negative")
    if not allow_zero and max_val < 1:
        raise ValueError("max_val must be at least 1 when allow_zero is False")

    # Set up vector of x values and pre-allocate vector of probabilities
    x_vec = np.arange(0, max_val + 1, dtype=int)
    p_vec = np.zeros(len(x_vec))
    # Calculate probability mass function at each x value
    for i in range(len(x_vec)):  # pylint: disable=consider-using-enumerate
        x = x_vec[i]
        integrand = functools.partial(_integrand_func, x)
        p_vec[i] = scipy.integrate.quad(
            integrand,
            x - 1 if x > 0 else 1e-12,
            x + 1,
        )[0]
    if not allow_zero:
        # Assign mass from 0 to 1
        x_vec = x_vec[1:]
        p_vec[1] = p_vec[1] + p_vec[0]
        p_vec = p_vec[1:]
    # Assign residual mass to x_max
    p_vec[-1] = p_vec[-1] + 1 - np.sum(p_vec)
    return p_vec
