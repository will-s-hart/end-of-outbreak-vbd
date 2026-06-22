import functools
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
from numpy.typing import NDArray

from endoutbreakvbd.inference import DEFAULTS
from endoutbreakvbd.utils import (
    discretise_cori,
    rep_no_from_grid,
    rescale_rep_no_grid_in_time,
)


def get_inputs_schematic() -> dict[str, Any]:
    seasonal_amplitude = 2.5
    seasonal_peak_doy = 180
    seasonal_sigma = 65
    doy_grid_full = np.arange(1, 366)
    seasonal_full = seasonal_amplitude * np.exp(
        -(((doy_grid_full - seasonal_peak_doy) / seasonal_sigma) ** 2)
    )
    outbreak_rep_no_vec = 1.05 * seasonal_full

    serial_interval_gamma_shape = 2.5
    serial_interval_gamma_scale = 4.5
    serial_interval_max_days = 36
    days = np.arange(1, serial_interval_max_days + 1)
    serial_interval_pmf = scipy.stats.gamma.pdf(
        days - 0.5, a=serial_interval_gamma_shape, scale=serial_interval_gamma_scale
    )
    serial_interval_dist_vec = serial_interval_pmf / serial_interval_pmf.sum()

    results_dir = pathlib.Path(__file__).parents[1] / "results/schematic"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(__file__).parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    intervention_graphic_path = fig_dir / "schematic" / "intervention_graphic.png"
    safe_graphic_path = fig_dir / "schematic" / "safe_graphic.png"

    return {
        "seasonal_full": seasonal_full,
        "seasonal_amplitude": seasonal_amplitude,
        "outbreak_rep_no_vec": outbreak_rep_no_vec,
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "outbreak_doy_start": 92,
        "outbreak_seed": 1,
        "outbreak_t_stop": 260,
        "outbreak_size_min": 300,
        "outbreak_size_max": 1000,
        "outbreak_final_case_doy_min": 298,
        "outbreak_final_case_doy_max": 315,
        "outbreak_max_attempts": 10000,
        "current_day_offset": 15,
        "inference_seed": 42,
        "rep_no_factor_prior_median": seasonal_amplitude,
        "rep_no_factor_prior_percentile_2_5": 0.9 * seasonal_amplitude,
        "log_rep_no_factor_rho": 0.95,
        "suitability_std": 0.1,
        "suitability_rho": 0.95,
        "results_paths": {"outbreak": results_dir / "outbreak.csv"},
        "intervention_graphic_path": intervention_graphic_path,
        "safe_graphic_path": safe_graphic_path,
        "fig_path": fig_dir / "figure_1.svg",
    }


def get_inputs_weather_suitability_data() -> dict[str, Any]:

    # Temperature-suitability mapping (need to square values for full
    # transmission cycle)
    # df_suitability_grid = pd.read_csv(
    #     pathlib.Path(__file__).parents[1] / "data/mordecai_suitability_grid.csv"
    # )
    df_suitability_grid = pd.read_csv(  # from https://doi.org/10.1098/rsif.2025.0707
        pathlib.Path(__file__).parents[1] / "data/tegar_suitability_grid.csv"
    )
    df_suitability_grid = df_suitability_grid.assign(
        suitability=df_suitability_grid["suitability"] ** 2
    )

    # Assume suitability lags temperature by intrinsic incubation period +
    # 0.5 * (gen time - intrinsic incubation period) = 3 + 0.5 * (12.5 - 3) = 8 days
    # (https://doi.org/10.1016/j.antiviral.2013.06.009)
    suitability_lag_days = 8

    results_dir = pathlib.Path(__file__).parents[1] / "results/weather_suitability_data"
    results_paths = {
        "all": results_dir / "all.csv",
        "2017": results_dir / "2017.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures/weather_suitability_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "temperature": fig_dir / "temperature.svg",
        "suitability_model": fig_dir / "suitability_model.svg",
        "suitability": fig_dir / "suitability.svg",
    }

    return {
        "results_paths": results_paths,
        "fig_paths": fig_paths,
        "df_suitability_grid": df_suitability_grid,
        "suitability_lag_days": suitability_lag_days,
    }


def get_inputs_sim_study() -> dict[str, Any]:
    serial_interval_dist_vec = _get_serial_interval_dist()

    df_suitability = _get_2017_suitability_data()
    suitability_grid = df_suitability["suitability_smoothed_lagged"].to_numpy()

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
    example_outbreak_n_sims = 10000

    many_outbreak_n_sims = 100000
    many_outbreak_outbreak_size_threshold = 2
    # A single threshold for now; tuple form retains support for multiple.
    many_outbreak_perc_risk_threshold_vals = (1,)
    many_outbreak_example_outbreak_idx = 3

    results_dir = pathlib.Path(__file__).parents[1] / "results/sim_study"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_paths = {
        "example_outbreak_prob": results_dir / "example_outbreak_prob.csv",
        "example_outbreak_decision": results_dir / "example_outbreak_decision.csv",
        "many_outbreak": results_dir / "many_outbreak.csv",
        "many_outbreak_example": results_dir / "many_outbreak_example.csv",
        "many_outbreak_decision": results_dir / "many_outbreak_decision.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures/sim_study"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "rep_no": fig_dir / "rep_no.svg",
        "example_outbreak_prob": fig_dir / "example_outbreak_prob.svg",
        "example_outbreak_decision": fig_dir / "example_outbreak_decision.svg",
        "many_outbreak_example": fig_dir / "many_outbreak_example.svg",
        "many_outbreak_decision": fig_dir / "many_outbreak_decision.svg",
    }

    return {
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "rep_no_factor": rep_no_factor,
        "rep_no_grid": rep_no_grid,
        "rep_no_func_doy": rep_no_func_doy,
        "rep_no_from_doy_start": rep_no_from_doy_start,
        "example_outbreak_doy_start_vals": example_outbreak_doy_start_vals,
        "example_outbreak_incidence_vec": example_outbreak_incidence_vec,
        "example_outbreak_perc_risk_threshold_vals": example_outbreak_perc_risk_threshold_vals,
        "example_outbreak_n_sims": example_outbreak_n_sims,
        "many_outbreak_n_sims": many_outbreak_n_sims,
        "many_outbreak_outbreak_size_threshold": many_outbreak_outbreak_size_threshold,
        "many_outbreak_perc_risk_threshold_vals": many_outbreak_perc_risk_threshold_vals,
        "many_outbreak_example_outbreak_idx": many_outbreak_example_outbreak_idx,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def get_inputs_sim_sensitivity() -> dict[str, Any]:
    inputs_sim_study = get_inputs_sim_study()

    serial_interval_dist_vec = inputs_sim_study["serial_interval_dist_vec"]
    rep_no_factor_default = inputs_sim_study["rep_no_factor"]
    rep_no_grid_default = inputs_sim_study["rep_no_grid"]
    suitability_lag_days = get_inputs_weather_suitability_data()["suitability_lag_days"]

    # The reproduction-number profile is centred on the day of peak (smoothed)
    # temperature, shifted by the lag with which suitability follows temperature.
    df_suitability = _get_2017_suitability_data()
    peak_temperature_date = df_suitability.index[
        int(df_suitability["temperature_smoothed"].argmax())
    ]
    season_centre_doy = (
        peak_temperature_date + pd.Timedelta(days=suitability_lag_days)
    ).dayofyear

    # Values explored either side of the default in each analysis.
    rep_no_factor_vals = (1.5, 3)
    decay_speed_default = 1
    decay_speed_vals = (0.7, 1.3)

    # Reuse the main study's settings so the sensitivity runs are comparable.
    many_outbreak_n_sims = inputs_sim_study["many_outbreak_n_sims"]
    many_outbreak_outbreak_size_threshold = inputs_sim_study[
        "many_outbreak_outbreak_size_threshold"
    ]
    many_outbreak_perc_risk_threshold_vals = inputs_sim_study[
        "many_outbreak_perc_risk_threshold_vals"
    ]

    results_dir = pathlib.Path(__file__).parents[1] / "results/sim_sensitivity"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(__file__).parents[1] / "figures/sim_sensitivity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _make_rep_no_func_doy(rep_no_grid):
        return functools.partial(
            rep_no_from_grid, rep_no_grid=rep_no_grid, periodic=True, doy_start=0
        )

    def _make_rep_no_from_doy_start(rep_no_grid):
        return functools.partial(
            rep_no_from_grid, rep_no_grid=rep_no_grid, periodic=True
        )

    def _grid_for_rep_no_factor(rep_no_factor):
        # rep_no_grid_default = rep_no_factor_default * suitability, so rescaling
        # the default grid gives the grid for any target maximum reproduction number.
        return (rep_no_factor / rep_no_factor_default) * rep_no_grid_default

    def _rep_no_factor_label(rep_no_factor):
        return rf"$R_{{\max}} = {rep_no_factor}$"

    def _decay_speed_label(decay_speed):
        return rf"$\sigma = {decay_speed}$"

    # Maximum-reproduction-number analysis
    rep_no_factor_low_grid = _grid_for_rep_no_factor(rep_no_factor_vals[0])
    rep_no_factor_high_grid = _grid_for_rep_no_factor(rep_no_factor_vals[1])
    # Curves ordered by increasing maximum reproduction number.
    rep_no_factor_curves = [
        (
            _rep_no_factor_label(rep_no_factor_vals[0]),
            _make_rep_no_func_doy(rep_no_factor_low_grid),
            "Oranges",
        ),
        (
            _rep_no_factor_label(rep_no_factor_default),
            _make_rep_no_func_doy(rep_no_grid_default),
            "Blues",
        ),
        (
            _rep_no_factor_label(rep_no_factor_vals[1]),
            _make_rep_no_func_doy(rep_no_factor_high_grid),
            "Greens",
        ),
    ]
    rep_no_factor_runs = [
        {
            "rep_no_from_doy_start": _make_rep_no_from_doy_start(
                rep_no_factor_low_grid
            ),
            "cmap_name": "Oranges",
            "results_path": results_dir / "rep_no_factor_low.csv",
            "fig_path": fig_dir / "rep_no_factor_low.svg",
        },
        {
            "rep_no_from_doy_start": _make_rep_no_from_doy_start(
                rep_no_factor_high_grid
            ),
            "cmap_name": "Greens",
            "results_path": results_dir / "rep_no_factor_high.csv",
            "fig_path": fig_dir / "rep_no_factor_high.svg",
        },
    ]

    # Seasonal-decline-speed analysis (at the default maximum reproduction number)
    decay_speed_low_grid = rescale_rep_no_grid_in_time(
        rep_no_grid_default, season_centre_doy, decay_speed_vals[0]
    )
    decay_speed_high_grid = rescale_rep_no_grid_in_time(
        rep_no_grid_default, season_centre_doy, decay_speed_vals[1]
    )
    # Curves ordered by increasing decline speed (sigma).
    decay_speed_curves = [
        (
            _decay_speed_label(decay_speed_vals[0]),
            _make_rep_no_func_doy(decay_speed_low_grid),
            "Oranges",
        ),
        (
            _decay_speed_label(decay_speed_default),
            _make_rep_no_func_doy(rep_no_grid_default),
            "Blues",
        ),
        (
            _decay_speed_label(decay_speed_vals[1]),
            _make_rep_no_func_doy(decay_speed_high_grid),
            "Greens",
        ),
    ]
    decay_speed_runs = [
        {
            "rep_no_from_doy_start": _make_rep_no_from_doy_start(decay_speed_low_grid),
            "cmap_name": "Oranges",
            "results_path": results_dir / "decay_speed_low.csv",
            "fig_path": fig_dir / "decay_speed_low.svg",
        },
        {
            "rep_no_from_doy_start": _make_rep_no_from_doy_start(decay_speed_high_grid),
            "cmap_name": "Greens",
            "results_path": results_dir / "decay_speed_high.csv",
            "fig_path": fig_dir / "decay_speed_high.svg",
        },
    ]

    return {
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "many_outbreak_n_sims": many_outbreak_n_sims,
        "many_outbreak_outbreak_size_threshold": many_outbreak_outbreak_size_threshold,
        "many_outbreak_perc_risk_threshold_vals": many_outbreak_perc_risk_threshold_vals,
        "rep_no_factor": {
            "curves": rep_no_factor_curves,
            "curves_ylim": (0, 1.05 * rep_no_factor_high_grid.max()),
            "curves_fig_path": fig_dir / "rep_no_factor_curves.svg",
            "runs": rep_no_factor_runs,
        },
        "decay_speed": {
            "curves": decay_speed_curves,
            "curves_ylim": (0, 1.05 * rep_no_grid_default.max()),
            "curves_fig_path": fig_dir / "decay_speed_curves.svg",
            "runs": decay_speed_runs,
        },
    }


def get_inputs_inference_test(quasi_real_time: bool = False) -> dict[str, Any]:
    serial_interval_dist_vec = _get_serial_interval_dist()

    df_suitability = _get_2017_suitability_data()
    suitability_mean_grid = df_suitability["suitability_smoothed_lagged"].to_numpy()

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
        "additional_case_prob": fig_dir / "additional_case_prob.svg",
        "decision": fig_dir / "decision.svg",
        "suitability": fig_dir / "suitability.svg",
        "scaling_factor": fig_dir / "scaling_factor.svg",
    }

    return {
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "suitability_mean_grid": suitability_mean_grid,
        "suitability_model_params": suitability_model_params,
        "doy_start": doy_start,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def get_inputs_lazio_outbreak(quasi_real_time: bool = False) -> dict[str, Any]:
    serial_interval_dist_vec = _get_serial_interval_dist()

    df_data = _get_lazio_outbreak_data()
    start_date = df_data.index[0]
    doy_start = df_data["doy"].to_numpy()[0]
    incidence_vec = np.append(
        df_data["cases"].to_numpy(),
        np.zeros(len(serial_interval_dist_vec) + 1, dtype=int),
    )
    doy_vec = (np.arange(doy_start, doy_start + len(incidence_vec)) - 1) % 365 + 1

    df_suitability = _get_2017_suitability_data()
    suitability_mean_vec = (
        df_suitability["suitability_smoothed_lagged"]
        .loc[df_suitability["doy"].isin(doy_vec)]
        .to_numpy()
    )

    time_final_case = np.nonzero(incidence_vec)[0][-1]
    doy_final_case = doy_start + time_final_case
    date_blood_resumed_rome = pd.Timestamp("2017-11-17")
    date_blood_resumed_anzio = pd.Timestamp("2017-12-01")
    doy_blood_resumed_rome = date_blood_resumed_rome.dayofyear
    doy_blood_resumed_anzio = date_blood_resumed_anzio.dayofyear
    existing_decisions = {
        "blood_resumed_rome": {
            "date": date_blood_resumed_rome,
            "doy": doy_blood_resumed_rome,
            "outbreak_day": doy_blood_resumed_rome - doy_start,
            "days_from_final_case": doy_blood_resumed_rome - doy_final_case,
        },
        "blood_resumed_anzio": {
            "date": date_blood_resumed_anzio,
            "doy": doy_blood_resumed_anzio,
            "outbreak_day": doy_blood_resumed_anzio - doy_start,
            "days_from_final_case": doy_blood_resumed_anzio - doy_final_case,
        },
        "45_day_rule": {
            "date": df_data.index[0] + pd.Timedelta(days=time_final_case + 45),
            "doy": doy_final_case + 45,
            "outbreak_day": time_final_case + 45,
            "days_from_final_case": 45,
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
        "serial_interval_dist": fig_dir / "serial_interval_dist.svg",
        "rep_no": fig_dir / "rep_no.svg",
        "additional_case_prob": fig_dir / "additional_case_prob.svg",
        "decision": fig_dir / "decision.svg",
        "suitability": fig_dir / "suitability.svg",
        "scaling_factor": fig_dir / "scaling_factor.svg",
    }

    return {
        "start_date": start_date,
        "time_final_case": time_final_case,
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "doy_vec": doy_vec,
        "incidence_vec": incidence_vec,
        "suitability_mean_vec": suitability_mean_vec,
        "existing_decisions": existing_decisions,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def get_inputs_lazio_frozen() -> dict[str, Any]:
    inputs = get_inputs_lazio_outbreak(quasi_real_time=False)
    # Reuse the suitability fit produced by the lazio_outbreak analysis
    suitability_path = inputs["results_paths"]["suitability"]

    results_dir = pathlib.Path(__file__).parents[1] / "results" / "lazio_frozen"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(__file__).parents[1] / "figures" / "lazio_frozen"
    fig_dir.mkdir(parents=True, exist_ok=True)

    inputs["results_paths"] = {
        "suitability": suitability_path,
        "autoregressive_frozen": results_dir / "autoregressive_frozen.csv",
    }
    inputs["fig_paths"] = {
        "rep_no": fig_dir / "rep_no.svg",
        "additional_case_prob": fig_dir / "additional_case_prob.svg",
        "decision": fig_dir / "decision.svg",
    }
    return inputs


def get_inputs_lazio_epiestim() -> dict[str, Any]:
    inputs = get_inputs_lazio_outbreak(quasi_real_time=False)
    # Reuse the suitability fit produced by the lazio_outbreak analysis
    suitability_path = inputs["results_paths"]["suitability"]

    results_dir = pathlib.Path(__file__).parents[1] / "results" / "lazio_epiestim"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(__file__).parents[1] / "figures" / "lazio_epiestim"
    fig_dir.mkdir(parents=True, exist_ok=True)

    inputs["results_paths"] = {
        "suitability": suitability_path,
        "epiestim": results_dir / "epiestim.csv",
    }
    inputs["fig_paths"] = {
        "rep_no": fig_dir / "rep_no.svg",
        "additional_case_prob": fig_dir / "additional_case_prob.svg",
        "decision": fig_dir / "decision.svg",
    }
    return inputs


def _get_lazio_outbreak_data() -> pd.DataFrame:
    df = pd.read_csv(
        pathlib.Path(__file__).parents[1] / "data/lazio_chik_2017.csv",
        index_col="onset_date",
        parse_dates=True,
    )
    df["doy"] = pd.DatetimeIndex(df.index).day_of_year
    return df


def _get_2017_suitability_data() -> pd.DataFrame:
    df = pd.read_csv(
        pathlib.Path(__file__).parents[1] / "results/weather_suitability_data/2017.csv",
        index_col="date",
        parse_dates=True,
    )
    df["doy"] = pd.DatetimeIndex(df.index).day_of_year
    return df


def _get_serial_interval_dist() -> NDArray[np.float64]:
    serial_interval_max = 40
    serial_interval_dist_cont = scipy.stats.gamma(a=8.53, scale=1.46)
    serial_interval_dist_vec = discretise_cori(
        dist_cont=serial_interval_dist_cont, max_val=serial_interval_max
    )
    return serial_interval_dist_vec
