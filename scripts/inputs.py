import functools
import pathlib
from typing import Any

import numpy as np
import pandas as pd
import scipy.stats
from numpy.typing import NDArray

from endoutbreakvbd.rep_no_models import DEFAULTS
from endoutbreakvbd.utils import (
    dates_to_calendar_day_index,
    discretise_cori,
    rep_no_from_grid,
    rescale_rep_no_grid_in_time,
)

# Shared risk-threshold grid for the decision-delay-vs-threshold plots.
RISK_THRESHOLD_PCT_GRID = np.linspace(0.1, 10, 101)

# Zero-report days appended after the final reported case, shared by the full-reporting and
# under-reporting Lazio analyses so their decision-delay curves span the same range and can be
# overlaid. The additional-case probability is only reported over the series, so this padding
# caps the largest decision delay an analysis can resolve: `calc_decision_delay` returns NaN
# (and the threshold curve simply stops) for any threshold the risk has not fallen below by the
# end of the series. 60 days clears the 45-day relaxation rule with margin.
PADDING_DAYS_AFTER_FINAL_CASE = 60

# Right-hand limit for calendar-axis panels (31 December). The padded series runs a few days into
# 2018, but the x-axis is labelled for 2017 and the post-outbreak risk is negligible well before
# the year end. Decision-delay panels are indexed by risk threshold rather than date, so they are
# unaffected and keep the full `PADDING_DAYS_AFTER_FINAL_CASE` range.
CALENDAR_DAY_INDEX_MAX = 365


def get_inputs_schematic() -> dict[str, Any]:
    seasonal_amplitude = 2.5
    doy_seasonal_peak = 180
    seasonal_sigma = 65
    doy_grid = np.arange(1, 366)
    seasonal_profile_grid = seasonal_amplitude * np.exp(
        -(((doy_grid - doy_seasonal_peak) / seasonal_sigma) ** 2)
    )
    outbreak_rep_no_grid = 1.05 * seasonal_profile_grid

    serial_interval_gamma_shape = 2.5
    serial_interval_gamma_scale = 4.5
    serial_interval_max_days = 36
    serial_interval_lag_vec = np.arange(1, serial_interval_max_days + 1)
    serial_interval_pmf_vec = scipy.stats.gamma.pdf(
        serial_interval_lag_vec - 0.5,
        a=serial_interval_gamma_shape,
        scale=serial_interval_gamma_scale,
    )
    serial_interval_dist_vec = serial_interval_pmf_vec / serial_interval_pmf_vec.sum()

    results_dir = pathlib.Path(__file__).parents[1] / "results/schematic"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(__file__).parents[1] / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)
    intervention_graphic_path = fig_dir / "schematic" / "intervention_graphic.png"
    safe_graphic_path = fig_dir / "schematic" / "safe_graphic.png"

    return {
        "seasonal_profile_grid": seasonal_profile_grid,
        "seasonal_amplitude": seasonal_amplitude,
        "outbreak_rep_no_grid": outbreak_rep_no_grid,
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "doy_start": 92,
        "outbreak_seed": 1,
        "t_outbreak_stop": 260,
        "outbreak_min_size": 300,
        "outbreak_max_size": 1000,
        "doy_final_case_min": 298,
        "doy_final_case_max": 315,
        "outbreak_max_attempts": 10000,
        "days_after_final_case": 15,
        "inference_seed": 7,
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
    # suitability_grid_df = pd.read_csv(
    #     pathlib.Path(__file__).parents[1] / "data/mordecai_suitability_grid.csv"
    # )
    suitability_grid_df = pd.read_csv(  # from https://doi.org/10.1098/rsif.2025.0707
        pathlib.Path(__file__).parents[1] / "data/tegar_suitability_grid.csv"
    )
    suitability_grid_df = suitability_grid_df.assign(
        suitability=suitability_grid_df["suitability"] ** 2
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
        "suitability_grid_df": suitability_grid_df,
        "suitability_lag_days": suitability_lag_days,
    }


def get_inputs_sim_study() -> dict[str, Any]:
    serial_interval_dist_vec = _get_serial_interval_dist_vec()

    suitability_df = _get_2017_suitability_df()
    suitability_grid = suitability_df["suitability_smoothed_lagged"].to_numpy()

    rep_no_factor = 2
    rep_no_grid = rep_no_factor * suitability_grid

    rep_no_doy_func = functools.partial(
        rep_no_from_grid, rep_no_grid=rep_no_grid, periodic=True, doy_start=0
    )
    rep_no_from_doy_start = functools.partial(
        rep_no_from_grid, rep_no_grid=rep_no_grid, periodic=True
    )

    example_outbreak_doy_start_vals = (
        np.nonzero(rep_no_grid > 1.2)[0][0] + 1,
        np.nonzero(rep_no_grid > 1.2)[0][-1] + 1,
    )
    example_outbreak_incidence_vec = np.array([1])
    example_outbreak_risk_threshold_pct_vals = (1, 2.5, 5)
    example_outbreak_n_sims = 10000

    many_outbreak_n_sims = 100000
    many_outbreak_min_size = 2
    # A single threshold for now; tuple form retains support for multiple.
    many_outbreak_risk_threshold_pct_vals = (1,)
    many_outbreak_example_idx = 3

    results_dir = pathlib.Path(__file__).parents[1] / "results/sim_study"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_paths = {
        "example_outbreak_prob": results_dir / "example_outbreak_prob.csv",
        "example_outbreak_decision_delay": results_dir
        / "example_outbreak_decision.csv",
        "many_outbreak": results_dir / "many_outbreak.csv",
        "many_outbreak_example": results_dir / "many_outbreak_example.csv",
        "many_outbreak_decision_delay": results_dir / "many_outbreak_decision.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures/sim_study"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "rep_no": fig_dir / "rep_no.svg",
        "example_outbreak_prob": fig_dir / "example_outbreak_prob.svg",
        "example_outbreak_decision_delay": fig_dir / "example_outbreak_decision.svg",
        "many_outbreak_example": fig_dir / "many_outbreak_example.svg",
        "many_outbreak_decision_delay": fig_dir / "many_outbreak_decision.svg",
    }

    return {
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "rep_no_factor": rep_no_factor,
        "rep_no_grid": rep_no_grid,
        "rep_no_doy_func": rep_no_doy_func,
        "rep_no_from_doy_start": rep_no_from_doy_start,
        "example_outbreak_doy_start_vals": example_outbreak_doy_start_vals,
        "example_outbreak_incidence_vec": example_outbreak_incidence_vec,
        "example_outbreak_risk_threshold_pct_vals": example_outbreak_risk_threshold_pct_vals,
        "example_outbreak_n_sims": example_outbreak_n_sims,
        "many_outbreak_n_sims": many_outbreak_n_sims,
        "many_outbreak_min_size": many_outbreak_min_size,
        "many_outbreak_risk_threshold_pct_vals": many_outbreak_risk_threshold_pct_vals,
        "many_outbreak_example_idx": many_outbreak_example_idx,
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
    suitability_df = _get_2017_suitability_df()
    peak_temperature_date = suitability_df.index[
        int(suitability_df["temperature_smoothed"].argmax())
    ]
    doy_season_centre = (
        peak_temperature_date + pd.Timedelta(days=suitability_lag_days)
    ).dayofyear

    # Values explored either side of the default in each analysis.
    rep_no_factor_vals = (1.5, 3)
    decay_speed_default = 1
    decay_speed_vals = (0.7, 1.3)

    # Reuse the main study's settings so the sensitivity runs are comparable.
    many_outbreak_n_sims = inputs_sim_study["many_outbreak_n_sims"]
    many_outbreak_min_size = inputs_sim_study["many_outbreak_min_size"]
    many_outbreak_risk_threshold_pct_vals = inputs_sim_study[
        "many_outbreak_risk_threshold_pct_vals"
    ]

    results_dir = pathlib.Path(__file__).parents[1] / "results/sim_sensitivity"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(__file__).parents[1] / "figures/sim_sensitivity"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def _make_rep_no_doy_func(rep_no_grid):
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
        return rf"$\gamma = {decay_speed}$"

    # Maximum-reproduction-number analysis
    rep_no_factor_low, rep_no_factor_high = rep_no_factor_vals
    rep_no_factor_low_grid = _grid_for_rep_no_factor(rep_no_factor_low)
    rep_no_factor_high_grid = _grid_for_rep_no_factor(rep_no_factor_high)
    # Curves ordered by increasing maximum reproduction number.
    rep_no_factor_curve_specs = [
        (
            _rep_no_factor_label(rep_no_factor_low),
            _make_rep_no_doy_func(rep_no_factor_low_grid),
            "Oranges",
        ),
        (
            _rep_no_factor_label(rep_no_factor_default),
            _make_rep_no_doy_func(rep_no_grid_default),
            "Blues",
        ),
        (
            _rep_no_factor_label(rep_no_factor_high),
            _make_rep_no_doy_func(rep_no_factor_high_grid),
            "Greens",
        ),
    ]
    rep_no_factor_run_specs = [
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
    decay_speed_low, decay_speed_high = decay_speed_vals
    decay_speed_low_grid = rescale_rep_no_grid_in_time(
        rep_no_grid_default, doy_season_centre, decay_speed_low
    )
    decay_speed_high_grid = rescale_rep_no_grid_in_time(
        rep_no_grid_default, doy_season_centre, decay_speed_high
    )
    # Curves ordered by increasing decline speed (gamma).
    decay_speed_curve_specs = [
        (
            _decay_speed_label(decay_speed_low),
            _make_rep_no_doy_func(decay_speed_low_grid),
            "Oranges",
        ),
        (
            _decay_speed_label(decay_speed_default),
            _make_rep_no_doy_func(rep_no_grid_default),
            "Blues",
        ),
        (
            _decay_speed_label(decay_speed_high),
            _make_rep_no_doy_func(decay_speed_high_grid),
            "Greens",
        ),
    ]
    decay_speed_run_specs = [
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
        "many_outbreak_min_size": many_outbreak_min_size,
        "many_outbreak_risk_threshold_pct_vals": many_outbreak_risk_threshold_pct_vals,
        "rep_no_factor": {
            "curve_specs": rep_no_factor_curve_specs,
            "curve_y_limits": (0, 1.05 * rep_no_factor_high_grid.max()),
            "curves_fig_path": fig_dir / "rep_no_factor_curves.svg",
            "run_specs": rep_no_factor_run_specs,
        },
        "decay_speed": {
            "curve_specs": decay_speed_curve_specs,
            "curve_y_limits": (0, 1.05 * rep_no_grid_default.max()),
            "curves_fig_path": fig_dir / "decay_speed_curves.svg",
            "run_specs": decay_speed_run_specs,
        },
    }


def get_inputs_inference_test(quasi_real_time: bool = False) -> dict[str, Any]:
    serial_interval_dist_vec = _get_serial_interval_dist_vec()

    suitability_df = _get_2017_suitability_df()
    suitability_mean_grid = suitability_df["suitability_smoothed_lagged"].to_numpy()

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
        "decision_delay": fig_dir / "decision.svg",
        "suitability": fig_dir / "suitability.svg",
        "rep_no_factor": fig_dir / "scaling_factor.svg",
    }

    return {
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "suitability_mean_grid": suitability_mean_grid,
        "suitability_model_params": suitability_model_params,
        "doy_start": doy_start,
        "risk_threshold_pct_grid": RISK_THRESHOLD_PCT_GRID,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def get_inputs_lazio_outbreak(quasi_real_time: bool = False) -> dict[str, Any]:
    serial_interval_dist_vec = _get_serial_interval_dist_vec()

    outbreak_df = _get_lazio_outbreak_df()
    outbreak_start_date = outbreak_df.index[0]
    doy_start = int(outbreak_df["doy"].to_numpy()[0])
    incidence_vec = _pad_incidence_after_final_case(
        outbreak_df["cases"].to_numpy(), len(serial_interval_dist_vec)
    )
    # Every fit (quasi-real-time or not) reports one projected day past the data (the current-day
    # risk), so the day axis and suitability prior extend one day beyond the incidence.
    n_output_times = len(incidence_vec) + 1
    # Continuous day index (not day-of-year): the padded series runs past 31 December, and a
    # wrapped day-of-year axis would send the plotted curves back to the start of the year.
    calendar_day_index_vec = dates_to_calendar_day_index(
        outbreak_start_date + pd.to_timedelta(np.arange(n_output_times), unit="D")
    )
    suitability_mean_vec = _get_suitability_mean_vec(doy_start, n_output_times)

    t_final_case = int(np.nonzero(incidence_vec)[0][-1])
    final_case_date = outbreak_start_date + pd.Timedelta(days=t_final_case)
    existing_decisions = {
        name: {
            "decision_date": decision_date,
            "calendar_day_index": int(
                dates_to_calendar_day_index(pd.DatetimeIndex([decision_date]))[0]
            ),
            "t": int((decision_date - outbreak_start_date).days),
            "days_after_final_case": int((decision_date - final_case_date).days),
        }
        for name, decision_date in [
            ("blood_resumed_rome", pd.Timestamp("2017-11-17")),
            ("blood_resumed_anzio", pd.Timestamp("2017-12-01")),
            ("45_day_rule", final_case_date + pd.Timedelta(days=45)),
        ]
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
        "decision_delay": fig_dir / "decision.svg",
        "suitability": fig_dir / "suitability.svg",
        "rep_no_factor": fig_dir / "scaling_factor.svg",
    }

    return {
        "outbreak_start_date": outbreak_start_date,
        "t_final_case": t_final_case,
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "calendar_day_index_vec": calendar_day_index_vec,
        "calendar_day_index_max": CALENDAR_DAY_INDEX_MAX,
        "incidence_vec": incidence_vec,
        "suitability_mean_vec": suitability_mean_vec,
        "existing_decisions": existing_decisions,
        "risk_threshold_pct_grid": RISK_THRESHOLD_PCT_GRID,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def get_inputs_lazio_frozen() -> dict[str, Any]:
    inputs = get_inputs_lazio_outbreak(quasi_real_time=False)
    # Reuse the suitability fit produced by the lazio_outbreak analysis
    suitability_results_path = inputs["results_paths"]["suitability"]

    results_dir = pathlib.Path(__file__).parents[1] / "results" / "lazio_frozen"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(__file__).parents[1] / "figures" / "lazio_frozen"
    fig_dir.mkdir(parents=True, exist_ok=True)

    inputs["results_paths"] = {
        "suitability": suitability_results_path,
        "autoregressive_frozen": results_dir / "autoregressive_frozen.csv",
    }
    inputs["fig_paths"] = {
        "rep_no": fig_dir / "rep_no.svg",
        "additional_case_prob": fig_dir / "additional_case_prob.svg",
        "decision_delay": fig_dir / "decision.svg",
    }
    return inputs


def get_inputs_lazio_epiestim() -> dict[str, Any]:
    inputs = get_inputs_lazio_outbreak(quasi_real_time=False)
    # Reuse the suitability fit produced by the lazio_outbreak analysis
    suitability_results_path = inputs["results_paths"]["suitability"]

    results_dir = pathlib.Path(__file__).parents[1] / "results" / "lazio_epiestim"
    results_dir.mkdir(parents=True, exist_ok=True)
    fig_dir = pathlib.Path(__file__).parents[1] / "figures" / "lazio_epiestim"
    fig_dir.mkdir(parents=True, exist_ok=True)

    inputs["results_paths"] = {
        "suitability": suitability_results_path,
        "epiestim": results_dir / "epiestim.csv",
    }
    inputs["fig_paths"] = {
        "rep_no": fig_dir / "rep_no.svg",
        "additional_case_prob": fig_dir / "additional_case_prob.svg",
        "decision_delay": fig_dir / "decision.svg",
    }
    return inputs


def get_inputs_lazio_underreporting_retro() -> dict[str, Any]:
    inputs_lazio = get_inputs_lazio_outbreak(quasi_real_time=False)
    serial_interval_dist_vec = inputs_lazio["serial_interval_dist_vec"]
    serial_interval_max_days = len(serial_interval_dist_vec)

    # Retrospective under-reporting fit over the full reported outbreak with a constant per-day
    # reporting probability and *no* delay/right-truncation. The incidence is padded with zero-
    # report days on the same `PADDING_DAYS_AFTER_FINAL_CASE` rule as the main Lazio analysis, so
    # the additional-case probability is reported over the post-outbreak tail (declining to ~0,
    # spanning the relaxation-decision dates) and the two analyses' decision-delay curves cover
    # the same range. The under-reporting model then projects R_t a further serial interval beyond
    # the padded data, so `suitability_mean_vec` is extended to match.
    outbreak_df = _get_lazio_outbreak_df()
    outbreak_start_date = outbreak_df.index[0]
    doy_start = int(outbreak_df["doy"].to_numpy()[0])
    incidence_vec = _pad_incidence_after_final_case(
        outbreak_df["cases"].to_numpy(), serial_interval_max_days
    )
    n_data_times = len(incidence_vec)
    suitability_mean_vec = _get_suitability_mean_vec(
        doy_start, n_data_times + serial_interval_max_days
    )

    # The additional-case probability is reported for every day of the series plus one projected
    # day past the data (the current-day risk); the decision-delay panels index it by day-of-
    # outbreak (`t_calc_vec`), dated from the outbreak start.
    t_calc_vec = np.arange(n_data_times + 1)
    decision_date_vec = outbreak_start_date + pd.to_timedelta(t_calc_vec, unit="D")

    t_final_case = int(inputs_lazio["t_final_case"])
    existing_decisions = inputs_lazio["existing_decisions"]

    # Constant 60% reporting probability, no delay/right-truncation, shared by both fits.
    reporting_prob = 0.6
    model_names = ["suitability_p60", "autoregressive_p60"]

    results_dir = (
        pathlib.Path(__file__).parents[1] / "results/lazio_underreporting_retro"
    )
    results_dir.mkdir(parents=True, exist_ok=True)
    results_paths = {
        **{name: results_dir / f"{name}.csv" for name in model_names},
        **{
            f"{name}_diagnostics": results_dir / f"{name}_diagnostics.csv"
            for name in model_names
        },
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures/lazio_underreporting_retro"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        # Results: incidence + prob + decision (both with the full-outbreak-knowledge overlay)
        "incidence": fig_dir / "cases.svg",
        "additional_case_prob": fig_dir / "additional_case_prob.svg",
        "decision_delay": fig_dir / "decision.svg",
        # Inference diagnostics (suitability-fit suitability / R_t-factor / R_t, plus the
        # autoregressive-fit R_t as a fourth comparison panel)
        "suitability": fig_dir / "suitability.svg",
        "rep_no_factor": fig_dir / "scaling_factor.svg",
        "rep_no": fig_dir / "rep_no.svg",
        "rep_no_ar": fig_dir / "rep_no_ar.svg",
    }

    return {
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "outbreak_start_date": outbreak_start_date,
        "incidence_vec": incidence_vec,
        "suitability_mean_vec": suitability_mean_vec,
        "t_calc_vec": t_calc_vec,
        "decision_date_vec": decision_date_vec,
        "calendar_day_index_max": CALENDAR_DAY_INDEX_MAX,
        "reporting_prob": reporting_prob,
        "existing_decisions": existing_decisions,
        "t_final_case": t_final_case,
        "risk_threshold_pct_grid": RISK_THRESHOLD_PCT_GRID,
        # Existing full-reporting retrospective fits, overlaid (dashed) on the prob panel as the
        # "full outbreak knowledge" benchmark.
        "full_reporting_paths": {
            "suitability": inputs_lazio["results_paths"]["suitability"],
            "autoregressive": inputs_lazio["results_paths"]["autoregressive"],
        },
        "full_reporting_outbreak_start_date": inputs_lazio["outbreak_start_date"],
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def _get_lazio_outbreak_df() -> pd.DataFrame:
    df = pd.read_csv(
        pathlib.Path(__file__).parents[1] / "data/lazio_chik_2017.csv",
        index_col="onset_date",
        parse_dates=True,
    )
    df["doy"] = pd.DatetimeIndex(df.index).day_of_year
    return df


def _get_2017_suitability_df() -> pd.DataFrame:
    df = pd.read_csv(
        pathlib.Path(__file__).parents[1] / "results/weather_suitability_data/2017.csv",
        index_col="date",
        parse_dates=True,
    )
    df["doy"] = pd.DatetimeIndex(df.index).day_of_year
    return df


def _pad_incidence_after_final_case(
    reported_incidence_vec: NDArray[np.int64], serial_interval_max_days: int
) -> NDArray[np.int64]:
    # Append zero-report days so the series runs to `PADDING_DAYS_AFTER_FINAL_CASE` past the final
    # reported case, never shorter than the one-serial-interval projection used throughout.
    t_final_case = int(np.nonzero(reported_incidence_vec)[0][-1])
    n_padding_days = max(
        serial_interval_max_days + 1,
        t_final_case
        + PADDING_DAYS_AFTER_FINAL_CASE
        - (len(reported_incidence_vec) - 1),
    )
    return np.append(
        reported_incidence_vec, np.zeros(n_padding_days, dtype=int)
    ).astype(int)


def _get_suitability_mean_vec(doy_start: int, n_times: int) -> NDArray[np.float64]:
    # Seasonal suitability prior for `n_times` days starting from day-of-year `doy_start`.
    # Looked up by label so the values stay in `doy_vec` order: a window running past 31 December
    # wraps back onto the start of the 2017 profile, which a boolean `isin` filter would silently
    # reorder into calendar order instead (placing the January values at the front).
    doy_vec = (np.arange(doy_start, doy_start + n_times) - 1) % 365 + 1
    return (
        _get_2017_suitability_df()
        .set_index("doy")["suitability_smoothed_lagged"]
        .loc[doy_vec]
        .to_numpy()
    )


def _get_serial_interval_dist_vec() -> NDArray[np.float64]:
    serial_interval_max_days = 40
    serial_interval_continuous_dist = scipy.stats.gamma(a=8.53, scale=1.46)
    serial_interval_dist_vec = discretise_cori(
        dist_cont=serial_interval_continuous_dist,
        max_val=serial_interval_max_days,
    )
    return serial_interval_dist_vec
