"""Simulate the schematic outbreak and run inference.

Generates a renewal-process outbreak driven by a seasonal R_t and a gamma
serial-interval distribution, then runs `fit_suitability_model` to infer R_t
and the probability of additional cases on each day. Writes the simulated
incidence together with the posterior R_t summary and daily additional-case probability
to a single CSV consumed by `scripts/schematic_plots.py`.
"""

import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd._types import IntArray, RepNoOutput
from endoutbreakvbd.inference import fit_suitability_model
from endoutbreakvbd.model import simulate_outbreak
from endoutbreakvbd.utils import rep_no_from_grid
from scripts.inputs import get_inputs_schematic
from scripts.schematic_plots import make_plots


def run_analyses():
    inputs = get_inputs_schematic()
    outbreak_simulation = _simulate_outbreak(inputs)
    doy_start = outbreak_simulation["doy_start"]
    incidence_vec = outbreak_simulation["incidence_vec"]
    doy_final_case = doy_start + int(np.nonzero(incidence_vec)[0][-1])
    doy_current = doy_final_case + inputs["days_after_final_case"]
    print(
        f"Outbreak: size={int(incidence_vec.sum())}, "
        f"doy_final_case={doy_final_case}, "
        f"doy_current={doy_current}"
    )

    n_inference_times = incidence_vec.size
    # The fit infers R_t one day past the data (the projected current-day risk), so the
    # suitability prior mean must cover that extra day too.
    doy_inference_vec = doy_start + np.arange(n_inference_times + 1)
    suitability_mean_vec = (
        inputs["seasonal_profile_grid"][doy_inference_vec - 1]
        / inputs["seasonal_amplitude"]
    )

    print("Running suitability-model inference...")
    posterior_ds = fit_suitability_model(
        incidence=incidence_vec,
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        suitability_mean_vec=suitability_mean_vec,
        suitability_std=inputs["suitability_std"],
        suitability_rho=inputs["suitability_rho"],
        rep_no_factor_prior_median=inputs["rep_no_factor_prior_median"],
        rep_no_factor_prior_percentile_2_5=inputs["rep_no_factor_prior_percentile_2_5"],
        log_rep_no_factor_rho=inputs["log_rep_no_factor_rho"],
        rng=np.random.default_rng(inputs["inference_seed"]),
    )
    t_calc = doy_current - doy_start
    print(
        f"Risk on current day (t={t_calc}) = "
        f"{float(posterior_ds['additional_case_prob'].values[t_calc]):.4f}"
    )

    # The fit reports one projected day past the data (the current-day risk), so the posterior
    # columns are one longer than the simulated series. Extend the day axes to match and pad the
    # projected day's case count with 0 (not NaN, which the plot's np.nonzero would read as a case).
    n_output_times = posterior_ds.sizes["time"]
    results_df = pd.DataFrame(
        {
            "day_of_outbreak": np.arange(n_output_times),
            "day_of_year": doy_start + np.arange(n_output_times),
            "incidence": np.append(
                incidence_vec,
                np.zeros(n_output_times - n_inference_times, dtype=int),
            ),
            "reproduction_number_mean": posterior_ds["rep_no_mean"].values,
            "reproduction_number_lower": posterior_ds["rep_no_lower"].values,
            "reproduction_number_upper": posterior_ds["rep_no_upper"].values,
            "additional_case_prob": posterior_ds["additional_case_prob"].values,
        }
    ).set_index("day_of_outbreak")
    results_df.to_csv(inputs["results_paths"]["outbreak"])


def _simulate_outbreak(inputs):
    outbreak_rep_no_grid = inputs["outbreak_rep_no_grid"]
    doy_start = inputs["doy_start"]

    def rep_no_func(t: int | IntArray) -> RepNoOutput:
        return rep_no_from_grid(
            t,
            rep_no_grid=outbreak_rep_no_grid,
            periodic=True,
            doy_start=doy_start,
        )

    def is_final_case_in_window(incidence_vec: IntArray) -> bool:
        doy_final_case = doy_start + int(np.nonzero(incidence_vec)[0][-1])
        return (
            inputs["doy_final_case_min"]
            <= doy_final_case
            <= inputs["doy_final_case_max"]
        )

    incidence_vec = simulate_outbreak(
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        rng=np.random.default_rng(inputs["outbreak_seed"]),
        min_size=inputs["outbreak_min_size"],
        max_size=inputs["outbreak_max_size"],
        accept=is_final_case_in_window,
        incidence_init=1,
        t_stop=inputs["t_outbreak_stop"],
        max_attempts=inputs["outbreak_max_attempts"],
    )
    return {"doy_start": doy_start, "incidence_vec": incidence_vec}


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    args = parser.parse_args()
    run_analyses()
    if not args.results_only:
        make_plots()
