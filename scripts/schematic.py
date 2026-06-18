"""Simulate the schematic outbreak and run inference (Figure 1).

Generates a renewal-process outbreak driven by a seasonal R_t and a gamma
serial-interval distribution, then runs `fit_suitability_model` to infer R_t
and the probability of additional cases on each day. Writes the simulated
cases together with the posterior R_t summary and the daily risk to a single
CSV consumed by `scripts/schematic_plots.py`.
"""

import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd._types import IntArray, RepNoOutput
from endoutbreakvbd.inference import fit_suitability_model
from endoutbreakvbd.model import run_renewal_model
from endoutbreakvbd.utils import rep_no_from_grid
from scripts.inputs import get_inputs_schematic
from scripts.schematic_plots import make_plots


def run_analyses():
    inputs = get_inputs_schematic()
    sim = _simulate_outbreak(inputs)
    doy_start = sim["doy_start"]
    incidence_vec = sim["incidence_vec"]
    final_case_doy = doy_start + int(np.nonzero(incidence_vec)[0][-1])
    current_day_doy = final_case_doy + inputs["current_day_offset"]
    print(
        f"Outbreak: attempt={sim['attempt']}, "
        f"size={int(incidence_vec.sum())}, "
        f"final_case_doy={final_case_doy}, "
        f"current_day_doy={current_day_doy}"
    )

    n_inf = incidence_vec.size
    doy_for_inf = doy_start + np.arange(n_inf)
    suitability_mean_vec = (
        inputs["seasonal_full"][doy_for_inf - 1] / inputs["seasonal_amplitude"]
    )

    print("Running suitability-model inference...")
    ds_posterior = fit_suitability_model(
        incidence_vec=incidence_vec,
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        suitability_mean_vec=suitability_mean_vec,
        suitability_std=inputs["suitability_std"],
        suitability_rho=inputs["suitability_rho"],
        rep_no_factor_prior_median=inputs["rep_no_factor_prior_median"],
        rep_no_factor_prior_percentile_2_5=inputs["rep_no_factor_prior_percentile_2_5"],
        log_rep_no_factor_rho=inputs["log_rep_no_factor_rho"],
        rng=np.random.default_rng(inputs["inference_seed"]),
    )
    t_calc = current_day_doy - doy_start
    print(
        f"Risk on current day (t={t_calc}) = "
        f"{float(ds_posterior['risk'].values[t_calc]):.4f}"
    )

    df_out = pd.DataFrame(
        {
            "day_of_outbreak": np.arange(n_inf),
            "day_of_year": doy_for_inf,
            "cases": incidence_vec,
            "reproduction_number_mean": ds_posterior["rep_no_mean"].values,
            "reproduction_number_lower": ds_posterior["rep_no_lower"].values,
            "reproduction_number_upper": ds_posterior["rep_no_upper"].values,
            "further_case_risk": ds_posterior["risk"].values,
        }
    ).set_index("day_of_outbreak")
    df_out.to_csv(inputs["results_paths"]["outbreak"])


def _simulate_outbreak(inputs):
    outbreak_rep_no_vec = inputs["outbreak_rep_no_vec"]
    gen_time_dist_vec = inputs["gen_time_dist_vec"]
    doy_start = inputs["outbreak_doy_start"]

    def rep_no_func(t: int | IntArray) -> RepNoOutput:
        return rep_no_from_grid(
            t,
            rep_no_grid=outbreak_rep_no_vec,
            periodic=True,
            doy_start=doy_start,
        )

    rng = np.random.default_rng(inputs["outbreak_seed"])
    for attempt in range(inputs["outbreak_max_attempts"]):
        incidence_vec = run_renewal_model(
            rep_no_func=rep_no_func,
            t_stop=inputs["outbreak_t_stop"],
            gen_time_dist_vec=gen_time_dist_vec,
            rng=rng,
            incidence_init=1,
        )
        size = int(incidence_vec.sum())
        if size < inputs["outbreak_size_min"] or size > inputs["outbreak_size_max"]:
            continue
        final_case_doy = doy_start + int(np.nonzero(incidence_vec)[0][-1])
        if not (
            inputs["outbreak_final_case_doy_min"]
            <= final_case_doy
            <= inputs["outbreak_final_case_doy_max"]
        ):
            continue
        return {
            "doy_start": doy_start,
            "incidence_vec": incidence_vec,
            "attempt": attempt,
        }
    raise RuntimeError(
        f"No outbreak with size in [{inputs['outbreak_size_min']}, "
        f"{inputs['outbreak_size_max']}] and final case in "
        f"[{inputs['outbreak_final_case_doy_min']}, "
        f"{inputs['outbreak_final_case_doy_max']}] "
        f"in {inputs['outbreak_max_attempts']} attempts; rethink inputs."
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    args_in = parser.parse_args()
    run_analyses()
    if not args_in.results_only:
        make_plots()
