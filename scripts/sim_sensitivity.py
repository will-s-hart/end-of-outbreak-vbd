import argparse

import numpy as np

from scripts.inputs import get_inputs_sim_sensitivity
from scripts.sim_sensitivity_plots import make_plots
from scripts.sim_study import _run_many_outbreak_analysis


def run_analyses():
    inputs = get_inputs_sim_sensitivity()
    rng = np.random.default_rng(3)
    runs = inputs["rep_no_factor"]["runs"] + inputs["decay_speed"]["runs"]
    for run in runs:
        _run_many_outbreak_analysis(
            n_sims=inputs["many_outbreak_n_sims"],
            outbreak_size_threshold=inputs["many_outbreak_outbreak_size_threshold"],
            perc_risk_threshold_vals=inputs["many_outbreak_perc_risk_threshold_vals"],
            rep_no_from_doy_start=run["rep_no_from_doy_start"],
            serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
            track_premature_decisions=False,
            rng=rng,
            save_path=run["results_path"],
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
