import argparse

import numpy as np

from scripts.inputs import get_inputs_sim_sensitivity
from scripts.sim_sensitivity_plots import make_plots
from scripts.sim_study import _run_many_outbreak_analysis


def run_analyses():
    inputs = get_inputs_sim_sensitivity()
    rng = np.random.default_rng(3)
    run_specs = (
        inputs["rep_no_factor"]["run_specs"] + inputs["decay_speed"]["run_specs"]
    )
    for run_spec in run_specs:
        _run_many_outbreak_analysis(
            n_sims=inputs["many_outbreak_n_sims"],
            outbreak_min_size=inputs["many_outbreak_min_size"],
            risk_threshold_pct_vals=inputs["many_outbreak_risk_threshold_pct_vals"],
            rep_no_from_doy_start=run_spec["rep_no_from_doy_start"],
            serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
            track_premature_decisions=False,
            rng=rng,
            results_path=run_spec["results_path"],
        )


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
