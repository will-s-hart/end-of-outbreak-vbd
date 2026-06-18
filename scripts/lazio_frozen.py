import argparse

import numpy as np

from scripts.inputs import get_inputs_lazio_frozen
from scripts.lazio_frozen_plots import make_plots
from scripts.lazio_outbreak import _run_analyses_for_model


def run_analyses():
    inputs = get_inputs_lazio_frozen()
    rng = np.random.default_rng(2)
    _run_analyses_for_model(
        model="autoregressive",
        incidence_vec=inputs["incidence_vec"],
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        fit_model_kwargs={
            "rng": rng,
            "quasi_real_time": False,
            "freeze_from_final_case": True,
        },
        save_path=inputs["results_paths"]["autoregressive_frozen"],
        compute_diagnostics=False,
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
