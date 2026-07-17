import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd.inference import fit_autoregressive_model, fit_suitability_model
from scripts.inputs import get_inputs_lazio_outbreak
from scripts.lazio_outbreak_plots import make_plots


def run_analyses(quasi_real_time=False, ar2=False):
    inputs = get_inputs_lazio_outbreak(quasi_real_time=quasi_real_time)
    rng = np.random.default_rng(2)
    _run_analyses_for_model(
        model="autoregressive",
        incidence_vec=inputs["incidence_vec"],
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        fit_model_kwargs={
            "rng": rng,
            "quasi_real_time": quasi_real_time,
            "rho": [0.8, 0.175] if ar2 else None,
        },
        results_path=inputs["results_paths"]["autoregressive"],
        diagnostics_path=inputs["results_paths"]["autoregressive_diagnostics"],
        raise_on_poor_diagnostics=not quasi_real_time,
    )
    _run_analyses_for_model(
        model="suitability",
        incidence_vec=inputs["incidence_vec"],
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        fit_model_kwargs={
            "suitability_mean_vec": inputs["suitability_mean_vec"],
            "rng": rng,
            "quasi_real_time": quasi_real_time,
        },
        results_path=inputs["results_paths"]["suitability"],
        diagnostics_path=inputs["results_paths"]["suitability_diagnostics"],
        raise_on_poor_diagnostics=not quasi_real_time,
    )


def _run_analyses_for_model(
    *,
    model,
    incidence_vec,
    serial_interval_dist_vec,
    fit_model_kwargs,
    results_path,
    diagnostics_path=None,
    compute_diagnostics=True,
    raise_on_poor_diagnostics=False,
):
    if model == "autoregressive":
        posterior_ds = fit_autoregressive_model(
            incidence=incidence_vec,
            serial_interval_dist_vec=serial_interval_dist_vec,
            compute_diagnostics=compute_diagnostics,
            raise_on_poor_diagnostics=raise_on_poor_diagnostics,
            **fit_model_kwargs,
        )
    elif model == "suitability":
        posterior_ds = fit_suitability_model(
            incidence=incidence_vec,
            serial_interval_dist_vec=serial_interval_dist_vec,
            compute_diagnostics=compute_diagnostics,
            raise_on_poor_diagnostics=raise_on_poor_diagnostics,
            **fit_model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    results_df = pd.DataFrame(
        {
            "day_of_outbreak": posterior_ds["time"].values,
            "reproduction_number_mean": posterior_ds["rep_no_mean"].values,
            "reproduction_number_lower": posterior_ds["rep_no_lower"].values,
            "reproduction_number_upper": posterior_ds["rep_no_upper"].values,
            "additional_case_prob": posterior_ds["additional_case_prob"].values,
        }
    ).set_index("day_of_outbreak")
    if model == "suitability":
        results_df = results_df.assign(
            suitability_mean=posterior_ds["suitability_mean"].values,
            suitability_lower=posterior_ds["suitability_lower"].values,
            suitability_upper=posterior_ds["suitability_upper"].values,
            rep_no_factor_mean=posterior_ds["rep_no_factor_mean"].values,
            rep_no_factor_lower=posterior_ds["rep_no_factor_lower"].values,
            rep_no_factor_upper=posterior_ds["rep_no_factor_upper"].values,
        )
    results_df.to_csv(results_path)
    if not compute_diagnostics:
        return
    # Diagnostics were computed, attached, and warned/raised on by the fit function.
    diagnostics = posterior_ds.attrs["diagnostics"]
    pd.Series(diagnostics, name="value").rename_axis("stat").to_csv(diagnostics_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    parser.add_argument(
        "-q",
        "--quasi-real-time",
        action="store_true",
        help="Perform quasi-real-time analyses",
    )
    parser.add_argument(
        "--ar2",
        action="store_true",
        help="Use AR(2) instead of AR(1) prior for autoregressive model",
    )
    args = parser.parse_args()
    run_analyses(quasi_real_time=args.quasi_real_time, ar2=args.ar2)
    if not args.results_only:
        make_plots(quasi_real_time=args.quasi_real_time)
