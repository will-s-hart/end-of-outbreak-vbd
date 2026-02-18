import argparse

import numpy as np
import pandas as pd
from arviz_stats import ess, rhat

from endoutbreakvbd.inference import fit_autoregressive_model, fit_suitability_model
from endoutbreakvbd.inputs import get_inputs_lazio_outbreak
from scripts.lazio_outbreak_plots import make_plots


def run_analyses(quasi_real_time=False):
    inputs = get_inputs_lazio_outbreak(quasi_real_time=quasi_real_time)
    rng = np.random.default_rng(2)
    _run_analyses_for_model(
        model="autoregressive",
        incidence_vec=inputs["incidence_vec"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        fit_model_kwargs={"rng": rng, "quasi_real_time": quasi_real_time},
        save_path=inputs["results_paths"]["autoregressive"],
        save_path_diagnostics=inputs["results_paths"]["autoregressive_diagnostics"],
    )
    _run_analyses_for_model(
        model="suitability",
        incidence_vec=inputs["incidence_vec"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        fit_model_kwargs={
            "suitability_mean_vec": inputs["suitability_mean_vec"],
            "rng": rng,
            "quasi_real_time": quasi_real_time,
        },
        save_path=inputs["results_paths"]["suitability"],
        save_path_diagnostics=inputs["results_paths"]["suitability_diagnostics"],
    )


def _run_analyses_for_model(
    *,
    model,
    incidence_vec,
    gen_time_dist_vec,
    fit_model_kwargs,
    save_path,
    save_path_diagnostics,
):
    if model == "autoregressive":
        ds_posterior = fit_autoregressive_model(
            incidence_vec=incidence_vec,
            gen_time_dist_vec=gen_time_dist_vec,
            **fit_model_kwargs,
        )
    elif model == "suitability":
        ds_posterior = fit_suitability_model(
            incidence_vec=incidence_vec,
            gen_time_dist_vec=gen_time_dist_vec,
            **fit_model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    df_out = pd.DataFrame(
        {
            "day_of_outbreak": ds_posterior["time"].values,
            "reproduction_number_mean": ds_posterior["rep_no_mean"].values,
            "reproduction_number_lower": ds_posterior["rep_no_lower"].values,
            "reproduction_number_upper": ds_posterior["rep_no_upper"].values,
            "further_case_risk": ds_posterior["risk"].values,
        }
    ).set_index("day_of_outbreak")
    if model == "suitability":
        df_out = df_out.assign(
            suitability_mean=ds_posterior["suitability_mean"].values,
            suitability_lower=ds_posterior["suitability_lower"].values,
            suitability_upper=ds_posterior["suitability_upper"].values,
            rep_no_factor_mean=ds_posterior["rep_no_factor_mean"].values,
            rep_no_factor_lower=ds_posterior["rep_no_factor_lower"].values,
            rep_no_factor_upper=ds_posterior["rep_no_factor_upper"].values,
        )
    df_out.to_csv(save_path)
    # Convergence diagnostics
    rhat_vals = rhat(ds_posterior, var_names="rep_no")["rep_no"]
    ess_vals = ess(ds_posterior, var_names="rep_no")["rep_no"]
    df_diagnostics = pd.DataFrame(
        {
            "stat": [
                "rhat_mean",
                "rhat_median",
                "rhat_max",
                "ess_mean",
                "ess_median",
                "ess_min",
            ],
            "value": [
                rhat_vals.mean().item(),
                rhat_vals.median().item(),
                rhat_vals.max().item(),
                ess_vals.mean().item(),
                ess_vals.median().item(),
                ess_vals.min().item(),
            ],
        }
    )
    df_diagnostics.to_csv(save_path_diagnostics, index=False)


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
    args = parser.parse_args()
    run_analyses(quasi_real_time=args.quasi_real_time)
    if not args.results_only:
        make_plots(quasi_real_time=args.quasi_real_time)
