import argparse
import functools

import arviz_base as azb
import numpy as np
import pandas as pd
from arviz_stats import ess, rhat

from endoutbreakvbd import calc_further_case_risk_analytical, rep_no_from_grid
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
        posterior = fit_autoregressive_model(
            incidence_vec=incidence_vec,
            gen_time_dist_vec=gen_time_dist_vec,
            **fit_model_kwargs,
        )
    elif model == "suitability":
        posterior = fit_suitability_model(
            incidence_vec=incidence_vec,
            gen_time_dist_vec=gen_time_dist_vec,
            **fit_model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    rep_no_mat = azb.extract(posterior, var_names="rep_no").to_numpy()
    rep_no_mean_vec = rep_no_mat.mean(axis=1)
    rep_no_lower_vec = np.percentile(rep_no_mat, 2.5, axis=1)
    rep_no_upper_vec = np.percentile(rep_no_mat, 97.5, axis=1)
    no_days = len(incidence_vec)
    t_vec = np.arange(no_days)
    rep_no_post_func = functools.partial(
        rep_no_from_grid, rep_no_grid=rep_no_mat, periodic=False
    )
    risk_vec = calc_further_case_risk_analytical(
        incidence_vec=incidence_vec,
        rep_no_func=rep_no_post_func,
        gen_time_dist_vec=gen_time_dist_vec,
        t_calc=t_vec,
    )
    df_out = pd.DataFrame(
        {
            "day_of_outbreak": t_vec,
            "reproduction_number_mean": rep_no_mean_vec,
            "reproduction_number_lower": rep_no_lower_vec,
            "reproduction_number_upper": rep_no_upper_vec,
            "further_case_risk": risk_vec,
        }
    ).set_index("day_of_outbreak")
    if model == "suitability":
        suitability_mat = azb.extract(posterior, var_names="suitability").to_numpy()
        suitability_mean_vec = suitability_mat.mean(axis=1)
        suitability_lower_vec = np.percentile(suitability_mat, 2.5, axis=1)
        suitability_upper_vec = np.percentile(suitability_mat, 97.5, axis=1)
        df_out["suitability_mean"] = suitability_mean_vec
        df_out["suitability_lower"] = suitability_lower_vec
        df_out["suitability_upper"] = suitability_upper_vec
        rep_no_factor_mat = azb.extract(posterior, var_names="rep_no_factor").to_numpy()
        rep_no_factor_mean_vec = rep_no_factor_mat.mean(axis=1)
        rep_no_factor_lower_vec = np.percentile(rep_no_factor_mat, 2.5, axis=1)
        rep_no_factor_upper_vec = np.percentile(rep_no_factor_mat, 97.5, axis=1)
        df_out["rep_no_factor_mean"] = rep_no_factor_mean_vec
        df_out["rep_no_factor_lower"] = rep_no_factor_lower_vec
        df_out["rep_no_factor_upper"] = rep_no_factor_upper_vec
    df_out.to_csv(save_path)
    # Convergence diagnostics
    rhat_vals = rhat(posterior, var_names="rep_no")["rep_no"]
    ess_vals = ess(posterior, var_names="rep_no")["rep_no"]
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
