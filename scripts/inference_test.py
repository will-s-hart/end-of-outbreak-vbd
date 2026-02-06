import argparse
import functools

import numpy as np
import pandas as pd

from endoutbreakvbd import calc_further_case_risk_analytical, rep_no_from_grid
from endoutbreakvbd.inputs import get_inputs_inference_test
from endoutbreakvbd.model import run_renewal_model

# from endoutbreakvbd.utils import lognormal_params_from_median_percentile_2_5
from scripts.inference_test_plots import make_plots
from scripts.lazio_outbreak import _run_analyses_for_model


def run_analyses(quasi_real_time=False):
    inputs = get_inputs_inference_test(quasi_real_time=quasi_real_time)
    rng = np.random.default_rng(3)
    data_df = _generate_outbreak_data(
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        doy_start=inputs["doy_start"],
        suitability_mean_grid=inputs["suitability_mean_grid"],
        suitability_model_params=inputs["suitability_model_params"],
        rng=rng,
        save_path=inputs["results_paths"]["outbreak_data"],
    )
    _run_analyses_for_model(
        model="autoregressive",
        incidence_vec=data_df["cases"].to_numpy(),
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        fit_model_kwargs={"rng": rng, "quasi_real_time": quasi_real_time},
        save_path=inputs["results_paths"]["autoregressive"],
        save_path_diagnostics=inputs["results_paths"]["autoregressive_diagnostics"],
    )
    _run_analyses_for_model(
        model="suitability",
        incidence_vec=data_df["cases"].to_numpy(),
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        fit_model_kwargs={
            "suitability_mean_vec": data_df["suitability_mean"].to_numpy(),
            "rng": rng,
            "quasi_real_time": quasi_real_time,
        },
        save_path=inputs["results_paths"]["suitability"],
        save_path_diagnostics=inputs["results_paths"]["suitability_diagnostics"],
    )


def _generate_outbreak_data(
    *,
    gen_time_dist_vec,
    doy_start,
    suitability_mean_grid,
    suitability_model_params,
    rng,
    save_path,
):
    suitability_std = suitability_model_params["suitability_std"]
    suitability_rho = suitability_model_params["suitability_rho"]
    # rep_no_factor_prior_median = suitability_model_params["rep_no_factor_prior_median"]
    # rep_no_factor_prior_percentile_2_5 = suitability_model_params[
    #     "rep_no_factor_prior_percentile_2_5"
    # ]
    # rep_no_factor_prior_lognormal_params = lognormal_params_from_median_percentile_2_5(
    #     median=rep_no_factor_prior_median,
    #     percentile_2_5=rep_no_factor_prior_percentile_2_5,
    # )
    # log_rep_no_factor_rho = suitability_model_params["log_rep_no_factor_rho"]

    t_max = 500
    suitability_mean_vec = rep_no_from_grid(
        np.arange(t_max),
        rep_no_grid=suitability_mean_grid,
        periodic=True,
        doy_start=doy_start,
    )
    outbreak_found = False
    outbreak_min_size = 100
    outbreak_max_size = 1000
    attempts = 0
    while not outbreak_found:
        suitability_vec = _run_ar_sim(
            mean=suitability_mean_vec,
            std=suitability_std,
            rho=suitability_rho,
            t_max=t_max,
            rng=rng,
        )
        suitability_vec = np.clip(suitability_vec, 0, 1)
        # log_rep_no_factor_vec = _run_ar_sim(
        #     mean=rep_no_factor_prior_lognormal_params["mu"],
        #     std=rep_no_factor_prior_lognormal_params["sigma"],
        #     rho=log_rep_no_factor_rho,
        #     t_max=t_max,
        #     rng=rng,
        # )
        # rep_no_factor_vec = np.exp(log_rep_no_factor_vec)
        rep_no_factor_vec = np.concatenate((np.full(80, 3), np.full(t_max - 80, 1.5)))

        rep_no_vec = suitability_vec * rep_no_factor_vec
        rep_no_func = functools.partial(
            rep_no_from_grid, rep_no_grid=rep_no_vec, periodic=False
        )
        incidence_vec = run_renewal_model(
            rep_no_func=rep_no_func, gen_time_dist_vec=gen_time_dist_vec, rng=rng
        )
        attempts += 1
        if outbreak_min_size <= np.sum(incidence_vec) <= outbreak_max_size:
            outbreak_found = True
            print(f"Outbreak found after {attempts} attempts")

    t_end = len(incidence_vec)
    risk_vec = calc_further_case_risk_analytical(
        incidence_vec=incidence_vec,
        gen_time_dist_vec=gen_time_dist_vec,
        rep_no_func=rep_no_func,
        t_calc=np.arange(t_end),
    )
    data_df = pd.DataFrame(
        {
            "day_of_year": np.arange(doy_start, doy_start + t_end),
            "cases": incidence_vec[:t_end],
            "rep_no": rep_no_vec[:t_end],
            "suitability": suitability_vec[:t_end],
            "suitability_mean": suitability_mean_vec[:t_end],
            "rep_no_factor": rep_no_factor_vec[:t_end],
            "further_case_risk": risk_vec,
        }
    )
    data_df.to_csv(save_path)
    return data_df


def _run_ar_sim(*, mean, std, rho, t_max, rng):
    deviation_vec = np.zeros(t_max)
    deviation_vec[0] = rng.normal(loc=0, scale=std)
    for t in range(1, t_max):
        deviation_vec[t] = rng.normal(
            loc=rho * deviation_vec[t - 1], scale=std * np.sqrt(1 - rho**2)
        )
    realised_vec = mean + deviation_vec
    return realised_vec


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
