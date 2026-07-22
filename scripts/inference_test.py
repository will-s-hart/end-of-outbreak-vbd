import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd import calc_additional_case_prob_analytical
from endoutbreakvbd._types import IntArray, RepNoOutput
from endoutbreakvbd.model import run_renewal_model
from endoutbreakvbd.utils import rep_no_from_grid

# from endoutbreakvbd.utils import lognormal_params_from_median_percentile_2_5
from scripts.inference_test_plots import make_plots
from scripts.inputs import get_inputs_inference_test
from scripts.lazio_outbreak import _run_analyses_for_model


def run_analyses(quasi_real_time=False):
    inputs = get_inputs_inference_test(quasi_real_time=quasi_real_time)
    rng = np.random.default_rng(3)
    outbreak_df = _generate_outbreak_data(
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        doy_start=inputs["doy_start"],
        suitability_mean_grid=inputs["suitability_mean_grid"],
        suitability_model_params=inputs["suitability_model_params"],
        rng=rng,
        results_path=inputs["results_paths"]["outbreak_data"],
    )
    # The generated table includes the projected decision day so its truth/prior columns align
    # with inference output. Incidence on that day is deliberately NaN and is not fitted.
    incidence_vec = outbreak_df["incidence"].dropna().to_numpy(dtype=int)
    _run_analyses_for_model(
        model="autoregressive",
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        fit_model_kwargs={"rng": rng, "quasi_real_time": quasi_real_time},
        results_path=inputs["results_paths"]["autoregressive"],
        diagnostics_path=inputs["results_paths"]["autoregressive_diagnostics"],
    )
    _run_analyses_for_model(
        model="suitability",
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        fit_model_kwargs={
            "suitability_mean_vec": outbreak_df["suitability_mean"].to_numpy(),
            "rng": rng,
            "quasi_real_time": quasi_real_time,
        },
        results_path=inputs["results_paths"]["suitability"],
        diagnostics_path=inputs["results_paths"]["suitability_diagnostics"],
    )


def _generate_outbreak_data(
    *,
    serial_interval_dist_vec,
    doy_start,
    suitability_mean_grid,
    suitability_model_params,
    rng,
    results_path,
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

    t_stop = 500
    suitability_mean_vec = rep_no_from_grid(
        np.arange(t_stop),
        rep_no_grid=suitability_mean_grid,
        periodic=True,
        doy_start=doy_start,
    )
    is_outbreak_found = False
    outbreak_min_size = 500
    outbreak_max_size = 1000
    n_attempts = 0
    while not is_outbreak_found:
        suitability_vec = _simulate_ar_vec(
            mean_vec=suitability_mean_vec,
            std=suitability_std,
            rho=suitability_rho,
            t_stop=t_stop,
            rng=rng,
        )
        suitability_vec = np.clip(suitability_vec, 0, 1)
        rep_no_factor_vec = np.concatenate((np.full(80, 3), np.full(t_stop - 80, 1.1)))

        rep_no_vec = suitability_vec * rep_no_factor_vec

        def rep_no_func(t: int | IntArray) -> RepNoOutput:
            return rep_no_from_grid(t, rep_no_grid=rep_no_vec, periodic=False)

        incidence_vec = run_renewal_model(
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rng=rng,
        )
        n_attempts += 1
        if outbreak_min_size <= np.sum(incidence_vec) <= outbreak_max_size:
            is_outbreak_found = True
            print(f"Outbreak found after {n_attempts} attempts")

    t_data_stop = len(incidence_vec)
    n_output_times = t_data_stop + 1
    prob_vec = calc_additional_case_prob_analytical(
        incidence=incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_func=rep_no_func,
        t_calc=np.arange(n_output_times),
    )
    outbreak_df = pd.DataFrame(
        {
            "day_of_year": np.arange(doy_start, doy_start + n_output_times),
            # Day `t_data_stop` is projected rather than observed; NaN prevents it from being
            # mistaken for a zero-incidence observation when this table is reused for fitting.
            "incidence": np.append(incidence_vec, np.nan),
            "reproduction_number": rep_no_vec[:n_output_times],
            "suitability": suitability_vec[:n_output_times],
            "suitability_mean": suitability_mean_vec[:n_output_times],
            "rep_no_factor": rep_no_factor_vec[:n_output_times],
            "additional_case_prob": prob_vec,
        }
    )
    outbreak_df.to_csv(results_path)
    return outbreak_df


def _simulate_ar_vec(*, mean_vec, std, rho, t_stop, rng):
    deviation_vec = np.zeros(t_stop)
    deviation_vec[0] = rng.normal(loc=0, scale=std)
    for t in range(1, t_stop):
        deviation_vec[t] = rng.normal(
            loc=rho * deviation_vec[t - 1], scale=std * np.sqrt(1 - rho**2)
        )
    realised_vec = mean_vec + deviation_vec
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
