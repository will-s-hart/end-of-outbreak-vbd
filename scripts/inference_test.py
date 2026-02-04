import argparse
import functools
import pathlib

import arviz_base as azb
import arviz_stats as azs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate

from endoutbreakvbd import calc_further_case_risk_analytical, rep_no_from_grid
from endoutbreakvbd.chikungunya import get_parameters, get_suitability_data
from endoutbreakvbd.inference import fit_autoregressive_model, fit_suitability_model
from endoutbreakvbd.model import run_renewal_model
from endoutbreakvbd.utils import (
    lognormal_params_from_median_percentile_2_5,
    month_start_xticks,
    plot_data_on_twin_ax,
)
from scripts.lazio_outbreak import (
    _make_rep_no_plot,
    _make_risk_plot,
    _make_scaling_factor_plot,
    _make_suitability_plot,
    _run_analyses_for_model,
)


def _get_inputs(quasi_real_time=False):
    parameters = get_parameters()
    gen_time_dist_vec = parameters["gen_time_dist_vec"]

    df_suitability = get_suitability_data()
    suitability_mean_grid = df_suitability["suitability_smoothed"].to_numpy()

    suitability_model_params = {
        "suitability_std": 0.05,
        "suitability_rho": 0.975,
        "rep_no_factor_prior_median": 2.0,
        "rep_no_factor_prior_percentile_2_5": 0.5,
        "log_rep_no_factor_rho": 0.975,
    }

    doy_start = 160

    analysis_label = "inference_test" + ("_qrt" if quasi_real_time else "")
    results_dir = pathlib.Path(__file__).parents[1] / "results" / analysis_label
    results_dir.mkdir(parents=True, exist_ok=True)
    results_paths = {
        "outbreak_data": results_dir / "outbreak_data.csv",
        "autoregressive": results_dir / "autoregressive.csv",
        "autoregressive_diagnostics": results_dir / "autoregressive_diagnostics.csv",
        "suitability": results_dir / "suitability.csv",
        "suitability_diagnostics": results_dir / "suitability_diagnostics.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures" / analysis_label
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "rep_no": fig_dir / "rep_no.svg",
        "risk": fig_dir / "risk.svg",
        "suitability": fig_dir / "suitability.svg",
        "scaling_factor": fig_dir / "scaling_factor.svg",
    }

    return {
        "parameters": parameters,
        "gen_time_dist_vec": gen_time_dist_vec,
        "suitability_mean_grid": suitability_mean_grid,
        "suitability_model_params": suitability_model_params,
        "doy_start": doy_start,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def run_analyses(quasi_real_time=False):
    inputs = _get_inputs(quasi_real_time=quasi_real_time)
    rng = np.random.default_rng(2)
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
            "suitability_mean_vec": data_df["suitability"].to_numpy(),
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
    rep_no_factor_prior_median = suitability_model_params["rep_no_factor_prior_median"]
    rep_no_factor_prior_percentile_2_5 = suitability_model_params[
        "rep_no_factor_prior_percentile_2_5"
    ]
    rep_no_factor_prior_params = lognormal_params_from_median_percentile_2_5(
        median=rep_no_factor_prior_median,
        percentile_2_5=rep_no_factor_prior_percentile_2_5,
    )
    log_rep_no_factor_rho = suitability_model_params["log_rep_no_factor_rho"]

    t_max = 300
    suitability_mean_vec = rep_no_from_grid(
        np.arange(t_max),
        rep_no_grid=suitability_mean_grid,
        periodic=True,
        doy_start=doy_start,
    )
    outbreak_found = False
    outbreak_threshold = 100
    while not outbreak_found:
        suitability_vec = _run_ar_sim(
            mean=suitability_mean_vec,
            std=suitability_std,
            rho=suitability_rho,
            t_max=t_max,
            rng=rng,
        )
        suitability_vec = np.clip(suitability_vec, 0, 1)
        log_rep_no_factor_vec = _run_ar_sim(
            mean=rep_no_factor_prior_params["mu"],
            std=rep_no_factor_prior_params["sigma"],
            rho=log_rep_no_factor_rho,
            t_max=t_max,
            rng=rng,
        )
        rep_no_factor_vec = np.exp(log_rep_no_factor_vec)

        rep_no_vec = suitability_vec * rep_no_factor_vec
        rep_no_func = functools.partial(
            rep_no_from_grid, rep_no_grid=rep_no_vec, periodic=False
        )
        incidence_vec = run_renewal_model(
            rep_no_func=rep_no_func, gen_time_dist_vec=gen_time_dist_vec, rng=rng
        )
        if np.sum(incidence_vec) >= outbreak_threshold:
            outbreak_found = True
    t_end = len(incidence_vec)
    risk_vec = calc_further_case_risk_analytical(
        incidence_vec=incidence_vec,
        gen_time_dist_vec=gen_time_dist_vec,
        rep_no_func=rep_no_func,
        t_calc=np.arange(t_end),
    )
    data_df = pd.DataFrame(
        {
            "day_of_year": np.arange(doy_start, doy_start + t_end) % 365,
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


def make_plots(quasi_real_time=False):
    inputs = _get_inputs(quasi_real_time=quasi_real_time)
    df_data = pd.read_csv(inputs["results_paths"]["outbreak_data"], index_col=0)
    doy_vec = df_data["day_of_year"].to_numpy()
    incidence_vec = df_data["cases"].to_numpy()
    suitability_vec = df_data["suitability"].to_numpy()
    suitability_mean_vec = df_data["suitability_mean"].to_numpy()
    rep_no_factor_vec = df_data["rep_no_factor"].to_numpy()
    rep_no_vec = df_data["rep_no"].to_numpy()
    risk_vec = df_data["further_case_risk"].to_numpy()

    for plot_func, plot_kwargs, actual_vec, save_path in [
        (
            _make_suitability_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "suitability_mean_vec": suitability_mean_vec,
                "data_path": inputs["results_paths"]["suitability"],
            },
            suitability_vec,
            inputs["fig_paths"]["suitability"],
        ),
        (
            _make_scaling_factor_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "data_path": inputs["results_paths"]["suitability"],
            },
            rep_no_factor_vec,
            inputs["fig_paths"]["scaling_factor"],
        ),
        (
            _make_rep_no_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "model_names": ["Autoregressive model", "Suitability model"],
                "data_paths": [
                    inputs["results_paths"]["autoregressive"],
                    inputs["results_paths"]["suitability"],
                ],
            },
            rep_no_vec,
            inputs["fig_paths"]["rep_no"],
        ),
        (
            _make_risk_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "model_names": ["Autoregressive model", "Suitability model"],
                "existing_declarations": None,
                "data_paths": [
                    inputs["results_paths"]["autoregressive"],
                    inputs["results_paths"]["suitability"],
                ],
            },
            risk_vec,
            inputs["fig_paths"]["risk"],
        ),
    ]:
        fig, ax = plot_func(**plot_kwargs)
        ax.plot(doy_vec, actual_vec, color="black", label="True")
        ax.legend()
        fig.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    parser.add_argument(
        "-p",
        "--plots-only",
        action="store_true",
        help="Only generate plots (using saved results)",
    )
    parser.add_argument(
        "-q",
        "--quasi-real-time",
        action="store_true",
        help="Perform quasi-real-time analyses",
    )
    args = parser.parse_args()
    if not args.plots_only:
        run_analyses(quasi_real_time=args.quasi_real_time)
    if not args.results_only:
        make_plots(quasi_real_time=args.quasi_real_time)
