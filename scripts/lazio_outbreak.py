import argparse
import pathlib

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate

from endoutbreakvbd import eop_analytical
from endoutbreakvbd.chikungunya import get_data, get_parameters, get_suitability_data
from endoutbreakvbd.inference import fit_autoregressive_model, fit_suitability_model
from endoutbreakvbd.utils import month_start_xticks, plot_data_on_twin_ax


def _get_inputs():
    parameters = get_parameters()
    gen_time_dist_vec = parameters["gen_time_dist_vec"]

    df_data = get_data()
    doy_start = df_data["doy"].to_numpy()[0]
    incidence_vec = np.append(
        df_data["cases"].to_numpy(), np.zeros(len(gen_time_dist_vec) + 1, dtype=int)
    )
    doy_vec = (np.arange(doy_start, doy_start + len(incidence_vec)) - 1) % 365 + 1

    df_suitability = get_suitability_data()
    suitability_mean_vec = (
        df_suitability["suitability_smoothed"]
        .loc[df_suitability["doy"].isin(doy_vec)]
        .to_numpy()
    )

    results_dir = pathlib.Path(__file__).parents[1] / "results/lazio_outbreak"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_paths = {
        "autoregressive": results_dir / "autoregressive.csv",
        "suitability": results_dir / "suitability.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures/lazio_outbreak"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "gen_time_dist": fig_dir / "gen_time_dist.svg",
        "rep_no": fig_dir / "rep_no.svg",
        "eop": fig_dir / "eop.svg",
        "declaration": fig_dir / "declaration.svg",
    }

    return {
        "parameters": parameters,
        "gen_time_dist_vec": gen_time_dist_vec,
        "doy_vec": doy_vec,
        "incidence_vec": incidence_vec,
        "suitability_mean_vec": suitability_mean_vec,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def run_analyses():
    inputs = _get_inputs()
    _run_analyses_for_model(
        model="autoregressive",
        incidence_vec=inputs["incidence_vec"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        fit_model_kwargs={},
        save_path=inputs["results_paths"]["autoregressive"],
    )
    _run_analyses_for_model(
        model="suitability",
        incidence_vec=inputs["incidence_vec"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        fit_model_kwargs={
            "suitability_mean_vec": inputs["suitability_mean_vec"],
        },
        save_path=inputs["results_paths"]["suitability"],
    )


def _run_analyses_for_model(
    *, model, incidence_vec, gen_time_dist_vec, fit_model_kwargs, save_path
):
    if model == "autoregressive":
        idata = fit_autoregressive_model(
            incidence_vec=incidence_vec,
            gen_time_dist_vec=gen_time_dist_vec,
            **fit_model_kwargs,
        )
    elif model == "suitability":
        idata = fit_suitability_model(
            incidence_vec=incidence_vec,
            gen_time_dist_vec=gen_time_dist_vec,
            **fit_model_kwargs,
        )
    else:
        raise ValueError(f"Unknown model: {model}")
    rep_no_mat = az.extract(idata, var_names="rep_no_vec").to_numpy()
    rep_no_mean_vec = rep_no_mat.mean(axis=1)
    rep_no_lower_vec = np.percentile(rep_no_mat, 2.5, axis=1)
    rep_no_upper_vec = np.percentile(rep_no_mat, 97.5, axis=1)
    no_days = len(incidence_vec)
    t_vec = np.arange(no_days)
    rep_no_func_estim = scipy.interpolate.interp1d(
        t_vec, rep_no_mat, axis=0, bounds_error=True
    )
    eop_vec = eop_analytical(
        incidence_vec=incidence_vec,
        rep_no_func=rep_no_func_estim,
        gen_time_dist_vec=gen_time_dist_vec,
        t_calc=t_vec,
    )
    df_out = pd.DataFrame(
        {
            "day_of_outbreak": t_vec,
            "reproduction_number_mean": rep_no_mean_vec,
            "reproduction_number_lower": rep_no_lower_vec,
            "reproduction_number_upper": rep_no_upper_vec,
            "end_of_outbreak_probability": eop_vec,
        }
    )
    df_out.to_csv(save_path)


def make_plots():
    inputs = _get_inputs()
    incidence_vec = inputs["incidence_vec"]
    gen_time_dist_vec = inputs["gen_time_dist_vec"]
    _make_gen_time_dist_plot()
    _make_rep_no_plot()
    _make_eop_plot()
    _make_declaration_plot()


def _make_rep_no_plot(
    *, doy_vec, incidence_vec, model_names, data_paths, colors, save_path
):
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, doy_vec, incidence_vec)
    for model_name, data_path, color in zip(model_names, data_paths, colors):
        df = pd.read_csv(data_path)
        ax.plot(
            doy_vec,
            df["reproduction_number_mean"],
            color=color,
            label=model_name,
        )
        ax.fill_between(
            doy_vec,
            df["reproduction_number_lower"],
            df["reproduction_number_upper"],
            color=color,
            alpha=0.3,
        )
    month_start_xticks(ax, interval_months=1)
    ax.set_ylim(0, 8)
    ax.set_ylabel("Time-dependent reproduction number")
    ax.legend()
    fig.savefig(save_path)


def _make_eop_plot(
    *, doy_vec, incidence_vec, model_names, data_paths, colors, save_path
):
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, doy_vec, incidence_vec)
    for model_name, data_path, color in zip(model_names, data_paths, colors):
        df = pd.read_csv(data_path)
        ax.plot(
            doy_vec,
            1 - df["end_of_outbreak_probability"],
            color=color,
            label=model_name,
        )
    month_start_xticks(ax, interval_months=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Risk of additional cases")
    ax.legend()
    fig.savefig(save_path)


def _make_plots(idata_list, model_name_list, color_list):
    _, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 4), constrained_layout=True)
    twin_axs = [plot_data_and_twin(ax) for ax in axs]
    for idata, model_name, color in zip(
        idata_list, model_name_list, color_list, strict=True
    ):
        twin_axs[0].plot(
            doy_vec,
            rep_no_mean_vec,
            color=color,
            label=model_name,
        )
        twin_axs[0].fill_between(
            doy_vec,
            rep_no_lower_vec,
            rep_no_upper_vec,
            color=color,
            alpha=0.3,
        )
        twin_axs[1].plot(
            doy_vec,
            1 - eop_vec,
            color=color,
            label=model_name,
        )
    month_start_xticks(axs[0], interval_months=1)
    twin_axs[0].set_ylim(0, 8)
    twin_axs[0].set_ylabel("Time-dependent reproduction number")
    twin_axs[0].legend()
    month_start_xticks(axs[1], interval_months=1)
    twin_axs[1].set_ylim(0, 1)
    twin_axs[1].set_ylabel("Risk of additional cases")


idata_ar = fit_autoregressive_model(
    incidence_vec=incidence_vec,
    gen_time_dist_vec=gen_time_dist_vec,
)
idata_suitability = fit_suitability_model(
    incidence_vec=incidence_vec,
    gen_time_dist_vec=gen_time_dist_vec,
    suitability_mean_vec=suitability_vec_smoothed,
)
make_plots(
    [idata_ar, idata_suitability],
    ["Autoregressive model", "Suitability model"],
    ["tab:blue", "tab:red"],
)
fig, axs = plt.subplots(1, 2, sharex=True, figsize=(10, 4), constrained_layout=True)
twin_axs = [plot_data_and_twin(ax) for ax in axs]
twin_axs[0].plot(
    doy_vec,
    idata_suitability.posterior["suitability_vec"].mean(dim=["chain", "draw"]),
    color="tab:blue",
)
twin_axs[0].fill_between(
    doy_vec,
    np.percentile(
        idata_suitability.posterior["suitability_vec"].values, 2.5, axis=(0, 1)
    ),
    np.percentile(
        idata_suitability.posterior["suitability_vec"].values, 97.5, axis=(0, 1)
    ),
    color="tab:blue",
    alpha=0.3,
)
twin_axs[0].plot(
    doy_vec,
    suitability_vec_smoothed,
    color="black",
    linestyle="dashed",
)
twin_axs[1].plot(
    doy_vec,
    idata_suitability.posterior["rep_no_factor_vec"].mean(dim=["chain", "draw"]),
    color="tab:blue",
)
twin_axs[1].fill_between(
    doy_vec,
    np.percentile(
        idata_suitability.posterior["rep_no_factor_vec"].values, 2.5, axis=(0, 1)
    ),
    np.percentile(
        idata_suitability.posterior["rep_no_factor_vec"].values, 97.5, axis=(0, 1)
    ),
    color="tab:blue",
    alpha=0.3,
)
month_start_xticks(axs[0], interval_months=1)
twin_axs[0].set_ylim(0, 1)
twin_axs[0].set_ylabel("Suitability")
month_start_xticks(axs[1], interval_months=1)
twin_axs[1].set_ylim(0, 7)
twin_axs[1].set_ylabel("Reproduction number scaling factor")

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
    args = parser.parse_args()
    if not args.plots_only:
        run_analyses()
    if not args.results_only:
        make_plots()
