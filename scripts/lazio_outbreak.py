import argparse
import pathlib

import arviz_base as azb
import arviz_stats as azs
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.interpolate

from endoutbreakvbd import calc_declaration_delay, calc_further_case_risk_analytical
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
        "autoregressive_diagnostics": results_dir / "autoregressive_diagnostics.csv",
        "suitability": results_dir / "suitability.csv",
        "suitability_diagnostics": results_dir / "suitability_diagnostics.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures/lazio_outbreak"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "gen_time_dist": fig_dir / "gen_time_dist.svg",
        "rep_no": fig_dir / "rep_no.svg",
        "risk": fig_dir / "risk.svg",
        "declaration": fig_dir / "declaration.svg",
        "suitability": fig_dir / "suitability.svg",
        "scaling_factor": fig_dir / "scaling_factor.svg",
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
    rng = np.random.default_rng(2)
    _run_analyses_for_model(
        model="autoregressive",
        incidence_vec=inputs["incidence_vec"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        fit_model_kwargs={"rng": rng},
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
    rep_no_mat = azb.extract(idata, var_names="rep_no_vec").to_numpy()
    rep_no_mean_vec = rep_no_mat.mean(axis=1)
    rep_no_lower_vec = np.percentile(rep_no_mat, 2.5, axis=1)
    rep_no_upper_vec = np.percentile(rep_no_mat, 97.5, axis=1)
    no_days = len(incidence_vec)
    t_vec = np.arange(no_days)
    rep_no_func_estim = scipy.interpolate.interp1d(
        t_vec, rep_no_mat, axis=0, bounds_error=True
    )
    risk_vec = calc_further_case_risk_analytical(
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
            "further_case_risk": risk_vec,
        }
    ).set_index("day_of_outbreak")
    if model == "suitability":
        suitability_mat = azb.extract(idata, var_names="suitability_vec").to_numpy()
        suitability_mean_vec = suitability_mat.mean(axis=1)
        suitability_lower_vec = np.percentile(suitability_mat, 2.5, axis=1)
        suitability_upper_vec = np.percentile(suitability_mat, 97.5, axis=1)
        df_out["suitability_mean"] = suitability_mean_vec
        df_out["suitability_lower"] = suitability_lower_vec
        df_out["suitability_upper"] = suitability_upper_vec
        rep_no_factor_mat = azb.extract(idata, var_names="rep_no_factor_vec").to_numpy()
        rep_no_factor_mean_vec = rep_no_factor_mat.mean(axis=1)
        rep_no_factor_lower_vec = np.percentile(rep_no_factor_mat, 2.5, axis=1)
        rep_no_factor_upper_vec = np.percentile(rep_no_factor_mat, 97.5, axis=1)
        df_out["rep_no_factor_mean"] = rep_no_factor_mean_vec
        df_out["rep_no_factor_lower"] = rep_no_factor_lower_vec
        df_out["rep_no_factor_upper"] = rep_no_factor_upper_vec
    df_out.to_csv(save_path)
    # Convergence diagnostics
    rhat = azs.rhat(idata, var_names="rep_no_vec")["rep_no_vec"]
    ess = azs.ess(idata, var_names="rep_no_vec")["rep_no_vec"]
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
                rhat.mean().item(),
                rhat.median().item(),
                rhat.max().item(),
                ess.mean().item(),
                ess.median().item(),
                ess.min().item(),
            ],
        }
    )
    df_diagnostics.to_csv(save_path_diagnostics, index=False)


def make_plots():
    inputs = _get_inputs()
    incidence_vec = inputs["incidence_vec"]
    gen_time_dist_vec = inputs["gen_time_dist_vec"]
    _make_gen_time_dist_plot(
        gen_time_dist_vec=gen_time_dist_vec,
        save_path=inputs["fig_paths"]["gen_time_dist"],
    )
    _make_rep_no_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=incidence_vec,
        model_names=["Autoregressive model", "Suitability model"],
        data_paths=[
            inputs["results_paths"]["autoregressive"],
            inputs["results_paths"]["suitability"],
        ],
        save_path=inputs["fig_paths"]["rep_no"],
    )
    _make_risk_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=incidence_vec,
        model_names=["Autoregressive model", "Suitability model"],
        data_paths=[
            inputs["results_paths"]["autoregressive"],
            inputs["results_paths"]["suitability"],
        ],
        save_path=inputs["fig_paths"]["risk"],
    )
    _make_declaration_plot(
        incidence_vec=incidence_vec,
        model_names=["Autoregressive model", "Suitability model"],
        data_paths=[
            inputs["results_paths"]["autoregressive"],
            inputs["results_paths"]["suitability"],
        ],
        save_path=inputs["fig_paths"]["declaration"],
    )
    _make_suitability_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=incidence_vec,
        suitability_mean_vec=inputs["suitability_mean_vec"],
        data_path=inputs["results_paths"]["suitability"],
        save_path=inputs["fig_paths"]["suitability"],
    )
    _make_scaling_factor_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=incidence_vec,
        data_path=inputs["results_paths"]["suitability"],
        save_path=inputs["fig_paths"]["scaling_factor"],
    )


def _make_gen_time_dist_plot(*, gen_time_dist_vec, save_path):
    t_vec = np.arange(1, len(gen_time_dist_vec) + 1)
    fig, ax = plt.subplots()
    ax.bar(t_vec, gen_time_dist_vec, color="tab:blue")
    ax.set_xlim(0, 35)
    ax.set_xlabel("Generation time (days)")
    ax.set_ylabel("Probability")
    fig.savefig(save_path)


def _make_rep_no_plot(*, doy_vec, incidence_vec, model_names, data_paths, save_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(model_names)]
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, doy_vec, incidence_vec)
    for model_name, data_path, color in zip(
        model_names, data_paths, colors, strict=True
    ):
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


def _make_risk_plot(*, doy_vec, incidence_vec, model_names, data_paths, save_path):
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"][: len(model_names)]
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, doy_vec, incidence_vec)
    for model_name, data_path, color in zip(
        model_names, data_paths, colors, strict=True
    ):
        df = pd.read_csv(data_path)
        ax.plot(doy_vec, df["further_case_risk"], color=color, label=model_name)
    month_start_xticks(ax, interval_months=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Risk of additional cases")
    ax.legend()
    fig.savefig(save_path)


def _make_declaration_plot(*, incidence_vec, model_names, data_paths, save_path):
    fig, ax = plt.subplots()
    perc_risk_thresholds = np.linspace(0.1, 10, 101)
    time_last_case = incidence_vec.nonzero()[0][-1]
    risk_days = np.arange(time_last_case + 1, len(incidence_vec))
    for model_name, data_path in zip(model_names, data_paths, strict=True):
        df = pd.read_csv(data_path)
        risk_vals = df["further_case_risk"].to_numpy()[risk_days]
        declaration_delays = calc_declaration_delay(
            risk_vec=risk_vals,
            perc_risk_threshold=perc_risk_thresholds,
            delay_of_first_risk=1,
        )
        ax.plot(perc_risk_thresholds, declaration_delays, label=model_name)
    ax.set_xticks(np.append(perc_risk_thresholds[0], ax.get_xticks()))
    ax.set_xlim(perc_risk_thresholds[0], perc_risk_thresholds[-1])
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from final case to declaration")
    ax.legend()
    fig.savefig(save_path)


def _make_suitability_plot(
    *, doy_vec, incidence_vec, suitability_mean_vec, data_path, save_path
):
    df = pd.read_csv(data_path)
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, doy_vec, incidence_vec)
    ax.plot(doy_vec, df["suitability_mean"], color="tab:blue")
    ax.fill_between(
        doy_vec,
        df["suitability_lower"],
        df["suitability_upper"],
        color="tab:blue",
        alpha=0.3,
    )
    ax.plot(doy_vec, suitability_mean_vec, color="black", linestyle="dashed")
    month_start_xticks(ax, interval_months=1)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Suitability")
    fig.savefig(save_path)


def _make_scaling_factor_plot(*, doy_vec, incidence_vec, data_path, save_path):
    df = pd.read_csv(data_path)
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, doy_vec, incidence_vec)
    ax.plot(doy_vec, df["rep_no_factor_mean"], color="tab:blue")
    ax.fill_between(
        doy_vec,
        df["rep_no_factor_lower"],
        df["rep_no_factor_upper"],
        color="tab:blue",
        alpha=0.3,
    )
    month_start_xticks(ax, interval_months=1)
    ax.set_ylim(0, 7)
    ax.set_ylabel("Reproduction number scaling factor")
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
    args = parser.parse_args()
    if not args.plots_only:
        run_analyses()
    if not args.results_only:
        make_plots()
