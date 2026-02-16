import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd import calc_declaration_delay
from endoutbreakvbd.inputs import get_inputs_lazio_outbreak
from endoutbreakvbd.utils import (
    get_colors,
    month_start_xticks,
    plot_data_on_twin_ax,
    set_plot_config,
)


def make_plots(quasi_real_time=False):
    set_plot_config()
    inputs = get_inputs_lazio_outbreak(quasi_real_time=quasi_real_time)
    _make_gen_time_dist_plot(
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        save_path=inputs["fig_paths"]["gen_time_dist"],
    )
    _make_rep_no_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        model_names=["Suitability-based", "Autoregressive"],
        data_paths=[
            inputs["results_paths"]["suitability"],
            inputs["results_paths"]["autoregressive"],
        ],
        save_path=inputs["fig_paths"]["rep_no"],
    )
    _make_risk_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        model_names=["Suitability-based", "Autoregressive"],
        existing_declarations=inputs["existing_declarations"],
        data_paths=[
            inputs["results_paths"]["suitability"],
            inputs["results_paths"]["autoregressive"],
        ],
        save_path=inputs["fig_paths"]["risk"],
    )
    _make_declaration_plot(
        incidence_vec=inputs["incidence_vec"],
        model_names=["Suitability-based", "Autoregressive"],
        existing_declarations=inputs["existing_declarations"],
        data_paths=[
            inputs["results_paths"]["suitability"],
            inputs["results_paths"]["autoregressive"],
        ],
        save_path=inputs["fig_paths"]["declaration"],
    )
    _make_suitability_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        suitability_mean_vec=inputs["suitability_mean_vec"],
        data_path=inputs["results_paths"]["suitability"],
        save_path=inputs["fig_paths"]["suitability"],
    )
    _make_scaling_factor_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        data_path=inputs["results_paths"]["suitability"],
        save_path=inputs["fig_paths"]["scaling_factor"],
    )


def _make_gen_time_dist_plot(*, gen_time_dist_vec, save_path=None):
    t_vec = np.arange(1, len(gen_time_dist_vec) + 1)
    fig, ax = plt.subplots()
    ax.bar(t_vec, gen_time_dist_vec, color="tab:blue")
    ax.set_xlim(0, 35)
    ax.set_xlabel("Serial interval (days)")
    ax.set_ylabel("Probability")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_rep_no_plot(
    *, doy_vec, incidence_vec, model_names, data_paths, save_path=None
):
    colors = get_colors()[: len(model_names)]
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
    month_start_xticks(ax)
    ax.set_ylim(0, 8)
    ax.set_ylabel("Time-dependent reproduction number")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_risk_plot(
    *,
    doy_vec,
    incidence_vec,
    model_names,
    existing_declarations,
    data_paths,
    save_path=None,
):
    colors = get_colors()[: (len(model_names) + 3)]
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, doy_vec, incidence_vec)
    for model_name, data_path, color in zip(
        model_names, data_paths, colors[: len(model_names)], strict=True
    ):
        df = pd.read_csv(data_path)
        ax.plot(doy_vec, df["further_case_risk"], color=color, label=model_name)
    if existing_declarations:
        ax.axvline(
            existing_declarations["blood_resumed_rome"]["doy"],
            0,
            1,
            color=colors[-3],
            linestyle="dashed",
            label="Blood measures lifted (Rome)",
        )
        ax.axvline(
            existing_declarations["blood_resumed_anzio"]["doy"],
            0,
            1,
            color=colors[-2],
            linestyle="dashed",
            label="Blood measures lifted (Anzio)",
        )
        ax.axvline(
            existing_declarations["45_day_rule"]["doy"],
            0,
            1,
            color=colors[-1],
            linestyle="dashed",
            label="45-day rule",
        )
    month_start_xticks(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Risk of additional cases")
    ax.legend(loc="upper left")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_declaration_plot(
    *, incidence_vec, model_names, existing_declarations, data_paths, save_path=None
):
    colors = get_colors()[: (len(model_names) + 2)]
    fig, ax = plt.subplots()
    perc_risk_thresholds = np.linspace(0.1, 10, 101)
    time_last_case = np.nonzero(incidence_vec)[0][-1]
    risk_days = np.arange(time_last_case + 1, len(incidence_vec))
    for model_name, color, data_path in zip(
        model_names, colors[: len(model_names)], data_paths, strict=True
    ):
        df = pd.read_csv(data_path)
        risk_vals = df["further_case_risk"].to_numpy()[risk_days]
        declaration_delays = calc_declaration_delay(
            risk_vec=risk_vals,
            perc_risk_threshold=perc_risk_thresholds,
            delay_of_first_risk=1,
        )
        ax.plot(perc_risk_thresholds, declaration_delays, color=color, label=model_name)
    if existing_declarations:
        ax.axhline(
            existing_declarations["blood_resumed_rome"]["days_from_last_case"],
            color=colors[-2],
            linestyle="dashed",
            label="Blood measures lifted (Rome)",
        )
        ax.axhline(
            existing_declarations["blood_resumed_anzio"]["days_from_last_case"],
            color=colors[-1],
            linestyle="dashed",
            label="Blood measures lifted (Anzio)",
        )
    ax.set_xticks(np.append(perc_risk_thresholds[0], ax.get_xticks()))
    ax.set_xlim(perc_risk_thresholds[0], perc_risk_thresholds[-1])
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from final case to declaration")
    ax.legend(loc="lower left")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_suitability_plot(
    *, doy_vec, incidence_vec, suitability_mean_vec, data_path, save_path=None
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
    month_start_xticks(ax)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Temperature suitability for transmission")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_scaling_factor_plot(*, doy_vec, incidence_vec, data_path, save_path=None):
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
    month_start_xticks(ax)
    ax.set_ylim(0, 7)
    ax.set_ylabel("Reproduction number scaling factor")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--quasi-real-time",
        action="store_true",
        help="Perform quasi-real-time analyses",
    )
    args = parser.parse_args()
    make_plots(quasi_real_time=args.quasi_real_time)
