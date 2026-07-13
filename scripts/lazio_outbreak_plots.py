import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd import calc_decision_delay
from endoutbreakvbd.utils import (
    get_colors,
    month_start_xticks,
    ordered_legend,
    plot_data_on_twin_ax,
    set_plot_config,
)
from scripts.inputs import get_inputs_lazio_outbreak


def make_plots(quasi_real_time=False):
    set_plot_config()
    inputs = get_inputs_lazio_outbreak(quasi_real_time=quasi_real_time)
    _make_serial_interval_dist_plot(
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        save_path=inputs["fig_paths"]["serial_interval_dist"],
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
    _make_prob_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        model_names=["Suitability-based", "Autoregressive"],
        existing_decisions=inputs["existing_decisions"],
        data_paths=[
            inputs["results_paths"]["suitability"],
            inputs["results_paths"]["autoregressive"],
        ],
        save_path=inputs["fig_paths"]["additional_case_prob"],
    )
    _make_decision_plot(
        incidence_vec=inputs["incidence_vec"],
        model_names=["Suitability-based", "Autoregressive"],
        existing_decisions=inputs["existing_decisions"],
        data_paths=[
            inputs["results_paths"]["suitability"],
            inputs["results_paths"]["autoregressive"],
        ],
        perc_risk_thresholds=inputs["perc_risk_threshold_grid"],
        save_path=inputs["fig_paths"]["decision"],
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


def _make_serial_interval_dist_plot(*, serial_interval_dist_vec, save_path=None):
    t_vec = np.arange(1, len(serial_interval_dist_vec) + 1)
    fig, ax = plt.subplots()
    ax.bar(t_vec, serial_interval_dist_vec, color="tab:blue")
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
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, np.minimum(ax.get_ylim()[1], 10))
    ax.set_ylabel("Time-dependent reproduction number")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_prob_plot(
    *,
    doy_vec,
    incidence_vec,
    model_names,
    existing_decisions,
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
        ax.plot(doy_vec, df["additional_case_prob"], color=color, label=model_name)
    if existing_decisions:
        # ax.axvline(
        #     existing_decisions["blood_resumed_rome"]["doy"],
        #     0,
        #     1,
        #     color=colors[-3],
        #     linestyle="dotted",
        #     label="Blood measures lifted (Rome)",
        # )
        ax.axvline(
            existing_decisions["blood_resumed_anzio"]["doy"],
            0,
            1,
            color=colors[-2],
            linestyle="dotted",
            label="Blood measures\nlifted",
        )
        ax.axvline(
            existing_decisions["45_day_rule"]["doy"],
            0,
            1,
            color=colors[-1],
            linestyle="dotted",
            label="45-day rule",
        )
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Probability of additional cases")
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.89))
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_decision_plot(
    *,
    incidence_vec,
    model_names,
    existing_decisions,
    data_paths,
    perc_risk_thresholds,
    save_path=None,
):
    colors = get_colors()[: (len(model_names) + 2)]
    fig, ax = plt.subplots()
    time_final_case = np.nonzero(incidence_vec)[0][-1]
    prob_days = np.arange(time_final_case + 1, len(incidence_vec))
    for model_name, color, data_path in zip(
        model_names, colors[: len(model_names)], data_paths, strict=True
    ):
        df = pd.read_csv(data_path)
        prob_vals = df["additional_case_prob"].to_numpy()[prob_days]
        decision_delays = calc_decision_delay(
            prob_vec=prob_vals,
            days=prob_days,
            perc_risk_threshold=perc_risk_thresholds,
            time_final_case=time_final_case,
        )
        ax.plot(perc_risk_thresholds, decision_delays, color=color, label=model_name)
    if existing_decisions:
        # ax.axhline(
        #     existing_decisions["blood_resumed_rome"]["days_from_final_case"],
        #     color=colors[-2],
        #     linestyle="dotted",
        #     label="Blood measures lifted (Rome)",
        # )
        ax.axhline(
            existing_decisions["blood_resumed_anzio"]["days_from_final_case"],
            color=colors[-1],
            linestyle="dotted",
            label="Blood measures lifted",
        )
    ax.set_xticks(np.append(perc_risk_thresholds[0], ax.get_xticks()))
    ax.set_xlim(perc_risk_thresholds[0], perc_risk_thresholds[-1])
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
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
    ax.plot(doy_vec, df["suitability_mean"], color="tab:blue", label="Posterior")
    ax.fill_between(
        doy_vec,
        df["suitability_lower"],
        df["suitability_upper"],
        color="tab:blue",
        alpha=0.3,
    )
    ax.plot(
        doy_vec,
        suitability_mean_vec,
        color="black",
        linestyle="dashed",
        label="Seasonal prior",
    )
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Temperature suitability for transmission")
    ordered_legend(ax, {"True": 0, "Seasonal prior": 1}, loc="upper right")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_scaling_factor_plot(*, doy_vec, incidence_vec, data_path, save_path=None):
    df = pd.read_csv(data_path)
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, doy_vec, incidence_vec)
    ax.plot(doy_vec, df["rep_no_factor_mean"], color="tab:blue", label="Posterior")
    ax.fill_between(
        doy_vec,
        df["rep_no_factor_lower"],
        df["rep_no_factor_upper"],
        color="tab:blue",
        alpha=0.3,
    )
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, np.minimum(ax.get_ylim()[1], 10))
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
