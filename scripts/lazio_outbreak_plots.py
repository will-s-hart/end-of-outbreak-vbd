import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd import calc_decision_delay
from endoutbreakvbd.utils import (
    get_colors,
    month_start_xticks,
    ordered_legend,
    plot_incidence_on_twin_ax,
    set_plot_config,
)
from scripts.inputs import get_inputs_lazio_outbreak


def make_plots(quasi_real_time=False):
    set_plot_config()
    inputs = get_inputs_lazio_outbreak(quasi_real_time=quasi_real_time)
    _make_serial_interval_dist_plot(
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        fig_path=inputs["fig_paths"]["serial_interval_dist"],
    )
    _make_rep_no_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        model_names=["Suitability-based", "Autoregressive"],
        results_paths=[
            inputs["results_paths"]["suitability"],
            inputs["results_paths"]["autoregressive"],
        ],
        fig_path=inputs["fig_paths"]["rep_no"],
    )
    _make_additional_case_prob_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        model_names=["Suitability-based", "Autoregressive"],
        existing_decisions=inputs["existing_decisions"],
        results_paths=[
            inputs["results_paths"]["suitability"],
            inputs["results_paths"]["autoregressive"],
        ],
        fig_path=inputs["fig_paths"]["additional_case_prob"],
    )
    _make_decision_delay_plot(
        incidence_vec=inputs["incidence_vec"],
        model_names=["Suitability-based", "Autoregressive"],
        existing_decisions=inputs["existing_decisions"],
        results_paths=[
            inputs["results_paths"]["suitability"],
            inputs["results_paths"]["autoregressive"],
        ],
        risk_threshold_pct_vec=inputs["risk_threshold_pct_grid"],
        fig_path=inputs["fig_paths"]["decision_delay"],
    )
    _make_suitability_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        suitability_mean_vec=inputs["suitability_mean_vec"],
        results_path=inputs["results_paths"]["suitability"],
        fig_path=inputs["fig_paths"]["suitability"],
    )
    _make_rep_no_factor_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        results_path=inputs["results_paths"]["suitability"],
        fig_path=inputs["fig_paths"]["rep_no_factor"],
    )


def _make_serial_interval_dist_plot(*, serial_interval_dist_vec, fig_path=None):
    serial_interval_lag_vec = np.arange(1, len(serial_interval_dist_vec) + 1)
    fig, ax = plt.subplots()
    ax.bar(serial_interval_lag_vec, serial_interval_dist_vec, color="tab:blue")
    ax.set_xlim(0, 35)
    ax.set_xlabel("Serial interval (days)")
    ax.set_ylabel("Probability")
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


def _make_rep_no_plot(
    *, doy_vec, incidence_vec, model_names, results_paths, fig_path=None
):
    colors = get_colors()[: len(model_names)]
    fig, ax = plt.subplots()
    plot_incidence_on_twin_ax(ax, doy_vec, incidence_vec)
    for model_name, results_path, color in zip(
        model_names, results_paths, colors, strict=True
    ):
        results_df = pd.read_csv(results_path)
        ax.plot(
            doy_vec,
            results_df["reproduction_number_mean"],
            color=color,
            label=model_name,
        )
        ax.fill_between(
            doy_vec,
            results_df["reproduction_number_lower"],
            results_df["reproduction_number_upper"],
            color=color,
            alpha=0.3,
        )
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, np.minimum(ax.get_ylim()[1], 10))
    ax.set_ylabel("Time-dependent reproduction number")
    ax.legend()
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


def _make_additional_case_prob_plot(
    *,
    doy_vec,
    incidence_vec,
    model_names,
    existing_decisions,
    results_paths,
    fig_path=None,
):
    colors = get_colors()[: (len(model_names) + 3)]
    fig, ax = plt.subplots()
    plot_incidence_on_twin_ax(ax, doy_vec, incidence_vec)
    for model_name, results_path, color in zip(
        model_names, results_paths, colors[: len(model_names)], strict=True
    ):
        results_df = pd.read_csv(results_path)
        ax.plot(
            doy_vec,
            results_df["additional_case_prob"],
            color=color,
            label=model_name,
        )
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
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


def _make_decision_delay_plot(
    *,
    incidence_vec,
    model_names,
    existing_decisions,
    results_paths,
    risk_threshold_pct_vec,
    fig_path=None,
):
    colors = get_colors()[: (len(model_names) + 2)]
    fig, ax = plt.subplots()
    t_final_case = np.nonzero(incidence_vec)[0][-1]
    t_calc_vec = np.arange(t_final_case + 1, len(incidence_vec))
    for model_name, color, results_path in zip(
        model_names, colors[: len(model_names)], results_paths, strict=True
    ):
        results_df = pd.read_csv(results_path)
        prob_vec = results_df["additional_case_prob"].to_numpy()[t_calc_vec]
        decision_delay_vec = calc_decision_delay(
            prob_vec=prob_vec,
            t_vec=t_calc_vec,
            risk_threshold_pct=risk_threshold_pct_vec,
            t_final_case=t_final_case,
        )
        ax.plot(
            risk_threshold_pct_vec,
            decision_delay_vec,
            color=color,
            label=model_name,
        )
    if existing_decisions:
        # ax.axhline(
        #     existing_decisions["blood_resumed_rome"]["days_after_final_case"],
        #     color=colors[-2],
        #     linestyle="dotted",
        #     label="Blood measures lifted (Rome)",
        # )
        ax.axhline(
            existing_decisions["blood_resumed_anzio"]["days_after_final_case"],
            color=colors[-1],
            linestyle="dotted",
            label="Blood measures lifted",
        )
    ax.set_xticks(np.append(risk_threshold_pct_vec[0], ax.get_xticks()))
    ax.set_xlim(risk_threshold_pct_vec[0], risk_threshold_pct_vec[-1])
    ax.set_ylim(0, ax.get_ylim()[1])
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
    ax.legend(loc="lower left")
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


def _make_suitability_plot(
    *, doy_vec, incidence_vec, suitability_mean_vec, results_path, fig_path=None
):
    results_df = pd.read_csv(results_path)
    fig, ax = plt.subplots()
    plot_incidence_on_twin_ax(ax, doy_vec, incidence_vec)
    ax.plot(
        doy_vec,
        results_df["suitability_mean"],
        color="tab:blue",
        label="Posterior",
    )
    ax.fill_between(
        doy_vec,
        results_df["suitability_lower"],
        results_df["suitability_upper"],
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
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


def _make_rep_no_factor_plot(*, doy_vec, incidence_vec, results_path, fig_path=None):
    results_df = pd.read_csv(results_path)
    fig, ax = plt.subplots()
    plot_incidence_on_twin_ax(ax, doy_vec, incidence_vec)
    ax.plot(
        doy_vec,
        results_df["rep_no_factor_mean"],
        color="tab:blue",
        label="Posterior",
    )
    ax.fill_between(
        doy_vec,
        results_df["rep_no_factor_lower"],
        results_df["rep_no_factor_upper"],
        color="tab:blue",
        alpha=0.3,
    )
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, np.minimum(ax.get_ylim()[1], 10))
    ax.set_ylabel("Reproduction-number factor")
    if fig_path is not None:
        fig.savefig(fig_path)
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
