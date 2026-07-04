"""Plots for the retrospective under-reporting analysis (supplementary figures S5 + S6).

Fig S5 (results): the additional-case probability with the "full outbreak knowledge" dashed
benchmark (S5A), decision delay vs risk threshold (S5B), and the reporting-ceiling sensitivity
sweep (S5C). Fig S6 (diagnostics): the latent true-case trajectory (S6A) and the suitability /
R_t-factor / R_t posteriors (S6B–D).

The cases (S6A) and probability (S5A) panels are shared with the quasi-real-time nowcast
(``lazio_underreporting_qrt_plots``); this module adds the decision and estimate panels.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd import calc_decision_delay
from endoutbreakvbd.utils import (
    get_colors,
    month_start_xticks,
    plot_data_on_twin_ax,
    set_plot_config,
)
from scripts.inputs import get_inputs_lazio_underreporting_retro
from scripts.lazio_underreporting_qrt_plots import (
    _doy,
    _make_cases_plot,
    _make_prob_plot,
)


def make_plots():
    set_plot_config()
    inputs = get_inputs_lazio_underreporting_retro()
    colors = get_colors()
    _make_prob_plot(inputs, colors)
    _make_decision_plot(inputs, colors)
    _make_cases_plot(inputs, colors)
    _make_estimate_plots(inputs)


def _make_decision_plot(inputs, colors):
    time_final_case = inputs["time_final_case"]
    calc_times = inputs["calc_times"]
    thresholds = inputs["perc_risk_threshold_grid"]
    decisions = inputs["existing_decisions"]

    df_suit = pd.read_csv(inputs["results_paths"]["suitability_p60"])
    df_ar = pd.read_csv(inputs["results_paths"]["autoregressive_p60"])

    def _delays(prob_vec, days):
        return calc_decision_delay(
            prob_vec=prob_vec,
            days=days,
            perc_risk_threshold=thresholds,
            time_final_case=time_final_case,
        )

    # Suitability vs autoregressive at 60% reporting (solid), each with the full-reporting
    # "full outbreak knowledge" benchmark (dashed, from the lazio_outbreak fit) — mirroring the
    # probability panel.
    fig, ax = plt.subplots()
    for df, color, label, full_key in [
        (df_suit, colors[0], "Suitability-based", "suitability"),
        (df_ar, colors[1], "Autoregressive", "autoregressive"),
    ]:
        ax.plot(
            thresholds,
            _delays(df["additional_case_prob"].to_numpy(), calc_times),
            color=color,
            label=label,
        )
        df_full = pd.read_csv(inputs["full_reporting_paths"][full_key])
        ax.plot(
            thresholds,
            _delays(
                df_full["additional_case_prob"].to_numpy(),
                df_full["day_of_outbreak"].to_numpy(),
            ),
            color=color,
            linestyle="dashed",
        )
    ax.plot(
        [], [], color="tab:gray", linestyle="dashed", label="Full outbreak knowledge"
    )
    ax.axhline(
        decisions["blood_resumed_anzio"]["days_from_final_case"],
        color=colors[3],
        linestyle="dotted",
        label="Blood measures lifted",
    )
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["decision"])


def _make_estimate_plots(inputs):
    df = pd.read_csv(inputs["results_paths"]["trajectory"], parse_dates=["date"])
    doy = _doy(df["date"])
    reported = df["reported"].to_numpy()
    suitability_prior = inputs["suitability_mean_vec"][: len(doy)]

    # Full-reporting (no under-reporting) posterior for comparison: the retrospective
    # lazio_outbreak suitability fit, mapped onto the same calendar axis.
    df_full = pd.read_csv(inputs["full_reporting_paths"]["suitability"])
    full_doy = _doy(
        inputs["full_reporting_start_date"]
        + pd.to_timedelta(df_full["day_of_outbreak"], unit="D")
    )

    def _plot_estimate(ax, column, *, prior=None):
        plot_data_on_twin_ax(ax, doy, reported)
        ax.plot(doy, df[f"{column}_mean"], color="tab:blue", label="Under-reporting")
        ax.fill_between(
            doy,
            df[f"{column}_lower"],
            df[f"{column}_upper"],
            color="tab:blue",
            alpha=0.3,
        )
        ax.plot(
            full_doy,
            df_full[f"{column}_mean"],
            color="tab:red",
            linestyle="dashed",
            label="Full reporting",
        )
        ax.fill_between(
            full_doy,
            df_full[f"{column}_lower"],
            df_full[f"{column}_upper"],
            color="tab:red",
            alpha=0.15,
        )
        if prior is not None:
            ax.plot(
                doy, prior, color="black", linestyle="dotted", label="Seasonal prior"
            )
        month_start_xticks(ax)
        ax.set_xlabel("Date (2017)")

    fig, ax = plt.subplots()
    _plot_estimate(ax, "suitability", prior=suitability_prior)
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Temperature suitability for transmission")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["suitability"])

    for column, ylabel, fig_key in [
        ("rep_no_factor", "Reproduction number scaling factor", "scaling_factor"),
        ("reproduction_number", "Time-dependent reproduction number", "rep_no"),
    ]:
        fig, ax = plt.subplots()
        _plot_estimate(ax, column)
        ax.set_ylim(0, np.minimum(ax.get_ylim()[1], 10))
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        fig.savefig(inputs["fig_paths"][fig_key])


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    make_plots()
