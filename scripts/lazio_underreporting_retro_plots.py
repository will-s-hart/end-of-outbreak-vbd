"""Plots for the retrospective under-reporting analysis.

Results: the additional-case probability with the "full outbreak knowledge" dashed
benchmark, decision delay vs risk threshold, and the reporting-ceiling sensitivity
sweep. Diagnostics: the latent true-case trajectory and the suitability /
R_t-factor / R_t posteriors.

The cases and probability panels are shared with the quasi-real-time nowcast
(``lazio_underreporting_qrt_plots``); this module adds the decision and estimate panels.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd import calc_decision_delay
from endoutbreakvbd.utils import (
    dates_to_day_index,
    get_colors,
    month_start_xticks,
    plot_data_on_twin_ax,
    set_plot_config,
)
from scripts.inputs import get_inputs_lazio_underreporting_retro
from scripts.lazio_underreporting_qrt_plots import (
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
            label=f"{label} (full reporting)",
        )
    # Decision markers follow the main Lazio outbreak colours: C3 (blood), C4 (45-day rule). The
    # 45-day line is included here (unlike in the main Lazio outbreak analysis, where it sits well
    # above the panel's y-range).
    ax.axhline(
        decisions["blood_resumed_anzio"]["days_from_final_case"],
        color=colors[3],
        linestyle="dotted",
        label="Blood measures lifted",
    )
    ax.axhline(
        decisions["45_day_rule"]["days_from_final_case"],
        color=colors[4],
        linestyle="dotted",
        label="45-day rule",
    )
    # Label the lower threshold (0.1%) so the axis floor is not misread as 0.
    ax.set_xticks(np.append(thresholds[0], ax.get_xticks()))
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["decision"])


def _make_estimate_plots(inputs):
    df = pd.read_csv(inputs["results_paths"]["trajectory"], parse_dates=["date"])
    doy = dates_to_day_index(df["date"])
    reported = df["reported"].to_numpy()
    suitability_prior = inputs["suitability_mean_vec"][: len(doy)]

    # Full-reporting (no under-reporting) posterior for comparison: the retrospective
    # lazio_outbreak suitability fit, mapped onto the same calendar axis.
    df_full = pd.read_csv(inputs["full_reporting_paths"]["suitability"])
    full_doy = dates_to_day_index(
        inputs["full_reporting_start_date"]
        + pd.to_timedelta(df_full["day_of_outbreak"], unit="D")
    )
    common_horizon = min(doy.max(), full_doy.max())
    plot_mask = doy <= common_horizon

    def _plot_estimate(ax, column, *, prior=None):
        plot_data_on_twin_ax(ax, doy[plot_mask], reported[plot_mask])
        ax.plot(
            doy[plot_mask],
            df.loc[plot_mask, f"{column}_mean"],
            color="tab:blue",
            label="Under-reporting",
        )
        ax.fill_between(
            doy[plot_mask],
            df.loc[plot_mask, f"{column}_lower"],
            df.loc[plot_mask, f"{column}_upper"],
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
                doy[plot_mask],
                prior[plot_mask],
                color="black",
                linestyle="dotted",
                label="Seasonal prior",
            )
        ax.set_xlim(doy.min(), common_horizon)
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
