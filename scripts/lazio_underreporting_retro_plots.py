"""Plots for the retrospective under-reporting analysis.

Results: the estimated true-case trajectory, additional-case probability with the
full-reporting benchmark, and decision delay vs risk threshold. Diagnostics: the
suitability, R_t-factor, and R_t posteriors.

The cases and probability panels are shared with the quasi-real-time nowcast
(``lazio_underreporting_qrt_plots``); this module adds the decision and estimate panels.
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd import calc_decision_delay
from endoutbreakvbd.utils import (
    dates_to_calendar_day_index,
    get_colors,
    month_start_xticks,
    plot_incidence_on_twin_ax,
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
    _make_estimate_plots(inputs, colors)


def _make_decision_plot(inputs, colors):
    t_final_case = inputs["t_final_case"]
    t_calc_vec = inputs["t_calc_vec"]
    risk_threshold_pct_vec = inputs["risk_threshold_pct_grid"]
    decisions = inputs["existing_decisions"]

    suitability_df = pd.read_csv(inputs["results_paths"]["suitability_p60"])
    autoregressive_df = pd.read_csv(inputs["results_paths"]["autoregressive_p60"])

    def _calc_decision_delay_vec(prob_vec, t_vec):
        return calc_decision_delay(
            prob_vec=prob_vec,
            t_vec=t_vec,
            risk_threshold_pct=risk_threshold_pct_vec,
            t_final_case=t_final_case,
        )

    # Suitability vs autoregressive at 60% reporting (solid), each with the full-reporting
    # "full outbreak knowledge" benchmark (dashed, from the lazio_outbreak fit) — mirroring the
    # probability panel.
    fig, ax = plt.subplots()
    for results_df, color, label, full_key in [
        (suitability_df, colors[0], "Suitability-based", "suitability"),
        (autoregressive_df, colors[1], "Autoregressive", "autoregressive"),
    ]:
        ax.plot(
            risk_threshold_pct_vec,
            _calc_decision_delay_vec(
                results_df["additional_case_prob"].to_numpy(), t_calc_vec
            ),
            color=color,
            label=label,
        )
        full_reporting_df = pd.read_csv(inputs["full_reporting_paths"][full_key])
        ax.plot(
            risk_threshold_pct_vec,
            _calc_decision_delay_vec(
                full_reporting_df["additional_case_prob"].to_numpy(),
                full_reporting_df["day_of_outbreak"].to_numpy(),
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
    ax.set_xticks(np.append(risk_threshold_pct_vec[0], ax.get_xticks()))
    ax.set_xlim(risk_threshold_pct_vec[0], risk_threshold_pct_vec[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["decision_delay"])


def _make_estimate_plots(inputs, colors):
    suitability_df = pd.read_csv(
        inputs["results_paths"]["suitability_p60"], parse_dates=["date"]
    )
    autoregressive_df = pd.read_csv(
        inputs["results_paths"]["autoregressive_p60"], parse_dates=["date"]
    )
    # Both fits run on the same reported series, so they share the under-reporting calendar axis.
    calendar_day_index_vec = dates_to_calendar_day_index(suitability_df["date"])
    reported_incidence_vec = suitability_df["reported_incidence"].to_numpy()
    suitability_prior_vec = inputs["suitability_mean_vec"][
        : len(calendar_day_index_vec)
    ]

    # Full-reporting (no under-reporting) posteriors for comparison: the retrospective
    # lazio_outbreak fits, mapped onto the same calendar axis.
    full_reporting_outbreak_start_date = inputs["full_reporting_outbreak_start_date"]
    full_reporting_suitability_df = pd.read_csv(
        inputs["full_reporting_paths"]["suitability"]
    )
    full_reporting_autoregressive_df = pd.read_csv(
        inputs["full_reporting_paths"]["autoregressive"]
    )

    def _full_reporting_calendar_day_index_vec(full_reporting_df):
        return dates_to_calendar_day_index(
            full_reporting_outbreak_start_date
            + pd.to_timedelta(full_reporting_df["day_of_outbreak"], unit="D")
        )

    full_reporting_suitability_calendar_day_index_vec = (
        _full_reporting_calendar_day_index_vec(full_reporting_suitability_df)
    )
    full_reporting_autoregressive_calendar_day_index_vec = (
        _full_reporting_calendar_day_index_vec(full_reporting_autoregressive_df)
    )
    common_calendar_day_index_max = min(
        calendar_day_index_vec.max(),
        full_reporting_suitability_calendar_day_index_vec.max(),
        full_reporting_autoregressive_calendar_day_index_vec.max(),
    )
    plot_mask = calendar_day_index_vec <= common_calendar_day_index_max

    # Under-reporting uses the model colour (suitability / autoregressive), matching the
    # probability and decision panels; the full-reporting benchmark is a dashed pink line with a
    # matching band, a distinct hue that stays legible where the two credible intervals overlap
    # (the prob/decision panels have no bands, so they reuse the model colour there).
    def _plot_estimate(
        ax,
        results_df,
        full_reporting_df,
        full_reporting_calendar_day_index_vec,
        column,
        color,
        *,
        prior_vec=None,
    ):
        plot_incidence_on_twin_ax(
            ax, calendar_day_index_vec[plot_mask], reported_incidence_vec[plot_mask]
        )
        # Draw both credible-interval bands first, then every mean line on top, so no line is
        # dimmed by an overlapping band.
        ax.fill_between(
            calendar_day_index_vec[plot_mask],
            results_df.loc[plot_mask, f"{column}_lower"],
            results_df.loc[plot_mask, f"{column}_upper"],
            color=color,
            alpha=0.2,
        )
        ax.fill_between(
            full_reporting_calendar_day_index_vec,
            full_reporting_df[f"{column}_lower"],
            full_reporting_df[f"{column}_upper"],
            color="tab:pink",
            alpha=0.2,
        )
        ax.plot(
            calendar_day_index_vec[plot_mask],
            results_df.loc[plot_mask, f"{column}_mean"],
            color=color,
            label="Under-reporting",
        )
        ax.plot(
            full_reporting_calendar_day_index_vec,
            full_reporting_df[f"{column}_mean"],
            color="tab:pink",
            linestyle="dashed",
            label="Full reporting",
        )
        if prior_vec is not None:
            ax.plot(
                calendar_day_index_vec[plot_mask],
                prior_vec[plot_mask],
                color="black",
                linestyle="dotted",
                label="Seasonal prior",
            )
        ax.set_xlim(calendar_day_index_vec.min(), common_calendar_day_index_max)
        month_start_xticks(ax)
        ax.set_xlabel("Date (2017)")

    fig, ax = plt.subplots()
    _plot_estimate(
        ax,
        suitability_df,
        full_reporting_suitability_df,
        full_reporting_suitability_calendar_day_index_vec,
        "suitability",
        colors[0],
        prior_vec=suitability_prior_vec,
    )
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Temperature suitability for transmission")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["suitability"])

    # Suitability-fit R_t-factor and R_t, then the autoregressive-fit R_t comparison panel.
    for (
        results_df,
        full_reporting_df,
        full_reporting_calendar_day_index_vec,
        column,
        ylabel,
        fig_key,
        color,
    ) in [
        (
            suitability_df,
            full_reporting_suitability_df,
            full_reporting_suitability_calendar_day_index_vec,
            "rep_no_factor",
            "Reproduction number scaling factor",
            "rep_no_factor",
            colors[0],
        ),
        (
            suitability_df,
            full_reporting_suitability_df,
            full_reporting_suitability_calendar_day_index_vec,
            "reproduction_number",
            "Time-dependent reproduction number\n(suitability-based model)",
            "rep_no",
            colors[0],
        ),
        (
            autoregressive_df,
            full_reporting_autoregressive_df,
            full_reporting_autoregressive_calendar_day_index_vec,
            "reproduction_number",
            "Time-dependent reproduction number\n(autoregressive model)",
            "rep_no_ar",
            colors[1],
        ),
    ]:
        fig, ax = plt.subplots()
        _plot_estimate(
            ax,
            results_df,
            full_reporting_df,
            full_reporting_calendar_day_index_vec,
            column,
            color,
        )
        ax.set_ylim(0, np.minimum(ax.get_ylim()[1], 10))
        ax.set_ylabel(ylabel)
        ax.legend(loc="upper right")
        fig.savefig(inputs["fig_paths"][fig_key])


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    make_plots()
