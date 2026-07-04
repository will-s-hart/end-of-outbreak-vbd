"""Plots for the quasi-real-time under-reporting nowcast (figure 5).

Three panels: the fitted onset-to-report delay distribution (5A), the latest-snapshot true-case
trajectory (5B), and the real-time additional-case probability (5C) with the retrospective
"full outbreak knowledge" benchmark (the full-reporting ``lazio_outbreak`` fits) overlaid as
dashed lines. ``_make_cases_plot`` and ``_make_prob_plot`` are shared with the retrospective
under-reporting analysis (``lazio_underreporting_retro_plots``).

Builds its own calendar-date axis from the reporting-matrix snapshot dates rather than routing
through the ``doy_vec``-zipped ``lazio_outbreak_plots`` helpers (whose strict zip assumes the
retrospective grid length).
"""

import argparse

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd.utils import (
    get_colors,
    month_start_xticks,
    plot_data_on_twin_ax,
    set_plot_config,
)
from scripts.inputs import get_inputs_lazio_underreporting_qrt


def _doy(dates) -> np.ndarray:
    return pd.DatetimeIndex(dates).dayofyear.to_numpy()


def make_plots(start_date="2017-10-01", end_date="2017-12-31", stride=1):
    set_plot_config()
    inputs = get_inputs_lazio_underreporting_qrt(
        start_date=start_date, end_date=end_date, stride=stride
    )
    colors = get_colors()
    _make_delay_plot(inputs)
    _make_cases_plot(inputs, colors)
    _make_prob_plot(inputs, colors, ylabel="Real-time probability of additional cases")


def _make_delay_plot(inputs):
    df = pd.read_csv(inputs["results_paths"]["delay"])
    fig, ax = plt.subplots()
    ax.bar(
        df["delay"], df["pmf_empirical"], color="tab:gray", alpha=0.6, label="Empirical"
    )
    ax.plot(df["delay"], df["pmf_fitted"], color="tab:blue", label="Fitted gamma")
    ax.set_xlim(0, None)
    ax.set_xlabel("Onset-to-report delay (days)")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["delay"])
    return fig, ax


def _make_cases_plot(inputs, colors):
    df = pd.read_csv(inputs["results_paths"]["trajectory"], parse_dates=["date"])
    doy = _doy(df["date"])
    fig, ax = plt.subplots()
    ax.bar(doy, df["reported"], color="tab:gray", alpha=0.5, label="Reported")
    ax.plot(
        doy, df["cases_mean"], color=colors[0], label="Estimated true (suitability)"
    )
    ax.fill_between(
        doy, df["cases_lower"], df["cases_upper"], color=colors[0], alpha=0.3
    )
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, None)
    ax.set_ylabel("Cases per day")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["cases"])
    return fig, ax


def _make_prob_plot(inputs, colors, *, ylabel="Probability of additional cases"):
    df_suit = pd.read_csv(
        inputs["results_paths"]["suitability_p60"], parse_dates=["date"]
    )
    df_ar = pd.read_csv(
        inputs["results_paths"]["autoregressive_p60"], parse_dates=["date"]
    )
    df_traj = pd.read_csv(inputs["results_paths"]["trajectory"], parse_dates=["date"])
    decisions = inputs["existing_decisions"]
    full_start = inputs["full_reporting_start_date"]

    fig, ax = plt.subplots()
    # Observed reported cases (by onset) as a twin-axis underlay, matching the main Lazio panels.
    plot_data_on_twin_ax(ax, _doy(df_traj["date"]), df_traj["reported"].to_numpy())
    doys = []
    for df, color, label, full_key in [
        (df_suit, colors[0], "Suitability-based", "suitability"),
        (df_ar, colors[1], "Autoregressive", "autoregressive"),
    ]:
        doy = _doy(df["date"])
        doys.append(doy)
        ax.plot(doy, df["additional_case_prob"], color=color, label=label)
        # Dashed overlay: the retrospective full-reporting fit, i.e. the probability if the whole
        # outbreak (all future onsets, full reporting) were known.
        df_full = pd.read_csv(inputs["full_reporting_paths"][full_key])
        full_doy = _doy(
            full_start + pd.to_timedelta(df_full["day_of_outbreak"], unit="D")
        )
        ax.plot(
            full_doy, df_full["additional_case_prob"], color=color, linestyle="dashed"
        )
    # A neutral proxy entry names the dashed style without doubling the legend.
    ax.plot(
        [], [], color="tab:gray", linestyle="dashed", label="Full outbreak knowledge"
    )
    # Decision markers follow the main Lazio (figure 4) colours: C3 (blood), C4 (45-day rule).
    marker_doys = [
        decisions["blood_resumed_anzio"]["doy"],
        decisions["45_day_rule"]["doy"],
    ]
    ax.axvline(
        decisions["blood_resumed_anzio"]["doy"],
        color=colors[3],
        linestyle="dotted",
        label="Blood measures\nlifted",
    )
    ax.axvline(
        decisions["45_day_rule"]["doy"],
        color=colors[4],
        linestyle="dotted",
        label="45-day rule",
    )
    # Keep both decision markers (plus a margin) in view even when the estimate stops earlier.
    left = min(d.min() for d in doys)
    right = max(max(d.max() for d in doys), max(marker_doys)) + 6
    ax.set_xlim(left, right)
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["additional_case_prob"])
    return fig, ax


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2017-10-01")
    parser.add_argument("--end-date", default="2017-12-31")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    make_plots(start_date=args.start_date, end_date=args.end_date, stride=args.stride)
