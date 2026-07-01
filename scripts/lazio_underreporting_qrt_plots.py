"""Plots for the quasi-real-time under-reporting analysis (figure 5 + figure S5).

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

_PERC_RISK_THRESHOLDS = np.linspace(0.1, 10, 101)


def _doy(dates) -> np.ndarray:
    return pd.DatetimeIndex(dates).dayofyear.to_numpy()


def _decision_delays(calc_times, prob_vec, thresholds, time_final_case) -> np.ndarray:
    # Days from the last observed case until the real-time probability first drops below
    # each threshold (NaN where it never does over the snapshot window).
    delays = np.full(len(thresholds), np.nan)
    for j, threshold in enumerate(thresholds):
        below = np.nonzero(prob_vec < threshold / 100)[0]
        if below.size:
            delays[j] = calc_times[below[0]] - time_final_case
    return delays


def make_plots(start_date="2017-11-01", end_date="2017-12-20", stride=1):
    set_plot_config()
    inputs = get_inputs_lazio_underreporting_qrt(
        start_date=start_date, end_date=end_date, stride=stride
    )
    colors = get_colors()
    _make_cases_plot(inputs, colors)
    _make_prob_plot(inputs, colors)
    _make_decision_plots(inputs, colors)
    _make_estimate_plots(inputs)
    _make_delay_plot(inputs)


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


def _make_prob_plot(inputs, colors):
    df_suit = pd.read_csv(
        inputs["results_paths"]["suitability_p60"], parse_dates=["date"]
    )
    df_ar = pd.read_csv(
        inputs["results_paths"]["autoregressive_p60"], parse_dates=["date"]
    )
    decisions = inputs["existing_decisions"]
    fig, ax = plt.subplots()
    for df, color, label in [
        (df_suit, colors[0], "Suitability-based"),
        (df_ar, colors[1], "Autoregressive"),
    ]:
        ax.plot(_doy(df["date"]), df["additional_case_prob"], color=color, label=label)
    ax.axvline(
        decisions["blood_resumed_anzio"]["doy"],
        color=colors[-2],
        linestyle="dashed",
        label="Blood measures\nlifted",
    )
    ax.axvline(
        decisions["45_day_rule"]["doy"],
        color=colors[-1],
        linestyle="dashed",
        label="45-day rule",
    )
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Real-time probability of additional cases")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["additional_case_prob"])
    return fig, ax


def _make_decision_plots(inputs, colors):
    time_final_case = inputs["time_final_case"]
    calc_times = inputs["calc_times"]
    decisions = inputs["existing_decisions"]

    df_suit = pd.read_csv(inputs["results_paths"]["suitability_p60"])
    df_ar = pd.read_csv(inputs["results_paths"]["autoregressive_p60"])

    # Panel C: suitability vs autoregressive at 60%.
    fig, ax = plt.subplots()
    for df, color, label in [
        (df_suit, colors[0], "Suitability-based"),
        (df_ar, colors[1], "Autoregressive"),
    ]:
        delays = _decision_delays(
            calc_times,
            df["additional_case_prob"].to_numpy(),
            _PERC_RISK_THRESHOLDS,
            time_final_case,
        )
        ax.plot(_PERC_RISK_THRESHOLDS, delays, color=color, label=label)
    ax.axhline(
        decisions["blood_resumed_anzio"]["days_from_final_case"],
        color=colors[-1],
        linestyle="dashed",
        label="Blood measures lifted",
    )
    ax.set_xlim(_PERC_RISK_THRESHOLDS[0], _PERC_RISK_THRESHOLDS[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["decision"])

    # Panel D: reporting-probability sensitivity (suitability at 60/80/100%).
    fig, ax = plt.subplots()
    for name, prob_label, color in [
        ("suitability_p60", "60%", colors[0]),
        ("suitability_p80", "80%", colors[2]),
        ("suitability_p100", "100%", colors[3]),
    ]:
        df = pd.read_csv(inputs["results_paths"][name])
        delays = _decision_delays(
            calc_times,
            df["additional_case_prob"].to_numpy(),
            _PERC_RISK_THRESHOLDS,
            time_final_case,
        )
        ax.plot(
            _PERC_RISK_THRESHOLDS, delays, color=color, label=f"Reporting {prob_label}"
        )
    ax.set_xlim(_PERC_RISK_THRESHOLDS[0], _PERC_RISK_THRESHOLDS[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["reporting_sensitivity"])


def _make_estimate_plots(inputs):
    df = pd.read_csv(inputs["results_paths"]["trajectory"], parse_dates=["date"])
    doy = _doy(df["date"])
    reported = df["reported"].to_numpy()
    suitability_prior = inputs["suitability_mean_vec"][: len(doy)]

    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, doy, reported)
    ax.plot(doy, df["suitability_mean"], color="tab:blue", label="Posterior")
    ax.fill_between(
        doy,
        df["suitability_lower"],
        df["suitability_upper"],
        color="tab:blue",
        alpha=0.3,
    )
    ax.plot(
        doy,
        suitability_prior,
        color="black",
        linestyle="dashed",
        label="Seasonal prior",
    )
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Temperature suitability for transmission")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["suitability"])

    for column, ylabel, fig_key in [
        ("rep_no_factor", "Reproduction number scaling factor", "scaling_factor"),
        ("reproduction_number", "Time-dependent reproduction number", "rep_no"),
    ]:
        fig, ax = plt.subplots()
        plot_data_on_twin_ax(ax, doy, reported)
        ax.plot(doy, df[f"{column}_mean"], color="tab:blue", label="Posterior")
        ax.fill_between(
            doy,
            df[f"{column}_lower"],
            df[f"{column}_upper"],
            color="tab:blue",
            alpha=0.3,
        )
        month_start_xticks(ax)
        ax.set_xlabel("Date (2017)")
        ax.set_ylim(0, np.minimum(ax.get_ylim()[1], 10))
        ax.set_ylabel(ylabel)
        fig.savefig(inputs["fig_paths"][fig_key])


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


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2017-11-01")
    parser.add_argument("--end-date", default="2017-12-20")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    make_plots(start_date=args.start_date, end_date=args.end_date, stride=args.stride)
