"""Plots for the quasi-real-time under-reporting nowcast.

Three panels: the fitted onset-to-report delay distribution, the latest-snapshot true-case
trajectory, and the real-time additional-case probability with the retrospective
"full outbreak knowledge" benchmark (the full-reporting ``lazio_outbreak`` fits) overlaid as
dashed lines. ``_make_cases_plot`` and ``_make_prob_plot`` are shared with the retrospective
under-reporting analysis (``lazio_underreporting_retro_plots``).

Builds its own calendar-date axis from the reporting-matrix snapshot dates rather than routing
through the ``doy_vec``-zipped ``lazio_outbreak_plots`` helpers (whose strict zip assumes the
retrospective grid length).
"""

import argparse

import matplotlib.pyplot as plt
import pandas as pd

from endoutbreakvbd.utils import (
    dates_to_day_index,
    get_colors,
    month_start_xticks,
    plot_incidence_on_twin_ax,
    set_plot_config,
)
from scripts.inputs import get_inputs_lazio_underreporting_qrt


def make_plots(start_date="2017-09-30", end_date="2017-12-31", stride=1):
    set_plot_config()
    inputs = get_inputs_lazio_underreporting_qrt(
        start_date=start_date, end_date=end_date, stride=stride
    )
    colors = get_colors()
    _make_delay_plot(inputs)
    _make_cases_plot(inputs, colors)
    _make_prob_plot(inputs, colors, ylabel="Real-time probability of additional cases")


def _make_delay_plot(inputs):
    delay_df = pd.read_csv(inputs["results_paths"]["delay"])
    fig, ax = plt.subplots()
    ax.bar(
        delay_df["delay"],
        delay_df["pmf_empirical"],
        color="tab:gray",
        alpha=0.6,
        label="Empirical",
    )
    ax.plot(
        delay_df["delay"],
        delay_df["pmf_fitted"],
        color="tab:blue",
        label="Fitted gamma",
    )
    ax.set_xlim(0, None)
    ax.set_xlabel("Onset-to-report delay (days)")
    ax.set_ylabel("Probability")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["delay"])
    return fig, ax


def _make_cases_plot(inputs, colors):
    trajectory_df = _read_case_trajectory(inputs)
    day_index_vec = dates_to_day_index(trajectory_df["date"])
    fig, ax = plt.subplots()
    ax.bar(
        day_index_vec,
        trajectory_df["reported_incidence"],
        color="tab:gray",
        alpha=0.5,
        label="Reported",
    )
    ax.plot(
        day_index_vec,
        trajectory_df["incidence_mean"],
        color=colors[0],
        label="Estimated true\n(suitability-based)",
    )
    ax.fill_between(
        day_index_vec,
        trajectory_df["incidence_lower"],
        trajectory_df["incidence_upper"],
        color=colors[0],
        alpha=0.3,
    )
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, None)
    ax.set_ylabel("Number of cases")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["incidence"])
    return fig, ax


def _make_prob_plot(
    inputs,
    colors,
    *,
    ylabel="Probability of additional cases",
):
    suitability_df = pd.read_csv(
        inputs["results_paths"]["suitability_p60"], parse_dates=["date"]
    )
    autoregressive_df = pd.read_csv(
        inputs["results_paths"]["autoregressive_p60"], parse_dates=["date"]
    )
    trajectory_df = _read_case_trajectory(inputs)
    decisions = inputs["existing_decisions"]
    full_reporting_outbreak_start_date = inputs["full_reporting_outbreak_start_date"]

    fig, ax = plt.subplots()
    # Observed reported cases (by onset) as a twin-axis underlay, matching the main Lazio panels.
    plot_incidence_on_twin_ax(
        ax,
        dates_to_day_index(trajectory_df["date"]),
        trajectory_df["reported_incidence"].to_numpy(),
    )
    day_index_vectors = []
    for results_df, color, label, full_key in [
        (suitability_df, colors[0], "Suitability-based", "suitability"),
        (autoregressive_df, colors[1], "Autoregressive", "autoregressive"),
    ]:
        day_index_vec = dates_to_day_index(results_df["date"])
        day_index_vectors.append(day_index_vec)
        ax.plot(
            day_index_vec,
            results_df["additional_case_prob"],
            color=color,
            label=label,
        )
        # Dashed overlay: the retrospective full-reporting fit, i.e. the probability if the whole
        # outbreak (all future onsets, full reporting) were known.
        full_reporting_df = pd.read_csv(inputs["full_reporting_paths"][full_key])
        full_reporting_day_index_vec = dates_to_day_index(
            full_reporting_outbreak_start_date
            + pd.to_timedelta(full_reporting_df["day_of_outbreak"], unit="D")
        )
        ax.plot(
            full_reporting_day_index_vec,
            full_reporting_df["additional_case_prob"],
            color=color,
            linestyle="dashed",
            label=f"{label}\n(full reporting)",
        )
    # Decision markers follow the main Lazio outbreak colours: C3 (blood), C4 (45-day rule).
    marker_day_index_vals = [
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
    # Focus on the decision window: from the first estimate (the first decision date; with the
    # snapshot grid starting 30 Sep this is 1 Oct, whose month tick then shows) to just past the
    # later marker; the estimate may run on into a flat post-outbreak tail, which is clipped.
    ax.set_xlim(
        min(day_index_vec.min() for day_index_vec in day_index_vectors),
        max(marker_day_index_vals) + 6,
    )
    month_start_xticks(ax)
    ax.set_xlabel("Date (2017)")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel(ylabel)
    ax.legend(loc="upper left", bbox_to_anchor=(0, 0.89))
    fig.savefig(inputs["fig_paths"]["additional_case_prob"])
    return fig, ax


def _read_case_trajectory(inputs):
    """Read the case trajectory from the QRT or retrospective result layout."""
    result_key = (
        "trajectory" if "trajectory" in inputs["results_paths"] else "suitability_p60"
    )
    return pd.read_csv(inputs["results_paths"][result_key], parse_dates=["date"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--start-date", default="2017-10-01")
    parser.add_argument("--end-date", default="2017-12-31")
    parser.add_argument("--stride", type=int, default=1)
    args = parser.parse_args()
    make_plots(start_date=args.start_date, end_date=args.end_date, stride=args.stride)
