"""Plots for the under-reporting simulation study."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd import calc_decision_delay
from endoutbreakvbd.utils import (
    get_colors,
    plot_data_on_twin_ax,
    set_plot_config,
)
from scripts.inputs import get_inputs_sim_underreporting

# Reported cases in neutral gray; unreported (true minus reported) in a distinct colour so
# it stands out against the reported bars and the inferred-cases band.
_REPORTED_COLOR = "tab:gray"
_UNREPORTED_COLOR = "tab:purple"


def make_plots():
    set_plot_config()
    inputs = get_inputs_sim_underreporting()
    colors = get_colors()
    df = pd.read_csv(inputs["results_paths"]["sim"])
    day = df["day_of_outbreak"].to_numpy()
    reported = df["reported"].to_numpy()
    unreported = df["true"].to_numpy() - reported
    # The projected decision-day value is NaN, not an observed zero; exclude it when locating the
    # final reported case used as the decision-delay origin.
    time_final_reported = int(np.flatnonzero(np.nan_to_num(reported, nan=0.0))[-1])

    # Cases: true cases split into reported/unreported (stacked), plus the inferred true.
    # The inferred credible band is drawn behind the bars (zorder) so the true cases stay
    # visible; the band then shows where it exceeds the data (mostly the outbreak tail).
    fig, ax = plt.subplots()
    ax.fill_between(
        day, df["cases_lower"], df["cases_upper"], color=colors[0], alpha=0.3, zorder=1
    )
    ax.bar(day, reported, color=_REPORTED_COLOR, label="Reported", zorder=2)
    ax.bar(
        day,
        unreported,
        bottom=reported,
        color=_UNREPORTED_COLOR,
        label="Unreported (true)",
        zorder=2,
    )
    ax.plot(day, df["cases_mean"], color=colors[0], label="Inferred true", zorder=3)
    ax.set_xlabel("Day of outbreak")
    ax.set_ylim(0, None)
    ax.set_ylabel("Cases per day")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["cases"])

    # Reproduction number: true vs inferred (under-reporting) vs naive, over reported cases.
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, day, [reported, unreported])
    ax.plot(day, df["rep_no_true"], color="black", linestyle="dashed", label="True")
    ax.plot(day, df["rep_no_mean"], color=colors[0], label="Under-reporting (est. R)")
    ax.fill_between(
        day, df["rep_no_lower"], df["rep_no_upper"], color=colors[0], alpha=0.3
    )
    ax.plot(
        day, df["rep_no_naive_mean"], color=colors[1], label="Naive (reported only)"
    )
    ax.set_xlabel("Day of outbreak")
    ax.set_ylim(0, min(ax.get_ylim()[1], 6))
    ax.set_ylabel("Reproduction number")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["rep_no"])

    # Additional-case probability: true vs under-reporting (est./true R) vs naive, with a
    # credible interval on the true-R under-reporting model (isolating reporting uncertainty).
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, day, [reported, unreported])
    for suffix, color, label in _prob_series(colors):
        ax.plot(day, df[f"additional_case_prob_{suffix}"], color=color, label=label)
    ax.fill_between(
        day,
        df["additional_case_prob_known_r_lower"],
        df["additional_case_prob_known_r_upper"],
        color=colors[2],
        alpha=0.3,
    )
    ax.set_xlabel("Day of outbreak")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Probability of additional cases")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["additional_case_prob"])

    # Decision delay vs threshold.
    thresholds = inputs["perc_risk_threshold_grid"]
    fig, ax = plt.subplots()
    for suffix, color, label in _prob_series(colors):
        delays = calc_decision_delay(
            prob_vec=df[f"additional_case_prob_{suffix}"].to_numpy(),
            days=day,
            perc_risk_threshold=thresholds,
            time_final_case=time_final_reported,
        )
        ax.plot(thresholds, delays, color=color, label=label)
    ax.set_xlim(thresholds[0], thresholds[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from last reported case\nuntil risk falls below threshold")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["decision"])


def _prob_series(colors):
    # (column suffix, colour, label) for the four additional-case-probability curves.
    return [
        ("true", "black", "True"),
        ("est_r", colors[0], "Under-reporting (est. R)"),
        ("known_r", colors[2], "Under-reporting (true R)"),
        ("naive", colors[1], "Naive (reported only)"),
    ]


if __name__ == "__main__":
    make_plots()
