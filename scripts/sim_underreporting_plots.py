"""Plots for the under-reporting simulation study."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd import calc_decision_delay
from endoutbreakvbd.utils import (
    get_colors,
    plot_incidence_on_twin_ax,
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
    sim_df = pd.read_csv(inputs["results_paths"]["sim"])
    t_vec = sim_df["day_of_outbreak"].to_numpy()
    reported_incidence_vec = sim_df["reported_incidence"].to_numpy()
    unreported_incidence_vec = (
        sim_df["true_incidence"].to_numpy() - reported_incidence_vec
    )
    # The projected decision-day value is NaN, not an observed zero; exclude it when locating the
    # final reported case used as the decision-delay origin.
    t_final_reported_case = int(
        np.flatnonzero(np.nan_to_num(reported_incidence_vec, nan=0.0))[-1]
    )

    # Cases: true cases split into reported/unreported (stacked), plus the inferred true.
    # The inferred credible band is drawn behind the bars (zorder) so the true cases stay
    # visible; the band then shows where it exceeds the data (mostly the outbreak tail).
    fig, ax = plt.subplots()
    ax.fill_between(
        t_vec,
        sim_df["incidence_lower"],
        sim_df["incidence_upper"],
        color=colors[0],
        alpha=0.3,
        zorder=1,
    )
    ax.bar(
        t_vec, reported_incidence_vec, color=_REPORTED_COLOR, label="Reported", zorder=2
    )
    ax.bar(
        t_vec,
        unreported_incidence_vec,
        bottom=reported_incidence_vec,
        color=_UNREPORTED_COLOR,
        label="Unreported (true)",
        zorder=2,
    )
    ax.plot(
        t_vec,
        sim_df["incidence_mean"],
        color=colors[0],
        label="Inferred true",
        zorder=3,
    )
    ax.set_xlabel("Day of outbreak")
    ax.set_ylim(0, None)
    ax.set_ylabel("Cases per day")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["incidence"])

    # Reproduction number: true vs inferred (under-reporting) vs naive, over reported cases.
    fig, ax = plt.subplots()
    plot_incidence_on_twin_ax(
        ax, t_vec, [reported_incidence_vec, unreported_incidence_vec]
    )
    ax.plot(
        t_vec, sim_df["rep_no_true"], color="black", linestyle="dashed", label="True"
    )
    ax.plot(
        t_vec, sim_df["rep_no_mean"], color=colors[0], label="Under-reporting (est. R)"
    )
    ax.fill_between(
        t_vec,
        sim_df["rep_no_lower"],
        sim_df["rep_no_upper"],
        color=colors[0],
        alpha=0.3,
    )
    ax.plot(
        t_vec,
        sim_df["rep_no_naive_mean"],
        color=colors[1],
        label="Naive (reported only)",
    )
    ax.set_xlabel("Day of outbreak")
    ax.set_ylim(0, min(ax.get_ylim()[1], 6))
    ax.set_ylabel("Reproduction number")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["rep_no"])

    # Additional-case probability: true vs under-reporting (est./true R) vs naive, with a
    # credible interval on the true-R under-reporting model (isolating reporting uncertainty).
    fig, ax = plt.subplots()
    plot_incidence_on_twin_ax(
        ax, t_vec, [reported_incidence_vec, unreported_incidence_vec]
    )
    for suffix, color, label in _prob_series(colors):
        ax.plot(
            t_vec, sim_df[f"additional_case_prob_{suffix}"], color=color, label=label
        )
    ax.fill_between(
        t_vec,
        sim_df["additional_case_prob_known_r_lower"],
        sim_df["additional_case_prob_known_r_upper"],
        color=colors[2],
        alpha=0.3,
    )
    ax.set_xlabel("Day of outbreak")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Risk of additional cases")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["additional_case_prob"])

    # Decision delay vs threshold.
    risk_threshold_pct_vec = inputs["risk_threshold_pct_grid"]
    fig, ax = plt.subplots()
    for suffix, color, label in _prob_series(colors):
        decision_delay_vec = calc_decision_delay(
            prob_vec=sim_df[f"additional_case_prob_{suffix}"].to_numpy(),
            t_vec=t_vec,
            risk_threshold_pct=risk_threshold_pct_vec,
            t_final_case=t_final_reported_case,
        )
        ax.plot(risk_threshold_pct_vec, decision_delay_vec, color=color, label=label)
    ax.set_xlim(risk_threshold_pct_vec[0], risk_threshold_pct_vec[-1])
    ax.set_ylim(0, None)
    ax.set_xlabel("Risk threshold (%)")
    ax.set_ylabel("Days from last reported case\nuntil risk falls below threshold")
    ax.legend(loc="upper right")
    fig.savefig(inputs["fig_paths"]["decision_delay"])


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
