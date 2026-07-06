"""Plots for the under + delayed reporting nowcast verification (figure S9).

Panel A: the (synthetic) onset-to-report delay distribution used to right-truncate the reported
series. Panel B (verification): on a twin (cases) axis the true incidence up to the snapshot split
into reported / reported-later / never-reported stacked bars (which sum to the truth), plus the
under-reporting model's inferred true-case band; on the primary (probability) axis the additional-
case probability at the decision day for the five analyses, markers x-dodged, with posterior error
bars on the three MCMC fits (the two analytical benchmarks are exact points). A faint vertical line
marks the snapshot.
"""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd.utils import get_colors, set_plot_config
from scripts.inputs import get_inputs_sim_underreporting_nowcast

# Reporting-status bar colours (a gradient of "how reported": known now / coming / never), distinct
# from the categorical method colours used for the probability markers.
_REPORTED_COLOR = "tab:gray"
_NOT_YET_COLOR = "tab:orange"
_NEVER_COLOR = "tab:purple"


def _prob_series(colors):
    # (method, colour, label) for the five additional-case-probability markers. Colours match the
    # under-reporting simulation study (figure S8) for the four shared methods; colors[3] (red) is
    # the extra factor-isolation series, naive at true R.
    return [
        ("true", "black", "True"),
        ("naive_true_r", colors[3], "Naive (true R)"),
        ("naive_est_r", colors[1], "Naive (est. R)"),
        ("imperfect_true_r", colors[2], "Under-reporting (true R)"),
        ("imperfect_est_r", colors[0], "Under-reporting (est. R)"),
    ]


def _make_delay_plot(inputs):
    # Panel A: the synthetic onset-to-report delay distribution (its CDF is the model's delay_cdf).
    delay = inputs["delay"]
    fig, ax = plt.subplots()
    ax.bar(delay["support"], delay["pmf"], color="tab:gray", alpha=0.6)
    # Crop the near-zero upper tail (the support runs well past it so the Cori residual dump is
    # negligible); show the bulk of the distribution.
    ax.set_xlim(0, int(np.searchsorted(delay["cdf"], 0.995)))
    ax.set_ylim(0, None)
    ax.set_xlabel("Onset-to-report delay (days)")
    ax.set_ylabel("Probability")
    fig.savefig(inputs["fig_paths"]["delay"])
    return fig, ax


def make_plots():
    set_plot_config()
    inputs = get_inputs_sim_underreporting_nowcast()
    colors = get_colors()
    _make_delay_plot(inputs)
    df = pd.read_csv(inputs["results_paths"]["trajectory"])
    df_probs = pd.read_csv(inputs["results_paths"]["probs"]).set_index("method")
    day = df["onset_day"].to_numpy()
    reported = df["reported"].to_numpy()
    not_yet = df["not_yet"].to_numpy()
    never = df["never"].to_numpy()
    snapshot_day = int(inputs["snapshot_day"])
    t_calc = snapshot_day + 1  # the decision day (start of the day after the snapshot)

    fig, ax = plt.subplots()

    # Cases on a twin axis (styled like utils.plot_data_on_twin_ax): stacked reporting-status bars
    # summing to the true incidence, over the inferred true-case band. The credible band is drawn
    # behind the bars (as in figure S8A) so the true-case categories stay legible; the band and its
    # mean line then show where the model recovers the hidden (later + never reported) cases.
    twin_ax = ax.twinx()
    twin_ax.fill_between(
        day, df["cases_lower"], df["cases_upper"], color=colors[0], alpha=0.25, zorder=1
    )
    twin_ax.bar(
        day,
        reported,
        color=_REPORTED_COLOR,
        alpha=0.7,
        label="Reported (by snapshot)",
        zorder=2,
    )
    twin_ax.bar(
        day,
        not_yet,
        bottom=reported,
        color=_NOT_YET_COLOR,
        alpha=0.7,
        label="Reported later",
        zorder=2,
    )
    twin_ax.bar(
        day,
        never,
        bottom=reported + not_yet,
        color=_NEVER_COLOR,
        alpha=0.7,
        label="Never reported",
        zorder=2,
    )
    twin_ax.plot(
        day,
        df["cases_mean"],
        color=colors[0],
        label="Estimated true",
        zorder=3,
    )
    twin_ax.set_ylim(
        0, 1.05 * max((reported + not_yet + never).max(), df["cases_upper"].max())
    )
    twin_ax.set_ylabel("Number of cases")
    twin_ax.yaxis.label.set_color("tab:gray")
    twin_ax.tick_params(axis="y", colors="tab:gray")
    twin_ax.spines["right"].set_visible(True)
    twin_ax.spines["right"].set_color("tab:gray")
    twin_ax.spines["left"].set_visible(False)
    twin_ax.spines["bottom"].set_visible(False)
    # Draw the primary axis (markers, snapshot line, legend) above the incidence bars.
    ax.set_zorder(twin_ax.get_zorder() + 1)
    ax.patch.set_visible(False)

    ax.axvline(
        snapshot_day, color="tab:gray", linestyle="dashed", alpha=0.6, label="Snapshot"
    )

    # All five markers report the same decision day (D + 1); they are x-dodged into the gap just
    # right of the snapshot line only so the error bars are readable.
    series = _prob_series(colors)
    x0 = t_calc + 0.5
    dodge = (np.arange(len(series)) - (len(series) - 1) / 2) * 0.6
    for (method, color, label), dx in zip(series, dodge):
        row = df_probs.loc[method]
        prob, lower, upper = row["prob"], row["prob_lower"], row["prob_upper"]
        if np.isnan(lower):
            ax.plot(
                x0 + dx,
                prob,
                marker="o",
                linestyle="none",
                color=color,
                label=label,
            )
        else:
            yerr = [[max(prob - lower, 0.0)], [max(upper - prob, 0.0)]]
            ax.errorbar(
                x0 + dx,
                prob,
                yerr=yerr,
                fmt="o",
                color=color,
                capsize=3,
                label=label,
            )

    ax.set_xlim(0, t_calc + 5)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Day of outbreak")
    ax.set_ylabel("Probability of additional cases")

    # One combined legend (probability markers + snapshot from the primary axis, reporting-status
    # bars + inferred-cases band from the twin axis).
    handles, labels = ax.get_legend_handles_labels()
    handles_cases, labels_cases = twin_ax.get_legend_handles_labels()
    ax.legend(
        handles + handles_cases, labels + labels_cases, loc="upper left", fontsize=9
    )

    fig.savefig(inputs["fig_paths"]["verification"])
    return fig, ax


if __name__ == "__main__":
    make_plots()
