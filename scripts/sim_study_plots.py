import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from endoutbreakvbd.utils import (
    month_start_xticks,
    plot_data_on_twin_ax,
    set_plot_config,
)
from scripts.inputs import get_inputs_sim_study


def make_plots():
    set_plot_config()
    inputs = get_inputs_sim_study()
    _make_rep_no_plot(
        rep_no_func_doy=inputs["rep_no_func_doy"],
        example_doy_vals=inputs["example_outbreak_doy_start_vals"],
        save_path=inputs["fig_paths"]["rep_no"],
    )
    _make_example_outbreak_prob_plot(
        incidence_vec=inputs["example_outbreak_incidence_vec"],
        data_path=inputs["results_paths"]["example_outbreak_prob"],
        save_path=inputs["fig_paths"]["example_outbreak_prob"],
    )
    _make_example_outbreak_decision_plot(
        data_path=inputs["results_paths"]["example_outbreak_decision"],
        save_path=inputs["fig_paths"]["example_outbreak_decision"],
    )
    _make_many_outbreak_example_plot(
        perc_risk_threshold_vals=inputs["many_outbreak_perc_risk_threshold_vals"],
        data_path=inputs["results_paths"]["many_outbreak_example"],
        save_path=inputs["fig_paths"]["many_outbreak_example"],
    )
    _make_many_outbreak_decision_plot(
        perc_risk_threshold_vals=inputs["many_outbreak_perc_risk_threshold_vals"],
        example_outbreak_idx=inputs["many_outbreak_example_outbreak_idx"],
        data_path=inputs["results_paths"]["many_outbreak_decision"],
        save_path=inputs["fig_paths"]["many_outbreak_decision"],
    )


def _make_rep_no_plot(*, rep_no_func_doy, example_doy_vals, save_path=None):
    fig, ax = plt.subplots()
    doy_vec = np.arange(1, 366)
    ax.plot(doy_vec, rep_no_func_doy(doy_vec))
    ax.plot(example_doy_vals, rep_no_func_doy(np.array(example_doy_vals)), "o")
    month_start_xticks(ax, interval_months=2)
    ax.set_xlabel("Date")
    ax.set_ylim(0, 2.02)
    ax.set_ylabel("Time-dependent reproduction number")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_example_outbreak_prob_plot(*, incidence_vec, data_path, save_path=None):
    nontrivial_outbreak = np.sum(incidence_vec) > 1
    prob_df = pd.read_csv(data_path, index_col=[0, 1])
    doy_start_vals = prob_df.index.get_level_values("initial_case_day_of_year").unique()
    prob_days = prob_df.index.get_level_values("day_of_outbreak").unique()

    fig, ax = plt.subplots()
    if nontrivial_outbreak:
        plot_data_on_twin_ax(
            ax, t_vec=np.arange(len(incidence_vec)), incidence_vec=incidence_vec
        )
    for doy_start, color in zip(
        doy_start_vals,
        ["tab:blue", "tab:orange"],
        strict=True,
    ):
        date_start = pd.Timestamp(year=2017, month=1, day=1) + pd.Timedelta(
            days=doy_start - 1
        )
        prob_vals = prob_df.loc[doy_start, "analytical"].to_numpy()
        ax.plot(
            prob_days,
            prob_vals,
            label=f"Initial case on {date_start.day} {date_start:%b}",
            color=color,
        )
        prob_vals_sim = prob_df.loc[doy_start, "simulation"].to_numpy()
        ax.plot(prob_days, prob_vals_sim, ".", color=color)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel(
        "Day of outbreak" if nontrivial_outbreak else "Days since initial case"
    )
    if not nontrivial_outbreak:
        ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylabel("Probability of additional cases")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_example_outbreak_decision_plot(*, data_path, save_path=None):
    decision_delay_df = pd.read_csv(data_path, index_col=[0, 1])
    perc_risk_threshold_vals = decision_delay_df.index.get_level_values(
        "perc_risk_threshold"
    ).unique()
    doy_final_case_vec = decision_delay_df.index.get_level_values(
        "final_case_day_of_year"
    ).unique()

    fig, ax = plt.subplots()

    for perc_risk_threshold in perc_risk_threshold_vals:
        decision_delay_vec = decision_delay_df.loc[
            perc_risk_threshold, "delay_to_decision"
        ].to_numpy()
        ax.plot(
            doy_final_case_vec,
            decision_delay_vec,
            label=f"{perc_risk_threshold}% risk threshold",
        )
    month_start_xticks(ax, interval_months=2)
    ax.set_xlabel("Date of last observed case")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_many_outbreak_example_plot(
    *,
    perc_risk_threshold_vals,
    cmap_names=("Blues", "Oranges", "Greens", "Purples"),
    data_path,
    save_path=None,
):
    df = pd.read_csv(data_path)
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, t_vec=df["day_of_year"], incidence_vec=df["cases"])
    ax.plot(df["day_of_year"], df["additional_case_prob"], color="black")
    for perc_risk_threshold, cmap_name in zip(
        perc_risk_threshold_vals,
        cmap_names[: len(perc_risk_threshold_vals)],
        strict=True,
    ):
        # Match each threshold line to its series colour in the decision plot.
        ax.axhline(
            perc_risk_threshold / 100,
            color=matplotlib.colormaps[cmap_name](0.7),
            linestyle="--",
        )
    month_start_xticks(ax)
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Probability of additional cases")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_many_outbreak_decision_plot(
    *,
    data_path,
    perc_risk_threshold_vals,
    cmap_names=("Blues", "Oranges", "Greens", "Purples"),
    marker_colors=("blue", "red", "green", "purple"),
    example_outbreak_idx=None,
    xlim=None,
    ylim=None,
    xtick_interval_months=1,
    save_path=None,
):
    df = pd.read_csv(data_path)
    bin_width = 7
    df["final_case_doy_binned"] = pd.cut(
        df["final_case_day_of_year"],
        bins=range(1, 366, bin_width),
        right=False,
        include_lowest=True,
    )
    # Per-week frequency of outbreaks (shared across thresholds) sets the bar shade.
    proportions = df["final_case_doy_binned"].value_counts(normalize=True).sort_index()
    norm = plt.Normalize(0, 0.01)

    n_thresholds = len(perc_risk_threshold_vals)
    # Offset each threshold's markers horizontally so overlapping error bars separate.
    offset_step = 1.75
    x_offsets = (np.arange(n_thresholds) - (n_thresholds - 1) / 2) * offset_step

    fig, ax = plt.subplots()
    legend_handles = []
    for x_offset, perc_risk_threshold, cmap_name, marker_color in zip(
        x_offsets,
        perc_risk_threshold_vals,
        cmap_names[:n_thresholds],
        marker_colors[:n_thresholds],
        strict=True,
    ):
        cmap = matplotlib.colormaps[cmap_name]
        delay_col = f"delay_to_decision_{perc_risk_threshold}"
        stats = (
            df.groupby("final_case_doy_binned", observed=False)[delay_col]
            .quantile(np.array([0.025, 0.5, 0.975]))
            .unstack()
        )
        bin_centres = np.array([interval.mid for interval in stats.index])
        colors = cmap(norm(proportions.to_numpy()))
        for x, m, lo, hi, c in zip(
            bin_centres + x_offset,
            stats[0.5],
            stats[0.5] - stats[0.025],
            stats[0.975] - stats[0.5],
            colors,
        ):
            ax.errorbar(
                x,
                m,
                yerr=[[lo], [hi]],
                fmt="o",
                color=c,
                capsize=3,
            )
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=cmap(0.7),
                label=f"{perc_risk_threshold}% threshold",
            )
        )
        if example_outbreak_idx is not None:
            example_outbreak_delay = df.loc[example_outbreak_idx, delay_col]
            example_outbreak_final_case_bin_centre = (
                df.loc[example_outbreak_idx, "final_case_doy_binned"].mid + x_offset
            )
            ax.plot(
                example_outbreak_final_case_bin_centre,
                example_outbreak_delay,
                marker="x",
                color=marker_color,
                linewidth=2,
            )
    ax.set_xlabel("Week of last observed case")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
    if xlim is not None:
        ax.set_xlim(*xlim)
    if ylim is not None:
        ax.set_ylim(*ylim)
    month_start_xticks(ax, interval_months=xtick_interval_months)
    if len(perc_risk_threshold_vals) > 1:
        ax.legend(handles=legend_handles, loc="lower center")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


if __name__ == "__main__":
    make_plots()
