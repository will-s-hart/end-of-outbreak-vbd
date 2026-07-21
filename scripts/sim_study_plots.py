import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from endoutbreakvbd.utils import (
    month_start_xticks,
    plot_incidence_on_twin_ax,
    set_plot_config,
)
from scripts.inputs import get_inputs_sim_study


def make_plots():
    set_plot_config()
    inputs = get_inputs_sim_study()
    _make_rep_no_plot(
        rep_no_doy_func=inputs["rep_no_doy_func"],
        example_outbreak_doy_start_vals=inputs["example_outbreak_doy_start_vals"],
        fig_path=inputs["fig_paths"]["rep_no"],
    )
    _make_additional_case_prob_plot(
        incidence_vec=inputs["example_outbreak_incidence_vec"],
        results_path=inputs["results_paths"]["example_outbreak_prob"],
        fig_path=inputs["fig_paths"]["example_outbreak_prob"],
    )
    _make_decision_delay_plot(
        results_path=inputs["results_paths"]["example_outbreak_decision_delay"],
        fig_path=inputs["fig_paths"]["example_outbreak_decision_delay"],
    )
    _make_many_outbreak_example_plot(
        risk_threshold_pct_vals=inputs["many_outbreak_risk_threshold_pct_vals"],
        results_path=inputs["results_paths"]["many_outbreak_example"],
        fig_path=inputs["fig_paths"]["many_outbreak_example"],
    )
    _make_many_outbreak_decision_plot(
        risk_threshold_pct_vals=inputs["many_outbreak_risk_threshold_pct_vals"],
        example_outbreak_idx=inputs["many_outbreak_example_idx"],
        results_path=inputs["results_paths"]["many_outbreak_decision_delay"],
        fig_path=inputs["fig_paths"]["many_outbreak_decision_delay"],
    )


def _make_rep_no_plot(
    *, rep_no_doy_func, example_outbreak_doy_start_vals, fig_path=None
):
    fig, ax = plt.subplots()
    doy_vec = np.arange(1, 366)
    ax.plot(doy_vec, rep_no_doy_func(doy_vec))
    ax.plot(
        example_outbreak_doy_start_vals,
        rep_no_doy_func(np.array(example_outbreak_doy_start_vals)),
        "o",
    )
    month_start_xticks(ax, interval_months=2)
    ax.set_xlabel("Date")
    ax.set_ylim(0, 2.02)
    ax.set_ylabel("Time-dependent reproduction number")
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


def _make_additional_case_prob_plot(*, incidence_vec, results_path, fig_path=None):
    is_nontrivial_outbreak = np.sum(incidence_vec) > 1
    prob_df = pd.read_csv(results_path, index_col=[0, 1])
    doy_start_vec = prob_df.index.get_level_values("initial_case_day_of_year").unique()
    t_calc_vec = prob_df.index.get_level_values("day_of_outbreak").unique()

    fig, ax = plt.subplots()
    if is_nontrivial_outbreak:
        plot_incidence_on_twin_ax(
            ax, t_vec=np.arange(len(incidence_vec)), incidence=incidence_vec
        )
    for doy_start, color in zip(
        doy_start_vec,
        ["tab:blue", "tab:orange"],
        strict=True,
    ):
        initial_case_date = pd.Timestamp(year=2017, month=1, day=1) + pd.Timedelta(
            days=doy_start - 1
        )
        prob_vec = prob_df.loc[doy_start, "analytical"].to_numpy()
        ax.plot(
            t_calc_vec,
            prob_vec,
            label=f"Initial case on {initial_case_date.day} {initial_case_date:%b}",
            color=color,
        )
        prob_sim_vec = prob_df.loc[doy_start, "simulation"].to_numpy()
        ax.plot(t_calc_vec, prob_sim_vec, ".", color=color)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel(
        "Day of outbreak" if is_nontrivial_outbreak else "Days since initial case"
    )
    if not is_nontrivial_outbreak:
        ax.set_xlim(0, ax.get_xlim()[1])
    ax.set_ylabel("Probability of additional cases")
    ax.legend()
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


def _make_decision_delay_plot(*, results_path, fig_path=None):
    decision_delay_df = pd.read_csv(results_path, index_col=[0, 1])
    risk_threshold_pct_vec = decision_delay_df.index.get_level_values(
        "risk_threshold_pct"
    ).unique()
    doy_final_case_vec = decision_delay_df.index.get_level_values(
        "final_case_day_of_year"
    ).unique()

    fig, ax = plt.subplots()

    for risk_threshold_pct in risk_threshold_pct_vec:
        decision_delay_vec = decision_delay_df.loc[
            risk_threshold_pct, "delay_to_decision"
        ].to_numpy()
        ax.plot(
            doy_final_case_vec,
            decision_delay_vec,
            label=f"{risk_threshold_pct}% risk threshold",
        )
    month_start_xticks(ax, interval_months=2)
    ax.set_xlabel("Date of last observed case")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
    ax.legend()
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


def _make_many_outbreak_example_plot(
    *,
    risk_threshold_pct_vals,
    cmap_names=("Blues", "Oranges", "Greens", "Purples"),
    results_path,
    fig_path=None,
):
    outbreak_df = pd.read_csv(results_path)
    fig, ax = plt.subplots()
    plot_incidence_on_twin_ax(
        ax,
        t_vec=outbreak_df["day_of_year"],
        incidence=outbreak_df["incidence"],
    )
    ax.plot(
        outbreak_df["day_of_year"],
        outbreak_df["additional_case_prob"],
        color="black",
    )
    for risk_threshold_pct, cmap_name in zip(
        risk_threshold_pct_vals,
        cmap_names[: len(risk_threshold_pct_vals)],
        strict=True,
    ):
        # Match each threshold line to its series colour in the decision plot.
        ax.axhline(
            risk_threshold_pct / 100,
            color=matplotlib.colormaps[cmap_name](0.7),
            linestyle="--",
        )
    month_start_xticks(ax)
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Probability of additional cases")
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


def _make_many_outbreak_decision_plot(
    *,
    results_path,
    risk_threshold_pct_vals,
    cmap_names=("Blues", "Oranges", "Greens", "Purples"),
    example_marker_color="red",
    example_outbreak_idx=None,
    x_limits=None,
    y_limits=None,
    x_tick_interval_months=1,
    fig_path=None,
):
    results_df = pd.read_csv(results_path)
    bin_width_days = 7
    results_df["doy_final_case_binned"] = pd.cut(
        results_df["final_case_day_of_year"],
        bins=range(1, 366, bin_width_days),
        right=False,
        include_lowest=True,
    )
    # Per-week frequency of outbreaks (shared across thresholds) sets the bar shade.
    outbreak_proportion_series = (
        results_df["doy_final_case_binned"].value_counts(normalize=True).sort_index()
    )
    norm = plt.Normalize(0, 0.01)

    n_thresholds = len(risk_threshold_pct_vals)
    # Offset each threshold's markers horizontally so overlapping error bars separate.
    marker_offset_step_days = 1.75
    marker_offset_vec = (
        np.arange(n_thresholds) - (n_thresholds - 1) / 2
    ) * marker_offset_step_days

    fig, ax = plt.subplots()
    legend_handles = []
    for marker_offset, risk_threshold_pct, cmap_name in zip(
        marker_offset_vec,
        risk_threshold_pct_vals,
        cmap_names[:n_thresholds],
        strict=True,
    ):
        cmap = matplotlib.colormaps[cmap_name]
        delay_column = f"delay_to_decision_{risk_threshold_pct}"
        summary_df = (
            results_df.groupby("doy_final_case_binned", observed=False)[delay_column]
            .quantile(np.array([0.025, 0.5, 0.975]))
            .unstack()
        )
        bin_centre_vec = np.array([interval.mid for interval in summary_df.index])
        color_mat = cmap(norm(outbreak_proportion_series.to_numpy()))
        for bin_centre, median_delay, lower_error, upper_error, color in zip(
            bin_centre_vec + marker_offset,
            summary_df[0.5],
            summary_df[0.5] - summary_df[0.025],
            summary_df[0.975] - summary_df[0.5],
            color_mat,
        ):
            ax.errorbar(
                bin_centre,
                median_delay,
                yerr=[[lower_error], [upper_error]],
                fmt="o",
                color=color,
                capsize=3,
            )
        legend_handles.append(
            Line2D(
                [],
                [],
                marker="o",
                linestyle="none",
                color=cmap(0.7),
                label=f"{risk_threshold_pct}% threshold",
            )
        )
        if example_outbreak_idx is not None:
            example_outbreak_delay = results_df.loc[example_outbreak_idx, delay_column]
            example_outbreak_final_case_bin_centre = (
                results_df.loc[example_outbreak_idx, "doy_final_case_binned"].mid
                + marker_offset
            )
            ax.plot(
                example_outbreak_final_case_bin_centre,
                example_outbreak_delay,
                marker="x",
                color=example_marker_color,
                linewidth=2,
            )
    ax.set_xlabel("Week of last observed case")
    ax.set_ylabel("Days from last observed case\nuntil risk falls below threshold")
    if x_limits is not None:
        ax.set_xlim(*x_limits)
    if y_limits is not None:
        ax.set_ylim(*y_limits)
    month_start_xticks(ax, interval_months=x_tick_interval_months)
    if len(risk_threshold_pct_vals) > 1:
        ax.legend(handles=legend_handles, loc="lower center")
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


if __name__ == "__main__":
    make_plots()
