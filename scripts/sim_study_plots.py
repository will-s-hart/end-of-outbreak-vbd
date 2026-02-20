import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from endoutbreakvbd.inputs import get_inputs_sim_study
from endoutbreakvbd.utils import (
    month_start_xticks,
    plot_data_on_twin_ax,
    set_plot_config,
)


def make_plots():
    set_plot_config()
    inputs = get_inputs_sim_study()
    _make_rep_no_plot(
        rep_no_func_doy=inputs["rep_no_func_doy"],
        example_doy_vals=inputs["example_outbreak_doy_start_vals"],
        save_path=inputs["fig_paths"]["rep_no"],
    )
    _make_example_outbreak_risk_plot(
        incidence_vec=inputs["example_outbreak_incidence_vec"],
        data_path=inputs["results_paths"]["example_outbreak_risk"],
        save_path=inputs["fig_paths"]["example_outbreak_risk"],
    )
    _make_example_outbreak_declaration_plot(
        data_path=inputs["results_paths"]["example_outbreak_declaration"],
        save_path=inputs["fig_paths"]["example_outbreak_declaration"],
    )
    _make_many_outbreak_example_plot(
        perc_risk_threshold=inputs["many_outbreak_perc_risk_threshold"],
        data_path=inputs["results_paths"]["many_outbreak_example"],
        save_path=inputs["fig_paths"]["many_outbreak_example"],
    )
    _make_many_outbreak_declaration_plot(
        example_outbreak_idx=inputs["many_outbreak_example_outbreak_idx"],
        data_path=inputs["results_paths"]["many_outbreak_declaration"],
        save_path=inputs["fig_paths"]["many_outbreak_declaration"],
    )


def _make_rep_no_plot(*, rep_no_func_doy, example_doy_vals, save_path=None):
    fig, ax = plt.subplots()
    doy_vec = np.arange(1, 366)
    ax.plot(doy_vec, rep_no_func_doy(doy_vec))
    ax.plot(example_doy_vals, rep_no_func_doy(np.array(example_doy_vals)), "o")
    month_start_xticks(ax, interval_months=2)
    ax.set_ylim(0, 2.02)
    ax.set_ylabel("Time-dependent reproduction number")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_example_outbreak_risk_plot(*, incidence_vec, data_path, save_path=None):
    risk_df = pd.read_csv(data_path, index_col=[0, 1])
    doy_start_vals = risk_df.index.get_level_values("start_day_of_year").unique()
    risk_days = risk_df.index.get_level_values("day_of_outbreak").unique()

    fig, ax = plt.subplots()
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
        risk_vals = risk_df.loc[doy_start, "analytical"].to_numpy()
        ax.plot(
            risk_days,
            risk_vals,
            label=f"First case on {date_start.day} {date_start:%b}",
            color=color,
        )
        risk_vals_sim = risk_df.loc[doy_start, "simulation"].to_numpy()
        ax.plot(risk_days, risk_vals_sim, ".", color=color)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Day of outbreak")
    ax.set_ylabel("Risk of additional cases")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_example_outbreak_declaration_plot(*, data_path, save_path=None):
    declaration_delay_df = pd.read_csv(data_path, index_col=[0, 1])
    perc_risk_threshold_vals = declaration_delay_df.index.get_level_values(
        "perc_risk_threshold"
    ).unique()
    doy_last_case_vec = declaration_delay_df.index.get_level_values(
        "final_case_day_of_year"
    ).unique()

    fig, ax = plt.subplots()

    for perc_risk_threshold in perc_risk_threshold_vals:
        declaration_delay_vec = declaration_delay_df.loc[
            perc_risk_threshold, "delay_to_declaration"
        ].to_numpy()
        ax.plot(
            doy_last_case_vec,
            declaration_delay_vec,
            label=f"{perc_risk_threshold}% risk threshold",
        )
    month_start_xticks(ax, interval_months=2)
    ax.set_xlabel("Date of final case")
    ax.set_ylabel("Days from final case until declaration")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_many_outbreak_example_plot(*, perc_risk_threshold, data_path, save_path=None):
    df = pd.read_csv(data_path)
    fig, ax = plt.subplots()
    plot_data_on_twin_ax(ax, t_vec=df["day_of_year"], incidence_vec=df["cases"])
    ax.plot(df["day_of_year"], df["further_case_risk"], color="tab:blue")
    ax.axhline(perc_risk_threshold / 100, color="tab:red", linestyle="--")
    month_start_xticks(ax)
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Risk of additional cases")
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


def _make_many_outbreak_declaration_plot(
    *, example_outbreak_idx, data_path, save_path=None
):
    df = pd.read_csv(data_path)
    bin_width = 7
    df["final_case_doy_binned"] = pd.cut(
        df["final_case_day_of_year"],
        bins=range(1, 366, bin_width),
        right=False,
        include_lowest=True,
    )
    stats = (
        df.groupby("final_case_doy_binned", observed=False)["delay_to_declaration"]
        .quantile(np.array([0.025, 0.5, 0.975]))
        .unstack()
    )
    proportions = df["final_case_doy_binned"].value_counts(normalize=True).sort_index()
    bin_centers = [interval.mid for interval in stats.index]

    cmap = matplotlib.colormaps["Blues"]
    # norm = matplotlib.colors.LogNorm(0.001, proportions.max())
    norm = plt.Normalize(0, 0.01)
    colors = cmap(norm(proportions.to_numpy()))

    fig, ax = plt.subplots()
    for x, m, lo, hi, c in zip(
        bin_centers,
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
    example_outbreak_delay = df.loc[example_outbreak_idx, "delay_to_declaration"]
    example_outbreak_final_case_bin_center = df.loc[
        example_outbreak_idx, "final_case_doy_binned"
    ].mid
    ax.plot(
        example_outbreak_final_case_bin_center,
        example_outbreak_delay,
        marker="x",
        color="tab:red",
        linewidth=2,
    )
    # sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    # sm.set_array([])
    # cbar = plt.colorbar(sm, ax=ax)
    # cbar.set_ticks(np.linspace(0, norm.vmax, 11))
    ax.set_xlabel("Week of final case")
    ax.set_ylabel("Days from final case until declaration")
    # ax.set_xlim(121, 305)
    month_start_xticks(ax)
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


if __name__ == "__main__":
    make_plots()
