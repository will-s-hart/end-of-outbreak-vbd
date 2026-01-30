import argparse
import pathlib

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from tqdm import tqdm

from endoutbreakvbd import (
    calc_declaration_delay,
    calc_further_case_risk_analytical,
    calc_further_case_risk_simulation,
    renewal_model,
)
from endoutbreakvbd.chikungunya import get_parameters, get_suitability_data
from endoutbreakvbd.utils import month_start_xticks, plot_data_on_twin_ax


def _get_inputs():
    parameters = get_parameters()
    gen_time_dist_vec = parameters["gen_time_dist_vec"]

    doy_vec = np.arange(1, 366, dtype=int)

    df_suitability = get_suitability_data()
    suitability_vec = df_suitability["suitability_smoothed"].to_numpy()

    rep_no_factor = 2
    rep_no_vec = rep_no_factor * suitability_vec

    def rep_no_func_doy(doy):
        return np.interp(doy, doy_vec, rep_no_vec, period=365)

    def rep_no_func_getter(doy_start):
        def _rep_no_func(t):
            return rep_no_func_doy(doy_start + t)

        return _rep_no_func

    example_outbreak_doy_start_vals = (
        np.nonzero(rep_no_vec > 1.2)[0][0] + 1,
        np.nonzero(rep_no_vec > 1.2)[0][-1] + 1,
    )
    example_outbreak_incidence_vec = [1]
    example_outbreak_perc_risk_threshold_vals = (1, 2.5, 5)

    many_outbreak_no_sims = 100000
    many_outbreak_outbreak_size_threshold = 2
    many_outbreak_perc_risk_threshold = 5

    results_dir = pathlib.Path(__file__).parents[1] / "results/sim_study"
    results_dir.mkdir(parents=True, exist_ok=True)
    results_paths = {
        "example_outbreak_risk": results_dir / "example_outbreak_risk.csv",
        "example_outbreak_declaration": results_dir
        / "example_outbreak_declaration.csv",
        "many_outbreak": results_dir / "many_outbreak.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures/sim_study"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "rep_no": fig_dir / "rep_no.svg",
        "example_outbreak_risk": fig_dir / "example_outbreak_risk.svg",
        "example_outbreak_declaration": fig_dir / "example_outbreak_declaration.svg",
        "many_outbreak": fig_dir / "many_outbreak.svg",
    }

    return {
        "parameters": parameters,
        "gen_time_dist_vec": gen_time_dist_vec,
        "doy_vec": doy_vec,
        "rep_no_factor": rep_no_factor,
        "rep_no_func_doy": rep_no_func_doy,
        "rep_no_func_getter": rep_no_func_getter,
        "example_outbreak_doy_start_vals": example_outbreak_doy_start_vals,
        "example_outbreak_incidence_vec": example_outbreak_incidence_vec,
        "example_outbreak_perc_risk_threshold_vals": example_outbreak_perc_risk_threshold_vals,
        "many_outbreak_no_sims": many_outbreak_no_sims,
        "many_outbreak_outbreak_size_threshold": many_outbreak_outbreak_size_threshold,
        "many_outbreak_perc_risk_threshold": many_outbreak_perc_risk_threshold,
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def run_analyses():
    inputs = _get_inputs()
    rng = np.random.default_rng(2)
    _run_example_outbreak_risk_analysis(
        incidence_vec=inputs["example_outbreak_incidence_vec"],
        rep_no_func_getter=inputs["rep_no_func_getter"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        doy_start_vals=inputs["example_outbreak_doy_start_vals"],
        rng=rng,
        save_path=inputs["results_paths"]["example_outbreak_risk"],
    )
    _run_example_outbreak_declaration_analysis(
        incidence_vec=inputs["example_outbreak_incidence_vec"],
        perc_risk_threshold_vals=inputs["example_outbreak_perc_risk_threshold_vals"],
        rep_no_func_getter=inputs["rep_no_func_getter"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        save_path=inputs["results_paths"]["example_outbreak_declaration"],
    )
    _run_many_outbreak_analysis(
        no_sims=inputs["many_outbreak_no_sims"],
        outbreak_size_threshold=inputs["many_outbreak_outbreak_size_threshold"],
        perc_risk_threshold=inputs["many_outbreak_perc_risk_threshold"],
        rep_no_func_getter=inputs["rep_no_func_getter"],
        gen_time_dist_vec=inputs["gen_time_dist_vec"],
        rng=rng,
        save_path=inputs["results_paths"]["many_outbreak"],
    )


def make_plots():
    inputs = _get_inputs()
    _make_rep_no_plot(
        rep_no_func_doy=inputs["rep_no_func_doy"],
        doy_vec=inputs["doy_vec"],
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
    _make_many_outbreak_plot(
        data_path=inputs["results_paths"]["many_outbreak"],
        save_path=inputs["fig_paths"]["many_outbreak"],
    )


def _run_example_outbreak_risk_analysis(
    *,
    incidence_vec,
    rep_no_func_getter,
    gen_time_dist_vec,
    doy_start_vals,
    rng,
    save_path,
):
    risk_days = np.arange(1, len(incidence_vec) + len(gen_time_dist_vec) + 1, dtype=int)
    risk_days_sim = np.arange(
        1, len(incidence_vec) + len(gen_time_dist_vec) + 1, dtype=int
    )
    risk_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [doy_start_vals, risk_days], names=["start_day_of_year", "day_of_outbreak"]
        )
    )

    for doy_start in doy_start_vals:
        rep_no_func = rep_no_func_getter(doy_start)

        risk_vals = calc_further_case_risk_analytical(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            gen_time_dist_vec=gen_time_dist_vec,
            t_calc=risk_days,
        )
        risk_vals_sim = calc_further_case_risk_simulation(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            gen_time_dist_vec=gen_time_dist_vec,
            t_calc=risk_days_sim,
            n_sims=1000,
            rng=rng,
        )
        risk_df.loc[(doy_start, risk_days), "analytical"] = risk_vals
        risk_df.loc[(doy_start, risk_days_sim), "simulation"] = risk_vals_sim
    risk_df.to_csv(save_path)


def _run_example_outbreak_declaration_analysis(
    *,
    incidence_vec,
    perc_risk_threshold_vals,
    rep_no_func_getter,
    gen_time_dist_vec,
    save_path,
):
    time_last_case = np.nonzero(incidence_vec)[0][-1]
    risk_days = np.arange(
        time_last_case + 1, time_last_case + len(gen_time_dist_vec) + 2, dtype=int
    )

    doy_last_case_vec = np.arange(1, 366, dtype=int)
    declaration_delay_df = pd.DataFrame(
        index=pd.MultiIndex.from_product(
            [perc_risk_threshold_vals, doy_last_case_vec],
            names=["perc_risk_threshold", "final_case_day_of_year"],
        ),
        columns=["delay_to_declaration"],
    )

    for doy_last_case in doy_last_case_vec:
        doy_start = doy_last_case - len(incidence_vec) + 1
        rep_no_func = rep_no_func_getter(doy_start)
        risk_vals = calc_further_case_risk_analytical(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            gen_time_dist_vec=gen_time_dist_vec,
            t_calc=risk_days,
        )
        declaration_delay_vals = calc_declaration_delay(
            risk_vec=risk_vals,
            perc_risk_threshold=perc_risk_threshold_vals,
            delay_of_first_risk=1,
        )
        declaration_delay_df.loc[
            (list(perc_risk_threshold_vals), doy_last_case), "delay_to_declaration"
        ] = declaration_delay_vals
    declaration_delay_df.to_csv(save_path)


def _run_many_outbreak_analysis(
    *,
    no_sims,
    outbreak_size_threshold,
    perc_risk_threshold,
    rep_no_func_getter,
    gen_time_dist_vec,
    rng,
    save_path,
):
    df = pd.DataFrame(
        index=range(no_sims),
        columns=[
            "first_case_day_of_year",
            "final_case_day_of_year",
            "delay_to_declaration",
        ],
    )
    for sim_idx in tqdm(range(no_sims)):
        outbreak_found = False
        while not outbreak_found:
            doy_start = rng.integers(1, 366)
            rep_no_func = rep_no_func_getter(doy_start)
            incidence_vec = renewal_model(
                rep_no_func=rep_no_func,
                t_stop=1000,
                gen_time_dist_vec=gen_time_dist_vec,
                rng=rng,
                incidence_init=1,
            )
            if np.sum(incidence_vec) >= outbreak_size_threshold:
                outbreak_found = True
        time_last_case = np.nonzero(incidence_vec)[0][-1]
        doy_last_case = doy_start + time_last_case
        risk_days = np.arange(
            time_last_case + 1, time_last_case + len(gen_time_dist_vec) + 2, dtype=int
        )
        risk_vals = calc_further_case_risk_analytical(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            gen_time_dist_vec=gen_time_dist_vec,
            t_calc=risk_days,
        )
        declaration_delay = calc_declaration_delay(
            risk_vec=risk_vals,
            perc_risk_threshold=perc_risk_threshold,
            delay_of_first_risk=1,
        )
        df.loc[sim_idx, "first_case_day_of_year"] = doy_start
        df.loc[sim_idx, "final_case_day_of_year"] = doy_last_case
        df.loc[sim_idx, "delay_to_declaration"] = declaration_delay
        if 150 < doy_last_case < 250 and declaration_delay == 0:
            print(
                "Possible error - zero days to declaration for outbreak ending mid-year"
            )
    df.to_csv(save_path)


def _make_rep_no_plot(*, rep_no_func_doy, doy_vec, example_doy_vals, save_path):
    fig, ax = plt.subplots()
    ax.plot(doy_vec, rep_no_func_doy(doy_vec))
    ax.plot(example_doy_vals, rep_no_func_doy(np.array(example_doy_vals)), "o")
    month_start_xticks(ax)
    ax.set_ylabel("Time-dependent reproduction number")
    fig.savefig(save_path)


def _make_example_outbreak_risk_plot(*, incidence_vec, data_path, save_path):
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
    ax.set_ylim(0, 1)
    ax.set_xlabel("Day of outbreak")
    ax.set_ylabel("Risk of additional cases")
    ax.legend()
    fig.savefig(save_path)


def _make_example_outbreak_declaration_plot(*, data_path, save_path):
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
    month_start_xticks(ax)
    ax.set_xlabel("Date of final case")
    ax.set_ylabel("Days from final case until declaration of end of outbreak")
    ax.legend()
    fig.savefig(save_path)


def _make_many_outbreak_plot(*, data_path, save_path):
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
        .quantile([0.025, 0.5, 0.975])
        .unstack()
    )
    proportions = df["final_case_doy_binned"].value_counts(normalize=True).sort_index()
    bin_centers = [interval.mid for interval in stats.index]

    cmap = matplotlib.colormaps["Blues"]
    # norm = matplotlib.colors.LogNorm(0.001, proportions.max())
    norm = plt.Normalize(0, 0.01)
    colors = cmap(norm(proportions.values))

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

    sm = plt.cm.ScalarMappable(norm=norm, cmap=cmap)
    sm.set_array([])
    cbar = plt.colorbar(sm, ax=ax)
    cbar.set_ticks(np.linspace(0, norm.vmax, 11))
    ax.set_xlabel("Week of final case")
    ax.set_ylabel("Days from final case until declaration")
    ax.set_xlim(121, 305)
    month_start_xticks(ax, interval_months=1)
    fig.savefig(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    parser.add_argument(
        "-p",
        "--plots-only",
        action="store_true",
        help="Only generate plots (using saved results)",
    )
    args = parser.parse_args()
    if not args.plots_only:
        run_analyses()
    if not args.results_only:
        make_plots()
