import argparse
import pathlib
from datetime import datetime

import matplotlib.pyplot as plt
import meteostat
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.deterministic import DeterministicProcess

from endoutbreakvbd.utils import month_start_xticks


def _get_inputs():
    results_dir = pathlib.Path(__file__).parents[1] / "results/weather_suitability_data"
    results_paths = {
        "all": results_dir / "all.csv",
        "2017": results_dir / "2017.csv",
    }

    fig_dir = pathlib.Path(__file__).parents[1] / "figures/weather_suitability_data"
    fig_dir.mkdir(parents=True, exist_ok=True)
    fig_paths = {
        "temperature": fig_dir / "temperature.svg",
        "suitability": fig_dir / "suitability.svg",
    }

    return {
        "results_paths": results_paths,
        "fig_paths": fig_paths,
    }


def run_analyses():
    inputs = _get_inputs()
    _get_process_data(
        save_path_all=inputs["results_paths"]["all"],
        save_path_2017=inputs["results_paths"]["2017"],
    )


def make_plots():
    inputs = _get_inputs()
    _make_temperature_plot(
        data_path_all=inputs["results_paths"]["all"],
        save_path=inputs["fig_paths"]["temperature"],
    )
    _make_suitability_plot(
        data_path_2017=inputs["results_paths"]["2017"],
        save_path=inputs["fig_paths"]["suitability"],
    )


def _get_process_data(*, save_path_all, save_path_2017):
    # Retrieve temperature data
    df_data = (
        meteostat.Daily(
            loc=[16239],
            start=datetime(2010, 1, 1),
            end=datetime(2024, 12, 31),
        )
        .fetch()[["tavg"]]
        .rename({"tavg": "temperature"}, axis=1)
        .rename_axis("date")
    )
    # Fill in NaNs for missing dates (needed for fitting seasonal model)
    df_full = pd.DataFrame(
        {"temperature": np.nan},
        index=pd.date_range("2010-01-01", "2024-12-31"),
    ).rename_axis("date")
    df_full.loc[df_data.index, "temperature"] = df_data["temperature"]
    # Fit seasonal model to temperature data
    dp = DeterministicProcess(
        index=df_full.index,
        constant=True,
        fourier=2,
        period=365.25,
        drop=True,
    )
    x_full = dp.in_sample()
    y_full = df_full["temperature"]
    obs_mask = y_full.notna()
    x_train = x_full.loc[obs_mask]
    y_train = y_full.loc[obs_mask]
    model = OLS(y_train, x_train).fit()
    df_smoothed = pd.DataFrame(
        {"temperature": model.predict(x_full)}, index=df_full.index
    )
    # Define temperature-suitability mapping (need to square values for full
    # transmission cycle)
    # df_suitability_grid = pd.read_csv(
    #     pathlib.Path(__file__).parents[1] / "data/mordecai_suitability_grid.csv"
    # )
    df_suitability_grid = pd.read_csv(  # from https://doi.org/10.1098/rsif.2025.0707
        pathlib.Path(__file__).parents[1] / "data/tegar_suitability_grid.csv"
    )
    temperature_grid = df_suitability_grid["temperature"].to_numpy(dtype=float)
    suitability_grid = df_suitability_grid["suitability"].to_numpy(dtype=float) ** 2
    # Compute suitability for and save to CSV
    df_out_full = df_full.assign(temperature_smoothed=df_smoothed["temperature"])
    df_out_full = df_out_full.assign(
        suitability=np.interp(
            df_out_full["temperature"], temperature_grid, suitability_grid
        ),
        suitability_smoothed=np.interp(
            df_out_full["temperature_smoothed"], temperature_grid, suitability_grid
        ),
    )
    df_out_2017 = df_out_full.loc["2017"]
    df_out_full.to_csv(save_path_all)
    df_out_2017.to_csv(save_path_2017)


def _make_temperature_plot(*, data_path_all, save_path):
    df_all = pd.read_csv(data_path_all, index_col="date", parse_dates=True)
    # Split into 2017 and other years
    df_2017 = df_all.loc["2017"]
    df_other = df_all.loc[df_all.index.year != 2017]
    fig, ax = plt.subplots()
    ax.scatter(
        df_other.index.dayofyear,
        df_other["temperature"],
        color="tab:blue",
        alpha=0.3,
        s=5,
    )
    ax.scatter(
        df_2017.index.dayofyear, df_2017["temperature"], color="tab:orange", s=10
    )
    ax.plot(
        df_2017.index.dayofyear,
        df_2017["temperature_smoothed"],
        color="tab:red",
    )
    month_start_xticks(ax, interval_months=2)
    ax.set_ylabel("Temperature (°C)")
    fig.savefig(save_path)


def _make_suitability_plot(*, data_path_2017, save_path):
    df_2017 = pd.read_csv(data_path_2017, index_col="date", parse_dates=True)
    fig, ax = plt.subplots()
    ax.plot(df_2017.index.dayofyear, df_2017["suitability_smoothed"])
    month_start_xticks(ax, interval_months=2)
    ax.set_ylabel("Relative reproduction number")
    fig.savefig(save_path)


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    argparser.add_argument(
        "-p",
        "--plots-only",
        action="store_true",
        help="Only generate plots (using saved results)",
    )
    args = argparser.parse_args()
    if not args.plots_only:
        run_analyses()
    if not args.results_only:
        make_plots()
