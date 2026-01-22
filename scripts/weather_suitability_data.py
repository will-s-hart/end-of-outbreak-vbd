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
    results_dir = pathlib.Path(__file__).parents[1] / "results"
    results_paths = {
        "all": results_dir / "weather_suitability_data_all.csv",
        "2017": results_dir / "weather_suitability_data_2017.csv",
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
    # Define temperature-suitability mapping
    temperature_grid = np.linspace(10, 40, 61)
    suitability_grid = (
        np.array(
            [
                0.00000000e00,
                0.00000000e00,
                1.88203984e-08,
                1.28367618e-07,
                5.36961844e-07,
                2.82007101e-06,
                1.28234033e-05,
                5.25931083e-05,
                1.82882567e-04,
                5.50222548e-04,
                1.48801416e-03,
                3.62102778e-03,
                7.92083800e-03,
                1.57431781e-02,
                2.88100329e-02,
                4.86355377e-02,
                7.66192219e-02,
                1.13479057e-01,
                1.59883423e-01,
                2.15867501e-01,
                2.80759580e-01,
                3.53556862e-01,
                4.32786966e-01,
                5.16530769e-01,
                6.02553457e-01,
                6.88071907e-01,
                7.69834160e-01,
                8.44406132e-01,
                9.08117700e-01,
                9.57439287e-01,
                9.89023789e-01,
                1.00000000e00,
                9.88217659e-01,
                9.52537702e-01,
                8.92959231e-01,
                8.10907665e-01,
                7.09457680e-01,
                5.93673840e-01,
                4.69879650e-01,
                3.45600684e-01,
                2.29671637e-01,
                1.32825400e-01,
                6.50191949e-02,
                2.73829517e-02,
                1.02714042e-02,
                3.48809148e-03,
                1.03983518e-03,
                3.27940090e-04,
                7.74962840e-05,
                1.17503236e-05,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
                0.00000000e00,
            ]
        )
        ** 2
    )
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
