import argparse
from datetime import datetime

import meteostat
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.deterministic import DeterministicProcess

from endoutbreakvbd.inputs import get_inputs_weather_suitability_data
from scripts.weather_suitability_data_plots import make_plots


def run_analyses():
    inputs = get_inputs_weather_suitability_data()
    _get_process_data(
        df_suitability_grid=inputs["df_suitability_grid"],
        save_path_all=inputs["results_paths"]["all"],
        save_path_2017=inputs["results_paths"]["2017"],
    )


def _get_process_data(*, df_suitability_grid, save_path_all, save_path_2017):
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
    temperature_grid = df_suitability_grid["temperature"].to_numpy(dtype=float)
    suitability_grid = df_suitability_grid["suitability"].to_numpy(dtype=float)
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


if __name__ == "__main__":
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    args = argparser.parse_args()
    run_analyses()
    if not args.results_only:
        make_plots()
