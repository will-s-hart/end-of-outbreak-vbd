import argparse
from datetime import datetime

import meteostat
import numpy as np
import pandas as pd
from statsmodels.regression.linear_model import OLS
from statsmodels.tsa.deterministic import DeterministicProcess

from scripts.inputs import get_inputs_weather_suitability_data
from scripts.weather_suitability_data_plots import make_plots


def run_analyses():
    inputs = get_inputs_weather_suitability_data()
    _process_weather_suitability_data(
        suitability_grid_df=inputs["suitability_grid_df"],
        suitability_lag_days=inputs["suitability_lag_days"],
        all_results_path=inputs["results_paths"]["all"],
        results_2017_path=inputs["results_paths"]["2017"],
    )


def _process_weather_suitability_data(
    *, suitability_grid_df, suitability_lag_days, all_results_path, results_2017_path
):
    # Retrieve temperature data
    observed_temperature_df = (
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
    full_temperature_df = pd.DataFrame(
        {"temperature": np.nan},
        index=pd.date_range("2010-01-01", "2024-12-31"),
    ).rename_axis("date")
    full_temperature_df.loc[observed_temperature_df.index, "temperature"] = (
        observed_temperature_df["temperature"]
    )
    # Fit seasonal model to temperature data
    deterministic_process = DeterministicProcess(
        index=full_temperature_df.index,
        constant=True,
        fourier=2,
        period=365.25,
        drop=True,
    )
    full_design_df = deterministic_process.in_sample()
    full_temperature_series = full_temperature_df["temperature"]
    observed_mask = full_temperature_series.notna()
    training_design_df = full_design_df.loc[observed_mask]
    training_temperature_series = full_temperature_series.loc[observed_mask]
    temperature_model = OLS(training_temperature_series, training_design_df).fit()
    smoothed_temperature_df = pd.DataFrame(
        {"temperature": temperature_model.predict(full_design_df)},
        index=full_temperature_df.index,
    )
    # Define temperature-suitability mapping
    temperature_grid = suitability_grid_df["temperature"].to_numpy(dtype=float)
    suitability_grid = suitability_grid_df["suitability"].to_numpy(dtype=float)
    # Compute suitability for and save to CSV
    all_results_df = full_temperature_df.assign(
        temperature_smoothed=smoothed_temperature_df["temperature"]
    )
    all_results_df = all_results_df.assign(
        suitability_instantaneous=np.interp(
            all_results_df["temperature"], temperature_grid, suitability_grid
        ),
        suitability_smoothed_instantaneous=np.interp(
            all_results_df["temperature_smoothed"], temperature_grid, suitability_grid
        ),
    )
    all_results_df = all_results_df.assign(
        suitability_smoothed_lagged=all_results_df[
            "suitability_smoothed_instantaneous"
        ].shift(suitability_lag_days)
    )
    results_2017_df = all_results_df.loc["2017"]
    all_results_df.to_csv(all_results_path)
    results_2017_df.to_csv(results_2017_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    args = parser.parse_args()
    run_analyses()
    if not args.results_only:
        make_plots()
