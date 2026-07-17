import matplotlib.pyplot as plt
import pandas as pd

from endoutbreakvbd.utils import month_start_xticks, set_plot_config
from scripts.inputs import get_inputs_weather_suitability_data


def make_plots():
    set_plot_config()
    inputs = get_inputs_weather_suitability_data()
    _make_temperature_plot(
        all_results_path=inputs["results_paths"]["all"],
        fig_path=inputs["fig_paths"]["temperature"],
    )
    _make_suitability_model_plot(
        suitability_grid_df=inputs["suitability_grid_df"],
        fig_path=inputs["fig_paths"]["suitability_model"],
    )
    _make_suitability_plot(
        results_2017_path=inputs["results_paths"]["2017"],
        fig_path=inputs["fig_paths"]["suitability"],
    )


def _make_temperature_plot(*, all_results_path, fig_path):
    all_results_df = pd.read_csv(all_results_path, index_col="date", parse_dates=True)
    all_date_index = pd.DatetimeIndex(all_results_df.index)
    # Split into 2017 and other years
    results_2017_df = all_results_df.loc["2017"]
    other_years_df = all_results_df.loc[all_date_index.year != 2017]
    date_2017_index = pd.DatetimeIndex(results_2017_df.index)
    other_date_index = pd.DatetimeIndex(other_years_df.index)
    fig, ax = plt.subplots()
    ax.scatter(
        other_date_index.dayofyear,
        other_years_df["temperature"],
        color="tab:blue",
        alpha=0.3,
        s=5,
    )
    ax.scatter(
        date_2017_index.dayofyear,
        results_2017_df["temperature"],
        color="tab:orange",
        s=10,
    )
    ax.plot(
        date_2017_index.dayofyear,
        results_2017_df["temperature_smoothed"],
        color="tab:red",
    )
    month_start_xticks(ax, interval_months=2)
    ax.set_xlabel("Date")
    ax.set_ylabel("Temperature (°C)")
    fig.savefig(fig_path)


def _make_suitability_model_plot(*, suitability_grid_df, fig_path):
    fig, ax = plt.subplots()
    ax.plot(suitability_grid_df["temperature"], suitability_grid_df["suitability"])
    ax.set_xlim(10, 40)
    ax.set_ylim(0, 1.01)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Suitability for transmission")
    fig.savefig(fig_path)


def _make_suitability_plot(*, results_2017_path, fig_path):
    results_2017_df = pd.read_csv(results_2017_path, index_col="date", parse_dates=True)
    date_2017_index = pd.DatetimeIndex(results_2017_df.index)
    fig, ax = plt.subplots()
    ax.plot(
        date_2017_index.dayofyear,
        results_2017_df["suitability_smoothed_lagged"],
    )
    month_start_xticks(ax, interval_months=2)
    ax.set_xlabel("Date")
    ax.set_ylim(0, 1.01)
    ax.set_ylabel("Temperature suitability for transmission")
    fig.savefig(fig_path)


if __name__ == "__main__":
    make_plots()
