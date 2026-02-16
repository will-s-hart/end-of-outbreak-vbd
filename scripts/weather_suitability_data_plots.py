import matplotlib.pyplot as plt
import pandas as pd

from endoutbreakvbd.inputs import get_inputs_weather_suitability_data
from endoutbreakvbd.utils import month_start_xticks, set_plot_config


def make_plots():
    set_plot_config()
    inputs = get_inputs_weather_suitability_data()
    _make_temperature_plot(
        data_path_all=inputs["results_paths"]["all"],
        save_path=inputs["fig_paths"]["temperature"],
    )
    _make_suitability_model_plot(
        df_suitability_grid=inputs["df_suitability_grid"],
        save_path=inputs["fig_paths"]["suitability_model"],
    )
    _make_suitability_plot(
        data_path_2017=inputs["results_paths"]["2017"],
        save_path=inputs["fig_paths"]["suitability"],
    )


def _make_temperature_plot(*, data_path_all, save_path):
    df_all = pd.read_csv(data_path_all, index_col="date", parse_dates=True)
    df_all_index = pd.DatetimeIndex(df_all.index)
    # Split into 2017 and other years
    df_2017 = df_all.loc["2017"]
    df_other = df_all.loc[df_all_index.year != 2017]
    df_2017_index = pd.DatetimeIndex(df_2017.index)
    df_other_index = pd.DatetimeIndex(df_other.index)
    fig, ax = plt.subplots()
    ax.scatter(
        df_other_index.dayofyear,
        df_other["temperature"],
        color="tab:blue",
        alpha=0.3,
        s=5,
    )
    ax.scatter(
        df_2017_index.dayofyear, df_2017["temperature"], color="tab:orange", s=10
    )
    ax.plot(
        df_2017_index.dayofyear,
        df_2017["temperature_smoothed"],
        color="tab:red",
    )
    month_start_xticks(ax, interval_months=2)
    ax.set_ylabel("Temperature (°C)")
    fig.savefig(save_path)


def _make_suitability_model_plot(*, df_suitability_grid, save_path):
    fig, ax = plt.subplots()
    ax.plot(df_suitability_grid["temperature"], df_suitability_grid["suitability"])
    ax.set_xlim(10, 40)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Temperature (°C)")
    ax.set_ylabel("Suitability for transmission")
    fig.savefig(save_path)


def _make_suitability_plot(*, data_path_2017, save_path):
    df_2017 = pd.read_csv(data_path_2017, index_col="date", parse_dates=True)
    df_2017_index = pd.DatetimeIndex(df_2017.index)
    fig, ax = plt.subplots()
    ax.plot(df_2017_index.dayofyear, df_2017["suitability_smoothed"])
    month_start_xticks(ax, interval_months=2)
    ax.set_ylim(0, 1)
    ax.set_ylabel("Temperature suitability for transmission")
    fig.savefig(save_path)


if __name__ == "__main__":
    make_plots()
