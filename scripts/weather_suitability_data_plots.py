import matplotlib.pyplot as plt
import pandas as pd

from endoutbreakvbd.inputs import get_inputs_weather_suitability_data
from endoutbreakvbd.utils import month_start_xticks


def make_plots():
    inputs = get_inputs_weather_suitability_data()
    _make_temperature_plot(
        data_path_all=inputs["results_paths"]["all"],
        save_path=inputs["fig_paths"]["temperature"],
    )
    _make_suitability_plot(
        data_path_2017=inputs["results_paths"]["2017"],
        save_path=inputs["fig_paths"]["suitability"],
    )


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
    make_plots()
