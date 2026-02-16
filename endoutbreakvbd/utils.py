from typing import overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.stats
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import ArrayLike

from endoutbreakvbd.types import FloatArray, IntArray


@overload
def rep_no_from_grid(
    t_vec: int,
    *,
    rep_no_grid: FloatArray,
    periodic: bool,
    doy_start: int | None = None,
) -> float | FloatArray: ...


@overload
def rep_no_from_grid(
    t_vec: IntArray,
    *,
    rep_no_grid: FloatArray,
    periodic: bool,
    doy_start: int | None = None,
) -> FloatArray: ...


def rep_no_from_grid(
    t_vec: int | IntArray,
    *,
    rep_no_grid: FloatArray,
    periodic: bool,
    doy_start: int | None = None,
) -> float | FloatArray:
    if periodic:
        if doy_start is None:
            raise ValueError("doy_start should be provided when periodic is True")
        if len(rep_no_grid) != 365:
            raise ValueError(
                "For periodic interpolation, rep_no_grid must have length 365"
            )
        t_vec_grid_idx = (t_vec + (doy_start - 1)) % 365
    else:
        if np.any(t_vec < 0) or np.any(t_vec >= len(rep_no_grid)):
            raise ValueError("t_vec contains indices outside the range of rep_no_grid")
        t_vec_grid_idx = t_vec
    rep_no = rep_no_grid[t_vec_grid_idx, ...]
    if rep_no.size == 1:
        return float(rep_no.item())
    return rep_no


def lognormal_params_from_median_percentile_2_5(
    *, median: float, percentile_2_5: float
) -> dict[str, float]:
    mu = np.log(median)
    sigma = (mu - np.log(percentile_2_5)) / scipy.stats.norm.ppf(0.975)
    return {"mu": mu, "sigma": sigma}


def set_plot_config() -> None:
    rc_params = {
        "figure.figsize": (7, 7),
        "axes.spines.top": False,
        "axes.spines.right": False,
        "savefig.transparent": True,
        "savefig.format": "svg",
        "svg.fonttype": "none",
        "axes.titlesize": 16,
        "axes.labelsize": 16,
        "lines.linewidth": 2,
        "lines.markersize": 8,
        "xtick.labelsize": 16,
        "ytick.labelsize": 16,
        "legend.fontsize": 14,
        "font.family": "sans-serif",
        "font.sans-serif": "Arial",
    }
    plt.rcParams.update(rc_params)
    sns.set_palette("colorblind")


def month_start_xticks(ax: Axes, year: int = 2017, interval_months: int = 1) -> None:
    month_starts = pd.date_range(
        start=f"{year}-01-01", end=f"{year + 1}-01-01", freq="MS"
    )
    month_starts_doy = month_starts.dayofyear.to_numpy(copy=True)
    month_starts_doy[-1] = 366
    xlim = ax.get_xlim()
    month_start_in_range = (month_starts_doy >= xlim[0]) & (
        month_starts_doy <= xlim[-1]
    )
    month_starts = month_starts[month_start_in_range]
    month_starts_doy = month_starts_doy[month_start_in_range]
    ax.set_xticks(month_starts_doy)
    labels = [
        f"{d.day} {d:%b}" if (i % interval_months == 0) else ""
        for i, d in enumerate(month_starts)
    ]
    ax.set_xticklabels(labels, rotation=0)


def plot_data_on_twin_ax(ax: Axes, t_vec: ArrayLike, incidence_vec: ArrayLike) -> Axes:
    twin_ax = ax.twinx()
    twin_ax.bar(t_vec, incidence_vec, color="tab:gray", alpha=0.5)
    twin_ax.set_ylim(0, np.max(incidence_vec))
    twin_ax.set_ylabel("Number of cases")
    twin_ax.yaxis.label.set_color("tab:gray")
    twin_ax.tick_params(axis="y", colors="tab:gray")
    twin_ax.spines["right"].set_visible(True)
    twin_ax.spines["right"].set_color("tab:gray")
    twin_ax.spines["left"].set_visible(False)
    twin_ax.spines["bottom"].set_visible(False)
    return twin_ax


def get_colors() -> list[str]:
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]
