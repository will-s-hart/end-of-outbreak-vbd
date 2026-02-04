import numpy as np
import pandas as pd
import scipy.stats


def rep_no_from_grid(
    t_vec,
    *,
    rep_no_grid,
    periodic,
    doy_start=None,
):
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
    return rep_no_grid[t_vec_grid_idx, ...]


def lognormal_params_from_median_percentile_2_5(*, median, percentile_2_5):
    mu = np.log(median)
    sigma = (mu - np.log(percentile_2_5)) / scipy.stats.norm.ppf(0.975)
    return {"mu": mu, "sigma": sigma}


def month_start_xticks(ax, year=2017, interval_months=2):
    month_starts = pd.date_range(
        start=f"{year}-01-01", end=f"{year + 1}-01-01", freq=f"{interval_months}MS"
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
    labels = [f"{d.day} {d:%b}" for d in month_starts]
    ax.set_xticklabels(labels, rotation=0)


def plot_data_on_twin_ax(ax, t_vec, incidence_vec):
    twin_ax = ax.twinx()
    twin_ax.bar(t_vec, incidence_vec, color="tab:gray", alpha=0.5)
    twin_ax.set_ylim(0, np.max(incidence_vec))
    twin_ax.set_ylabel("Number of cases")
    twin_ax.yaxis.label.set_color("tab:gray")
    twin_ax.tick_params(axis="y", colors="tab:gray")
    return twin_ax
