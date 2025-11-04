import pandas as pd


def month_start_xticks(ax, year=2017, interval_months=2):
    month_starts = pd.date_range(
        start=f"{year}-01-01", end=f"{year + 1}-01-01", freq=f"{interval_months}MS"
    )
    month_starts_doy = month_starts.dayofyear.to_numpy()
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
