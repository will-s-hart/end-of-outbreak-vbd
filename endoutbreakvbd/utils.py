import functools
from typing import Any, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.stats
import seaborn as sns
import xarray as xr
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from endoutbreakvbd._types import FloatArray, IntArray


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
    """
    Function for interpolating a reproduction number grid to a vector of times.

    Parameters
    ----------
    t_vec : int or IntArray
        Vector of times (days) for which to interpolate the reproduction number.
    rep_no_grid : FloatArray
        Grid of reproduction numbers to interpolate from.
    periodic : bool
        Whether to treat the grid as periodic, describing reproduction numbers over a
        year. If True, the grid should have length 365, with the first element
        corresponding to day 1 of the year (January 1st).
    doy_start : int, optional
        Day of year (1-indexed) on which the outbreak starts (so that t_vec=0
        corresponds to this day of year). Should be provided if and only if periodic is
        True.

    Returns
    -------
    float or FloatArray
        Interpolated reproduction number(s) corresponding to the input times.
    """
    if periodic:
        if doy_start is None:
            raise ValueError("doy_start should be provided when periodic is True")
        if len(rep_no_grid) != 365:
            raise ValueError(
                "For periodic interpolation, rep_no_grid must have length 365"
            )
        t_vec_grid_idx = (t_vec + (doy_start - 1)) % 365
    else:
        if doy_start is not None:
            raise ValueError("doy_start should not be provided when periodic is False")
        if np.any(t_vec < 0) or np.any(t_vec >= len(rep_no_grid)):
            raise ValueError("t_vec contains indices outside the range of rep_no_grid")
        t_vec_grid_idx = t_vec
    rep_no = rep_no_grid[t_vec_grid_idx, ...]
    if rep_no.size == 1:
        return float(rep_no.item())
    return rep_no


def rescale_rep_no_grid_in_time(
    rep_no_grid: FloatArray, season_centre_doy: int, decay_speed: float
) -> FloatArray:
    """
    Rescale a periodic (year-long) reproduction number grid in time about the centre
    of the transmission season.

    The profile is stretched or compressed in time about ``season_centre_doy`` as
    ``R(d) = R_default(season_centre_doy + decay_speed * (d - season_centre_doy))``.
    More than half a year from the centre falls outside the transmission season, where
    R is held at its grid minimum (winter floor) so that a compressed season does not
    wrap around into a spurious repeated summer.

    Parameters
    ----------
    rep_no_grid : FloatArray
        Periodic grid of reproduction numbers over a year. Must have length 365, with
        the first element corresponding to day 1 of the year (January 1st).
    season_centre_doy : int
        Day of year (1-indexed) about which the seasonal profile is rescaled.
    decay_speed : float
        Factor controlling the speed of the seasonal decline. ``decay_speed > 1``
        compresses the profile (faster decline); ``decay_speed < 1`` stretches it
        (slower decline); ``decay_speed == 1`` returns the grid unchanged.

    Returns
    -------
    FloatArray
        Rescaled reproduction number grid, of length 365.
    """
    if len(rep_no_grid) != 365:
        raise ValueError("rep_no_grid must have length 365")
    centre_idx = season_centre_doy - 1  # grid is 0-indexed: doy d is at index d - 1
    half_year = 365 / 2
    day_idxs = np.arange(365)
    days_from_centre = ((day_idxs - centre_idx + half_year) % 365) - half_year
    sampled_idxs = centre_idx + decay_speed * days_from_centre
    rescaled_grid = np.interp(sampled_idxs, day_idxs, rep_no_grid, period=365)
    in_season = np.abs(sampled_idxs - centre_idx) <= half_year
    return np.where(in_season, rescaled_grid, rep_no_grid.min())


def discretise_cori(
    *, dist_cont: scipy.stats.rv_continuous, max_val: int, allow_zero: bool = False
) -> NDArray[np.float64]:
    """
    Function for discretising a continuous distribution using the method
    described in https://doi.org/10.1093/aje/kwt133 (web appendix 11).

    Parameters
    ----------
    dist_cont : scipy.stats.rv_continuous
        Continuous distribution to discretise.
    max_val : int
        Maximum value (days) of the discretised distribution. Any residual probability
        mass is assigned to this value.
    allow_zero : bool
        Whether the discretised distribution may place mass on zero. If False (the
        default), mass at zero is folded into the value 1 and the returned vector starts
        at 1.

    Returns
    -------
    NDArray[np.float64]
        Probability mass vector of the discretised distribution.
    """

    def _integrand_func(x: float, y: float) -> float:
        # To get probability mass function at time x, need to integrate this expression
        # with respect to y between y=x-1 and and y=x+1
        return (1 - abs(x - y)) * dist_cont.pdf(y)

    if max_val < 0:
        raise ValueError("max_val must be non-negative")
    if not allow_zero and max_val < 1:
        raise ValueError("max_val must be at least 1 when allow_zero is False")

    # Set up vector of x values and pre-allocate vector of probabilities
    x_vec = np.arange(0, max_val + 1, dtype=int)
    p_vec = np.zeros(len(x_vec))
    # Calculate probability mass function at each x value
    for i in range(len(x_vec)):  # pylint: disable=consider-using-enumerate
        x = x_vec[i]
        integrand = functools.partial(_integrand_func, x)
        p_vec[i] = scipy.integrate.quad(
            integrand,
            x - 1 if x > 0 else 1e-12,
            x + 1,
        )[0]
    if not allow_zero:
        # Assign mass from 0 to 1
        x_vec = x_vec[1:]
        p_vec[1] = p_vec[1] + p_vec[0]
        p_vec = p_vec[1:]
    # Assign residual mass to x_max
    p_vec[-1] = p_vec[-1] + 1 - np.sum(p_vec)
    return p_vec


def lognormal_params_from_median_percentile_2_5(
    *, median: float, percentile_2_5: float
) -> dict[str, float]:
    """
    Compute the parameters of a lognormal distribution from its median and 2.5th
    percentile.

    Parameters
    ----------
    median : float
        Median of the lognormal distribution.
    percentile_2_5 : float
        2.5th percentile of the lognormal distribution.

    Returns
    -------
    dict[str, float]
        Dictionary with keys ``"mu"`` and ``"sigma"``, the parameters of the underlying
        normal distribution.
    """
    mu = np.log(median)
    sigma = (mu - np.log(percentile_2_5)) / scipy.stats.norm.ppf(0.975)
    return {"mu": mu, "sigma": sigma}


def posterior_trajectory_frame(
    ds: xr.Dataset,
    *,
    onset_day: ArrayLike,
    date: ArrayLike,
    reported: ArrayLike,
) -> pd.DataFrame:
    """
    Assemble a trajectory table from a suitability-model posterior.

    Collects the reported incidence alongside the posterior mean / 2.5% / 97.5% summaries of
    the latent true cases, reproduction number, suitability, and reproduction-number factor,
    plus the additional-case probability, into a single frame indexed by ``onset_day``.

    Parameters
    ----------
    ds : xr.Dataset
        Suitability-model posterior carrying the ``*_mean/_lower/_upper`` summaries and
        ``additional_case_prob``.
    onset_day : ArrayLike
        Day-of-outbreak index for each row.
    date : ArrayLike
        Calendar date for each row.
    reported : ArrayLike
        Reported cases by onset day.

    Returns
    -------
    pd.DataFrame
        Trajectory table indexed by ``onset_day``.
    """
    return pd.DataFrame(
        {
            "onset_day": onset_day,
            "date": date,
            "reported": reported,
            "cases_mean": ds["cases_mean"].values,
            "cases_lower": ds["cases_lower"].values,
            "cases_upper": ds["cases_upper"].values,
            "reproduction_number_mean": ds["rep_no_mean"].values,
            "reproduction_number_lower": ds["rep_no_lower"].values,
            "reproduction_number_upper": ds["rep_no_upper"].values,
            "suitability_mean": ds["suitability_mean"].values,
            "suitability_lower": ds["suitability_lower"].values,
            "suitability_upper": ds["suitability_upper"].values,
            "rep_no_factor_mean": ds["rep_no_factor_mean"].values,
            "rep_no_factor_lower": ds["rep_no_factor_lower"].values,
            "rep_no_factor_upper": ds["rep_no_factor_upper"].values,
            "additional_case_prob": ds["additional_case_prob"].values,
        }
    ).set_index("onset_day")


def set_plot_config() -> None:
    """
    Apply the project's shared matplotlib and seaborn plotting configuration (figure
    size, fonts, colour palette, and SVG export settings).
    """
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
    """
    Set x-axis ticks at the start of each month, labelled by day and month.

    Ticks are restricted to the current x-limits of the axes.

    Parameters
    ----------
    ax : Axes
        Axes on which to set the ticks.
    year : int
        Year used to compute month-start days of year (controls leap-year handling).
    interval_months : int
        Label every ``interval_months``th month start; intermediate ticks are left
        unlabelled.
    """
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


def dates_to_day_index(
    dates: pd.Series | pd.DatetimeIndex, year: int = 2017
) -> NDArray[np.int64]:
    """
    Convert calendar dates to a continuous day index anchored at the start of ``year``.

    Within ``year`` this equals the day of year (1 January → 1). Unlike a bare day-of-year,
    it does not wrap when a window crosses into the next year, so it stays monotonic across
    the year boundary and pairs with ``month_start_xticks`` for a continuous calendar x-axis.

    Parameters
    ----------
    dates : pd.Series or pd.DatetimeIndex
        Datetime-valued series or index to convert.
    year : int
        Anchor year; day 1 is 1 January of this year.

    Returns
    -------
    NDArray[np.int64]
        Continuous day index for each date.
    """
    return (pd.DatetimeIndex(dates) - pd.Timestamp(f"{year}-01-01")).days.to_numpy() + 1


def plot_data_on_twin_ax(
    ax: Axes,
    t_vec: ArrayLike,
    bar_heights: ArrayLike,
    *,
    bar_labels: list[str | None] | None = None,
    alpha: float = 0.5,
    fitted: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
    fitted_color: str | None = None,
    fitted_label: str | None = None,
) -> Axes:
    """
    Plot an incidence time series as a bar chart on a twin y-axis.

    The primary axes are raised above the twin axis so the bars sit behind the primary-axis
    artists (curves and legend).

    Parameters
    ----------
    ax : Axes
        Primary axes to attach the twin axis to.
    t_vec : ArrayLike
        Times (days) for the incidence bars.
    bar_heights : ArrayLike
        Number of cases at each time. A single 1-D series draws one bar; a list of series
        draws them stacked (ordered least- to most-hidden), coloured by an internal palette
        (grey base, then orange/purple).
    bar_labels : list[str | None], optional
        Legend labels for the stacked bars (one per series; ``None`` to omit a label). If
        omitted, the bars are unlabelled.
    alpha : float
        Bar transparency.
    fitted : tuple[ArrayLike, ArrayLike, ArrayLike], optional
        ``(mean, lower, upper)`` of a fitted-case series to overlay: the credible band is
        drawn behind the bars and the mean line in front.
    fitted_color : str, optional
        Colour of the ``fitted`` band and mean line.
    fitted_label : str, optional
        Legend label for the ``fitted`` mean line.

    Returns
    -------
    Axes
        The created twin axis.
    """
    layers = _as_bar_layers(bar_heights)
    # The day axis may run one projected day past the incidence bars (a fit reports the
    # current-day risk one day beyond its data); pad the bars with a trailing zero so a longer
    # `t_vec` still aligns (no bar is drawn for the projected day).
    n_t = len(np.asarray(t_vec))
    layers = [
        np.append(layer, np.zeros(n_t - len(layer))) if len(layer) < n_t else layer
        for layer in layers
    ]
    colors = _stacked_bar_colors(len(layers))
    labels = bar_labels if bar_labels is not None else [None] * len(layers)

    twin_ax = ax.twinx()
    if fitted is not None:
        fitted_mean, fitted_lower, fitted_upper = (
            np.asarray(a, dtype=float) for a in fitted
        )
        twin_ax.fill_between(
            t_vec, fitted_lower, fitted_upper, color=fitted_color, alpha=0.25, zorder=1
        )
    total = np.zeros(len(layers[0]), dtype=float)
    for layer, color, label in zip(layers, colors, labels, strict=True):
        twin_ax.bar(
            t_vec, layer, bottom=total, color=color, alpha=alpha, label=label, zorder=2
        )
        total = total + layer
    if fitted is not None:
        twin_ax.plot(
            t_vec, fitted_mean, color=fitted_color, label=fitted_label, zorder=3
        )
        total = np.maximum(total, fitted_upper)
    twin_ax.set_ylim(0, np.max(total))
    twin_ax.set_ylabel("Number of cases")
    twin_ax.yaxis.label.set_color("tab:gray")
    twin_ax.tick_params(axis="y", colors="tab:gray")
    twin_ax.spines["right"].set_visible(True)
    twin_ax.spines["right"].set_color("tab:gray")
    twin_ax.spines["left"].set_visible(False)
    twin_ax.spines["bottom"].set_visible(False)
    # Draw the primary axis (curves, legend) above the incidence bars.
    ax.set_zorder(twin_ax.get_zorder() + 1)
    ax.patch.set_visible(False)
    return twin_ax


def get_colors() -> list[str]:
    """
    Return the colours of the current matplotlib colour cycle.

    Returns
    -------
    list[str]
        Colour strings from the active property cycle.
    """
    return plt.rcParams["axes.prop_cycle"].by_key()["color"]


def ordered_legend(ax: Axes, priorities: dict[str, float], **kwargs) -> None:
    """
    Draw the legend on ``ax`` with entries reordered by ``priorities``.

    Parameters
    ----------
    ax : Axes
        Axes whose labelled artists are placed in the legend.
    priorities : dict[str, float]
        Mapping from legend label to sort rank; entries with lower ranks appear
        first. Labels absent from the mapping keep their original order, after any
        ranked entries.
    **kwargs
        Additional keyword arguments passed to ``ax.legend``.
    """
    handles, labels = ax.get_legend_handles_labels()
    order = sorted(
        range(len(labels)),
        key=lambda i: (priorities.get(labels[i], len(labels)), i),
    )
    ax.legend([handles[i] for i in order], [labels[i] for i in order], **kwargs)


# Colours for stacked incidence bars, ordered least- to most-hidden. The base (grey) is the
# fully-observed layer; extras are taken from the tail so the final (most-hidden) segment is
# always purple (a 2-bar stack is grey+purple, a 3-bar stack grey+orange+purple).
_STACKED_BAR_BASE_COLOR = "tab:gray"
_STACKED_BAR_EXTRA_COLORS = ("tab:orange", "tab:purple")


def _stacked_bar_colors(n_layers: int) -> list[str]:
    n_extra = n_layers - 1
    if n_extra > len(_STACKED_BAR_EXTRA_COLORS):
        raise ValueError(
            f"at most {len(_STACKED_BAR_EXTRA_COLORS) + 1} stacked bar layers are supported"
        )
    return [
        _STACKED_BAR_BASE_COLOR,
        *_STACKED_BAR_EXTRA_COLORS[len(_STACKED_BAR_EXTRA_COLORS) - n_extra :],
    ]


def _as_bar_layers(bar_heights: Any) -> list[NDArray[np.float64]]:
    # A single series (1-D array or flat list of scalars) is one layer; a list/tuple whose
    # elements are themselves array-like is several stacked layers. Typed ``Any`` as a private
    # runtime dispatcher; the public ``plot_data_on_twin_ax`` carries the ``ArrayLike`` type.
    if (
        isinstance(bar_heights, (list, tuple))
        and len(bar_heights) > 0
        and all(np.ndim(layer) >= 1 for layer in bar_heights)
    ):
        return [np.asarray(layer, dtype=float) for layer in bar_heights]
    return [np.asarray(bar_heights, dtype=float)]
