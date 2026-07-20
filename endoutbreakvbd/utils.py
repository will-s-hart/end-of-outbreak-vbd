import functools
from typing import Any, overload

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scipy.integrate
import scipy.optimize
import scipy.stats
import seaborn as sns
from matplotlib.axes import Axes
from numpy.typing import ArrayLike, NDArray

from endoutbreakvbd._types import FloatArray, IntArray


@overload
def rep_no_from_grid(
    t: int,
    *,
    rep_no_grid: FloatArray,
    periodic: bool,
    doy_start: int | None = None,
) -> float | FloatArray: ...


@overload
def rep_no_from_grid(
    t: IntArray,
    *,
    rep_no_grid: FloatArray,
    periodic: bool,
    doy_start: int | None = None,
) -> FloatArray: ...


def rep_no_from_grid(
    t: int | IntArray,
    *,
    rep_no_grid: FloatArray,
    periodic: bool,
    doy_start: int | None = None,
) -> float | FloatArray:
    """
    Function for interpolating a reproduction number grid to a vector of times.

    Parameters
    ----------
    t : int or IntArray
        Time(s) for which to interpolate the reproduction number.
    rep_no_grid : FloatArray
        Grid of reproduction numbers to interpolate from.
    periodic : bool
        Whether to treat the grid as periodic, describing reproduction numbers over a
        year. If True, the grid should have length 365, with the first element
        corresponding to day 1 of the year (January 1st).
    doy_start : int, optional
        Day of year (1-indexed) on which the outbreak starts (so that t=0
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
        t_grid_idx = (t + (doy_start - 1)) % 365
    else:
        if doy_start is not None:
            raise ValueError("doy_start should not be provided when periodic is False")
        if np.any(t < 0) or np.any(t >= len(rep_no_grid)):
            raise ValueError("t contains indices outside the range of rep_no_grid")
        t_grid_idx = t
    rep_no = rep_no_grid[t_grid_idx, ...]
    if rep_no.size == 1:
        return float(rep_no.item())
    return rep_no


def rescale_rep_no_grid_in_time(
    rep_no_grid: FloatArray, doy_season_centre: int, decay_speed: float
) -> FloatArray:
    """
    Rescale a periodic (year-long) reproduction number grid in time about the centre
    of the transmission season.

    The profile is stretched or compressed in time about ``doy_season_centre`` as
    ``R(d) = R_default(doy_season_centre + decay_speed * (d - doy_season_centre))``.
    More than half a year from the centre falls outside the transmission season, where
    R is held at its grid minimum (winter floor) so that a compressed season does not
    wrap around into a spurious repeated summer.

    Parameters
    ----------
    rep_no_grid : FloatArray
        Periodic grid of reproduction numbers over a year. Must have length 365, with
        the first element corresponding to day 1 of the year (January 1st).
    doy_season_centre : int
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
    centre_idx = doy_season_centre - 1  # grid is 0-indexed: doy d is at index d - 1
    half_year = 365 / 2
    day_idx_vec = np.arange(365)
    days_from_centre_vec = ((day_idx_vec - centre_idx + half_year) % 365) - half_year
    sampled_idx_vec = centre_idx + decay_speed * days_from_centre_vec
    rescaled_grid = np.interp(sampled_idx_vec, day_idx_vec, rep_no_grid, period=365)
    in_season_mask = np.abs(sampled_idx_vec - centre_idx) <= half_year
    return np.where(in_season_mask, rescaled_grid, rep_no_grid.min())


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


def fit_discretised_gamma(
    samples: IntArray | list[int], *, x0: tuple[float, float] = (1.5, 8.0)
) -> dict[str, Any]:
    """
    Fit a discretised gamma distribution to non-negative integer samples by maximum
    likelihood.

    Each integer value ``k`` is assigned the continuous-gamma mass on ``(k - 0.5, k +
    0.5]`` (the first bin starting at ``-0.5``), renormalised over ``0..max(samples)`` so
    the returned probability mass sums to one. The shape and scale are found by directly
    maximising the discrete likelihood (Nelder-Mead).

    Parameters
    ----------
    samples : IntArray or list[int]
        Non-negative integer samples (e.g. onset-to-report delays in days).
    x0 : tuple[float, float]
        Initial ``(shape, scale)`` for the optimiser.

    Returns
    -------
    dict[str, Any]
        ``support`` (integers ``0..max``), the fitted ``pmf_fitted`` and ``cdf``, the
        ``pmf_empirical`` of the samples, the maximum-likelihood ``shape`` and ``scale``,
        the sample count ``n``, and the sample ``mean``.
    """
    samples = np.asarray(samples, dtype=int)
    if samples.ndim != 1 or samples.size == 0:
        raise ValueError("samples must be a non-empty 1-D array")
    if np.any(samples < 0):
        raise ValueError("samples must be non-negative")
    max_val = int(samples.max())
    bin_edges = np.arange(-0.5, max_val + 1.5)

    def _pmf(shape: float, scale: float) -> NDArray[np.float64]:
        pmf = np.diff(scipy.stats.gamma(a=shape, scale=scale).cdf(bin_edges))
        return pmf / pmf.sum()

    def _neg_log_likelihood(params: NDArray[np.float64]) -> float:
        shape, scale = params
        if shape <= 0 or scale <= 0:
            return np.inf
        return -float(np.sum(np.log(_pmf(shape, scale)[samples] + 1e-300)))

    result = scipy.optimize.minimize(
        _neg_log_likelihood, x0=np.asarray(x0, dtype=float), method="Nelder-Mead"
    )
    shape, scale = result.x
    pmf_fitted = _pmf(shape, scale)
    return {
        "support": np.arange(max_val + 1),
        "pmf_fitted": pmf_fitted,
        # Clamp at 1: cumsum of the normalised pmf can overshoot by a float ULP, which the
        # under-reporting reporting-probability validation (delay CDF in [0, 1]) would reject.
        "cdf": np.minimum(np.cumsum(pmf_fitted), 1.0),
        "pmf_empirical": np.bincount(samples, minlength=max_val + 1) / samples.size,
        "shape": float(shape),
        "scale": float(scale),
        "n": int(samples.size),
        "mean": float(samples.mean()),
    }


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
    doy_month_start_vec = month_starts.dayofyear.to_numpy(copy=True)
    doy_month_start_vec[-1] = 366
    x_limits = ax.get_xlim()
    month_start_in_range = (doy_month_start_vec >= x_limits[0]) & (
        doy_month_start_vec <= x_limits[-1]
    )
    month_starts = month_starts[month_start_in_range]
    doy_month_start_vec = doy_month_start_vec[month_start_in_range]
    ax.set_xticks(doy_month_start_vec)
    labels = [
        f"{d.day} {d:%b}" if (i % interval_months == 0) else ""
        for i, d in enumerate(month_starts)
    ]
    ax.set_xticklabels(labels, rotation=0)


def dates_to_calendar_day_index(
    dates: pd.Series | pd.DatetimeIndex, year: int = 2017
) -> NDArray[np.int64]:
    """
    Convert calendar dates to a continuous calendar day index anchored at the start of ``year``.

    Within ``year`` this equals the day of year (1 January → 1). Unlike a bare day-of-year,
    it does not wrap when a window crosses into the next year (31 December 2017 → 365,
    1 January 2018 → 366), so it stays monotonic across the year boundary and pairs with
    ``month_start_xticks`` for a continuous calendar x-axis.

    Parameters
    ----------
    dates : pd.Series or pd.DatetimeIndex
        Datetime-valued series or index to convert.
    year : int
        Anchor year; day 1 is 1 January of this year.

    Returns
    -------
    NDArray[np.int64]
        Continuous calendar day index for each date.
    """
    return (pd.DatetimeIndex(dates) - pd.Timestamp(f"{year}-01-01")).days.to_numpy() + 1


def plot_incidence_on_twin_ax(
    ax: Axes,
    t_vec: ArrayLike,
    incidence: ArrayLike,
    *,
    incidence_labels: list[str | None] | None = None,
    alpha: float = 0.5,
    fitted_incidence: tuple[ArrayLike, ArrayLike, ArrayLike] | None = None,
    fitted_incidence_color: str | None = None,
    fitted_incidence_label: str | None = None,
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
    incidence : ArrayLike
        Number of cases at each time. A single 1-D series draws one bar; a list of series
        draws them stacked (ordered least- to most-hidden), coloured by an internal palette
        (grey base, then orange/purple). NaN marks an unobserved time and draws no bar.
    incidence_labels : list[str | None], optional
        Legend labels for the stacked bars (one per series; ``None`` to omit a label). If
        omitted, the bars are unlabelled.
    alpha : float
        Bar transparency.
    fitted_incidence : tuple[ArrayLike, ArrayLike, ArrayLike], optional
        ``(mean, lower, upper)`` of a fitted-case series to overlay: the credible band is
        drawn behind the bars and the mean line in front.
    fitted_incidence_color : str, optional
        Colour of the ``fitted_incidence`` band and mean line.
    fitted_incidence_label : str, optional
        Legend label for the ``fitted_incidence`` mean line.

    Returns
    -------
    Axes
        The created twin axis.
    """
    incidence_layers = _as_incidence_layers(incidence)
    # The day axis may run one projected day past the incidence bars (a fit reports the
    # current-day risk one day beyond its data); pad the bars with a trailing zero so a longer
    # `t_vec` still aligns (no bar is drawn for the projected day).
    n_times = len(np.asarray(t_vec))
    incidence_layers = [
        np.append(layer, np.zeros(n_times - len(layer)))
        if len(layer) < n_times
        else layer
        for layer in incidence_layers
    ]
    colors = _stacked_bar_colors(len(incidence_layers))
    labels = (
        incidence_labels
        if incidence_labels is not None
        else [None] * len(incidence_layers)
    )

    twin_ax = ax.twinx()
    if fitted_incidence is not None:
        fitted_mean_vec, fitted_lower_vec, fitted_upper_vec = (
            np.asarray(a, dtype=float) for a in fitted_incidence
        )
        twin_ax.fill_between(
            t_vec,
            fitted_lower_vec,
            fitted_upper_vec,
            color=fitted_incidence_color,
            alpha=0.25,
            zorder=1,
        )
    incidence_total_vec = np.zeros(len(incidence_layers[0]), dtype=float)
    for layer, color, label in zip(incidence_layers, colors, labels, strict=True):
        twin_ax.bar(
            t_vec,
            layer,
            bottom=incidence_total_vec,
            color=color,
            alpha=alpha,
            label=label,
            zorder=2,
        )
        # Preserve NaN as a missing bar while excluding it from stacked totals and axis scaling.
        incidence_total_vec = incidence_total_vec + np.where(
            np.isnan(layer), 0.0, layer
        )
    if fitted_incidence is not None:
        twin_ax.plot(
            t_vec,
            fitted_mean_vec,
            color=fitted_incidence_color,
            label=fitted_incidence_label,
            zorder=3,
        )
        incidence_total_vec = np.fmax(incidence_total_vec, fitted_upper_vec)
    twin_ax.set_ylim(0, np.max(incidence_total_vec))
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


def _as_incidence_layers(incidence: Any) -> list[NDArray[np.float64]]:
    # A single series (1-D array or flat list of scalars) is one layer; a list/tuple whose
    # elements are themselves array-like is several stacked layers. Typed ``Any`` as a private
    # runtime dispatcher; the public ``plot_incidence_on_twin_ax`` carries the ``ArrayLike`` type.
    if (
        isinstance(incidence, (list, tuple))
        and len(incidence) > 0
        and all(np.ndim(layer) >= 1 for layer in incidence)
    ):
        return [np.asarray(layer, dtype=float) for layer in incidence]
    return [np.asarray(incidence, dtype=float)]
