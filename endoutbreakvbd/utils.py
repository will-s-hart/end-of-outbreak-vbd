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
        "cdf": np.cumsum(pmf_fitted),
        "pmf_empirical": np.bincount(samples, minlength=max_val + 1) / samples.size,
        "shape": float(shape),
        "scale": float(scale),
        "n": int(samples.size),
        "mean": float(samples.mean()),
    }


def renewal_convolution_matrix(
    serial_interval_dist_vec: ArrayLike, n_days: int
) -> FloatArray:
    """
    Renewal-equation force of infection expressed as a (constant) matrix operator.

    Returns the ``n_days`` by ``n_days`` lower-triangular matrix ``conv_mat`` for which
    ``conv_mat @ incidence_vec`` is the renewal force of infection
    ``foi[s] = sum_{r < s} incidence_vec[r] * serial_interval[s - 1 - r]`` — the same quantity
    ``model.run_renewal_model`` accumulates one day at a time, but vectorised for a fixed
    incidence series (as needed by the inference models). Current-day incidence never
    contributes to its own force of infection, so ``conv_mat`` is strictly lower-triangular
    (row 0 is zero).

    Parameters
    ----------
    serial_interval_dist_vec : ArrayLike
        Discretised serial interval distribution (probability mass per day). Zero-extended
        internally when shorter than ``n_days - 1``.
    n_days : int
        Number of days (the size of the square matrix).

    Returns
    -------
    FloatArray
        The ``n_days`` by ``n_days`` lower-triangular convolution matrix.
    """
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec, dtype=float)
    serial_interval_ext = np.concatenate(
        [
            serial_interval_dist_vec,
            np.zeros(max(n_days - 1 - len(serial_interval_dist_vec), 0)),
        ]
    )
    conv_mat = np.zeros((n_days, n_days))
    for s in range(1, n_days):
        conv_mat[s, :s] = serial_interval_ext[:s][::-1]
    return conv_mat


def decision_delays_from_final_case(
    *,
    prob_vec: ArrayLike,
    days: ArrayLike,
    perc_risk_thresholds: ArrayLike,
    time_final_case: int,
) -> FloatArray:
    """
    Days from the final case until the additional-case probability first drops below each risk
    threshold, considering only the days after the final case.

    ``prob_vec[i]`` is the probability on outbreak day ``days[i]``; the two share an ordering
    but ``days`` need not be contiguous (e.g. strided real-time snapshots). Returns NaN for any
    threshold the risk never crosses over that window.

    Parameters
    ----------
    prob_vec : ArrayLike
        Probability of additional cases at each of ``days``.
    days : ArrayLike
        Outbreak day of each entry in ``prob_vec``.
    perc_risk_thresholds : ArrayLike
        Risk threshold(s), expressed as a percentage.
    time_final_case : int
        Outbreak day of the final case; delays are measured from it.

    Returns
    -------
    FloatArray
        Delay (days) at which the risk first falls below each threshold (NaN if it never does).
    """
    prob_vec = np.asarray(prob_vec, dtype=float)
    days = np.asarray(days)
    if prob_vec.shape != days.shape:
        raise ValueError("prob_vec and days must have the same shape")
    thresholds = np.atleast_1d(np.asarray(perc_risk_thresholds, dtype=float))
    after_final_case = days > time_final_case
    prob_after = prob_vec[after_final_case]
    days_after = days[after_final_case]
    delays = np.full(len(thresholds), np.nan)
    for j, threshold in enumerate(thresholds):
        below = np.nonzero(prob_after < threshold / 100)[0]
        if below.size:
            delays[j] = days_after[below[0]] - time_final_case
    return delays


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


def plot_data_on_twin_ax(
    ax: Axes,
    t_vec: ArrayLike,
    incidence_vec: ArrayLike,
    incidence_vec_extra: ArrayLike | None = None,
    color_extra: str = "lightgray",
) -> Axes:
    """
    Plot an incidence time series as a bar chart on a twin y-axis.

    The primary axes are raised above the twin axis so that the bars sit behind
    the primary-axis artists (curves and legend).

    Parameters
    ----------
    ax : Axes
        Primary axes to attach the twin axis to.
    t_vec : ArrayLike
        Times (days) for the incidence bars.
    incidence_vec : ArrayLike
        Number of cases at each time.
    incidence_vec_extra : ArrayLike, optional
        A second series stacked on top of ``incidence_vec`` in a lighter colour (e.g.
        unreported cases on top of reported ones). If None, only ``incidence_vec`` is
        drawn.
    color_extra : str
        Colour of the stacked ``incidence_vec_extra`` bars.

    Returns
    -------
    Axes
        The created twin axis.
    """
    twin_ax = ax.twinx()
    twin_ax.bar(t_vec, incidence_vec, color="tab:gray", alpha=0.5)
    total = np.asarray(incidence_vec, dtype=float)
    if incidence_vec_extra is not None:
        twin_ax.bar(
            t_vec,
            incidence_vec_extra,
            bottom=incidence_vec,
            color=color_extra,
            alpha=0.5,
        )
        total = total + np.asarray(incidence_vec_extra, dtype=float)
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
