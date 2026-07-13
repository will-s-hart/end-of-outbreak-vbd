# Note that AI tools were used to generate tests

import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pytest
import scipy.stats

from endoutbreakvbd.utils import (
    dates_to_day_index,
    discretise_cori,
    get_colors,
    lognormal_params_from_median_percentile_2_5,
    month_start_xticks,
    plot_data_on_twin_ax,
    rep_no_from_grid,
    set_plot_config,
)


def test_discretise_cori_probabilities_are_nonnegative_and_sum_to_one():
    dist = scipy.stats.gamma(a=2.0, scale=1.0)
    p_vec = discretise_cori(dist_cont=dist, max_val=12, allow_zero=True)
    assert np.all(p_vec >= 0)
    assert np.isclose(np.sum(p_vec), 1.0, atol=1e-8)


def test_discretise_cori_allow_zero_changes_output_length():
    dist = scipy.stats.gamma(a=2.0, scale=1.0)
    p_with_zero = discretise_cori(dist_cont=dist, max_val=10, allow_zero=True)
    p_without_zero = discretise_cori(dist_cont=dist, max_val=10, allow_zero=False)
    assert len(p_with_zero) == 11
    assert len(p_without_zero) == 10


def test_discretise_cori_requires_keyword_only_dist_cont():
    dist = scipy.stats.gamma(a=2.0, scale=1.0)
    with pytest.raises(TypeError, match="positional arguments"):
        discretise_cori(dist, max_val=10, allow_zero=True)  # ty: ignore[missing-argument, too-many-positional-arguments]


def test_discretise_cori_requires_max_val():
    dist = scipy.stats.gamma(a=2.0, scale=1.0)
    with pytest.raises(TypeError, match="missing 1 required keyword-only argument"):
        discretise_cori(dist_cont=dist)  # ty: ignore[missing-argument]


def test_discretise_cori_requires_nonnegative_max_val():
    dist = scipy.stats.gamma(a=2.0, scale=1.0)
    with pytest.raises(ValueError, match="max_val must be non-negative"):
        discretise_cori(dist_cont=dist, max_val=-1, allow_zero=True)


def test_discretise_cori_requires_at_least_one_when_no_zero_allowed():
    dist = scipy.stats.gamma(a=2.0, scale=1.0)
    with pytest.raises(ValueError, match="at least 1"):
        discretise_cori(dist_cont=dist, max_val=0, allow_zero=False)


def test_rep_no_from_grid_nonperiodic_returns_expected_values():
    grid = np.array([10.0, 20.0, 30.0])
    t_vec = np.array([0, 2])
    out = rep_no_from_grid(t_vec, rep_no_grid=grid, periodic=False)
    np.testing.assert_array_equal(out, np.array([10.0, 30.0]))


def test_rep_no_from_grid_nonperiodic_raises_on_out_of_range():
    with pytest.raises(ValueError, match="outside the range"):
        rep_no_from_grid(
            np.array([-1, 1]), rep_no_grid=np.array([1.0, 2.0]), periodic=False
        )


def test_rep_no_from_grid_periodic_requires_doy_start():
    with pytest.raises(ValueError, match="doy_start"):
        rep_no_from_grid(np.array([0]), rep_no_grid=np.ones(365), periodic=True)


def test_rep_no_from_grid_periodic_requires_365_grid():
    with pytest.raises(ValueError, match="length 365"):
        rep_no_from_grid(
            np.array([0]), rep_no_grid=np.ones(364), periodic=True, doy_start=1
        )


def test_rep_no_from_grid_periodic_wraparound():
    grid = np.arange(1, 366, dtype=float)
    t_vec = np.array([0, 1, 364, 365])
    out = rep_no_from_grid(t_vec, rep_no_grid=grid, periodic=True, doy_start=1)
    np.testing.assert_array_equal(out, np.array([1, 2, 365, 1]))


def test_lognormal_params_from_median_percentile_2_5():
    params = lognormal_params_from_median_percentile_2_5(median=1.0, percentile_2_5=0.2)
    assert params["mu"] == pytest.approx(0.0)
    assert params["sigma"] > 0
    assert params["sigma"] == pytest.approx(0.8211568810086008)


def test_set_plot_config_updates_key_rcparams():
    set_plot_config()
    assert plt.rcParams["figure.figsize"] == [7.0, 7.0]
    assert plt.rcParams["savefig.format"] == "svg"
    assert plt.rcParams["svg.fonttype"] == "none"
    assert plt.rcParams["font.family"] == ["sans-serif"]


def test_month_start_xticks_sets_expected_ticks_and_labels():
    fig, ax = plt.subplots()
    try:
        ax.set_xlim(1, 120)
        month_start_xticks(ax, year=2017, interval_months=2)
        ticks = ax.get_xticks()
        labels = [tick.get_text() for tick in ax.get_xticklabels()]
        np.testing.assert_array_equal(ticks, np.array([1, 32, 60, 91]))
        assert labels == ["1 Jan", "", "1 Mar", ""]
    finally:
        plt.close(fig)


def test_plot_data_on_twin_ax_creates_configured_axis():
    fig, ax = plt.subplots()
    try:
        twin_ax = plot_data_on_twin_ax(ax, np.array([1, 2, 3]), np.array([0, 2, 1]))
        assert twin_ax.get_ylabel() == "Number of cases"
        assert twin_ax.get_ylim()[1] == pytest.approx(2)
        assert twin_ax.spines["right"].get_visible()
        assert not twin_ax.spines["left"].get_visible()
        assert not twin_ax.spines["bottom"].get_visible()
    finally:
        plt.close(fig)


def test_plot_data_on_twin_ax_stacks_layers_and_overlays_fitted():
    fig, ax = plt.subplots()
    try:
        twin_ax = plot_data_on_twin_ax(
            ax,
            np.array([0, 1, 2]),
            [np.ones(3), np.ones(3), np.ones(3)],
            bar_labels=["reported", "later", "never"],
            fitted=(np.ones(3), 0.5 * np.ones(3), 5.0 * np.ones(3)),
            fitted_color="tab:blue",
            fitted_label="fitted",
        )
        # Three stacked unit bars sum to 3, but the fitted upper band (5) sets the y-limit.
        assert twin_ax.get_ylim()[1] == pytest.approx(5.0)
        _, labels = twin_ax.get_legend_handles_labels()
        assert {"reported", "later", "never", "fitted"} <= set(labels)
    finally:
        plt.close(fig)


def test_plot_data_on_twin_ax_rejects_too_many_stacked_layers():
    fig, ax = plt.subplots()
    try:
        with pytest.raises(ValueError, match="at most 3 stacked"):
            plot_data_on_twin_ax(ax, [0, 1], [[1, 1], [1, 1], [1, 1], [1, 1]])
    finally:
        plt.close(fig)


def test_dates_to_day_index_matches_day_of_year_and_does_not_wrap():
    dates = pd.to_datetime(["2017-01-01", "2017-12-31", "2018-01-01"])
    # 2017 is not a leap year: 31 Dec is day 365, and the following day continues to 366
    # rather than wrapping back to 1 (which a bare day-of-year would).
    np.testing.assert_array_equal(dates_to_day_index(dates), np.array([1, 365, 366]))


def test_get_colors_returns_non_empty_color_list():
    colors = get_colors()
    assert isinstance(colors, list)
    assert len(colors) > 0
    for color in colors:
        mcolors.to_rgba(color)
