import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np
import pytest

from endoutbreakvbd.utils import (
    get_colors,
    lognormal_params_from_median_percentile_2_5,
    month_start_xticks,
    plot_data_on_twin_ax,
    rep_no_from_grid,
    set_plot_config,
)


def test_rep_no_from_grid_nonperiodic_returns_expected_values():
    grid = np.array([10.0, 20.0, 30.0])
    t_vec = np.array([0, 2])
    out = rep_no_from_grid(t_vec, rep_no_grid=grid, periodic=False)
    np.testing.assert_array_equal(out, np.array([10.0, 30.0]))


def test_rep_no_from_grid_nonperiodic_raises_on_out_of_range():
    with pytest.raises(ValueError, match="outside the range"):
        rep_no_from_grid(np.array([-1, 1]), rep_no_grid=np.array([1.0, 2.0]), periodic=False)


def test_rep_no_from_grid_periodic_requires_doy_start():
    with pytest.raises(ValueError, match="doy_start"):
        rep_no_from_grid(np.array([0]), rep_no_grid=np.ones(365), periodic=True)


def test_rep_no_from_grid_periodic_requires_365_grid():
    with pytest.raises(ValueError, match="length 365"):
        rep_no_from_grid(np.array([0]), rep_no_grid=np.ones(364), periodic=True, doy_start=1)


def test_rep_no_from_grid_periodic_wraparound():
    grid = np.arange(1, 366)
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


def test_get_colors_returns_non_empty_color_list():
    colors = get_colors()
    assert isinstance(colors, list)
    assert len(colors) > 0
    for color in colors:
        mcolors.to_rgba(color)
