"""Conceptual schematic figure for the paper.

Reads the outbreak CSV produced by `scripts/schematic.py` and draws the
2x2 schematic panel layout with arrow chrome.
"""

from typing import Any

import matplotlib.image as mpimg
import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.offsetbox import AnnotationBbox, OffsetImage
from matplotlib.path import Path

from endoutbreakvbd.utils import month_start_xticks, set_plot_config
from scripts.inputs import get_inputs_schematic

PANEL_COLORS = {
    "A": "#7E57C2",
    "B": "#2E9E5C",
    "C": "#1E88E5",
    "D": "#F57C00",
}

PANEL_TITLES = {
    "A": "A. Input 1: disease incidence time series",
    "B": "B. Input 2: serial interval distribution",
    "C": "C. Input 3: seasonal time-dependent reproduction number",
    "D": "D. Output: risk of additional cases",
}

FIGURE_WIDTH = 13.0
FIGURE_HEIGHT = 9.0
FIGURE_ASPECT_RATIO = FIGURE_WIDTH / FIGURE_HEIGHT

LEFT_MARGIN = 0.035
RIGHT_MARGIN = 0.035
TOP_MARGIN = 0.025
BOTTOM_MARGIN = 0.04
HORIZONTAL_GAP = 0.06
VERTICAL_GAP = 0.11
TITLE_BAR_HEIGHT = 0.05
ROUNDING = 0.010

PANEL_HEIGHT = (1 - TOP_MARGIN - BOTTOM_MARGIN - VERTICAL_GAP) / 2
LEFT_RIGHT_RATIO = 1.0
AVAILABLE_PANEL_WIDTH = 1 - LEFT_MARGIN - RIGHT_MARGIN - HORIZONTAL_GAP
PANEL_WIDTH_RIGHT = AVAILABLE_PANEL_WIDTH / (1 + LEFT_RIGHT_RATIO)
PANEL_WIDTH_LEFT = AVAILABLE_PANEL_WIDTH - PANEL_WIDTH_RIGHT

PANEL_BOUNDS = {
    "A": (
        LEFT_MARGIN,
        BOTTOM_MARGIN + PANEL_HEIGHT + VERTICAL_GAP,
        PANEL_WIDTH_LEFT,
        PANEL_HEIGHT,
    ),
    "B": (
        LEFT_MARGIN + PANEL_WIDTH_LEFT + HORIZONTAL_GAP,
        BOTTOM_MARGIN + PANEL_HEIGHT + VERTICAL_GAP,
        PANEL_WIDTH_RIGHT,
        PANEL_HEIGHT,
    ),
    "C": (LEFT_MARGIN, BOTTOM_MARGIN, PANEL_WIDTH_LEFT, PANEL_HEIGHT),
    "D": (
        LEFT_MARGIN + PANEL_WIDTH_LEFT + HORIZONTAL_GAP,
        BOTTOM_MARGIN,
        PANEL_WIDTH_RIGHT,
        PANEL_HEIGHT,
    ),
}

AXES_PADDING_LEFT = 0.045
AXES_PADDING_RIGHT = 0.020
AXES_PADDING_BOTTOM = 0.065
AXES_PADDING_TOP = 0.010

DOY_MIN = 91
DOY_MAX = 335
ADDITIONAL_CASE_PROB_THRESHOLD = 0.05


def make_plots():
    set_plot_config()
    inputs = get_inputs_schematic()
    outbreak_df = pd.read_csv(
        inputs["results_paths"]["outbreak"], index_col="day_of_outbreak"
    )
    doy_start = int(outbreak_df["day_of_year"].iloc[0])
    incidence_vec = outbreak_df["incidence"].to_numpy()
    doy_final_case = doy_start + int(np.nonzero(incidence_vec)[0][-1])
    doy_current = doy_final_case + inputs["days_after_final_case"]

    fig = plt.figure(figsize=(FIGURE_WIDTH, FIGURE_HEIGHT))

    for key in ("A", "B", "C", "D"):
        _decorate_panel(fig, key)

    _populate_panel_a(
        _add_panel_axes(fig, "A"),
        doy_start=doy_start,
        incidence_vec=incidence_vec,
        doy_current=doy_current,
    )
    _populate_panel_b(
        _add_panel_axes(fig, "B"),
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
    )
    _populate_panel_c(
        _add_panel_axes(fig, "C"),
        outbreak_df=outbreak_df,
        seasonal_profile_grid=inputs["seasonal_profile_grid"],
    )
    _populate_panel_d(
        _add_panel_axes(fig, "D"),
        outbreak_df=outbreak_df,
        doy_current=doy_current,
        intervention_graphic_path=inputs["intervention_graphic_path"],
        safe_graphic_path=inputs["safe_graphic_path"],
    )

    _draw_inference_arrows(fig)
    _draw_output_arrows(fig)

    fig.savefig(inputs["fig_path"], dpi=300)
    plt.close(fig)


def _rounded_box_path(left, bottom, width, height, *, top_radius, bottom_radius):
    top_radius_x = top_radius
    top_radius_y = top_radius * FIGURE_ASPECT_RATIO
    bottom_radius_x = bottom_radius
    bottom_radius_y = bottom_radius * FIGURE_ASPECT_RATIO
    vertices = [
        (left + top_radius_x, bottom + height),
        (left + width - top_radius_x, bottom + height),
        (left + width, bottom + height),
        (left + width, bottom + height - top_radius_y),
        (left + width, bottom + bottom_radius_y),
        (left + width, bottom),
        (left + width - bottom_radius_x, bottom),
        (left + bottom_radius_x, bottom),
        (left, bottom),
        (left, bottom + bottom_radius_y),
        (left, bottom + height - top_radius_y),
        (left, bottom + height),
        (left + top_radius_x, bottom + height),
        (0.0, 0.0),
    ]
    codes = [
        Path.MOVETO,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.LINETO,
        Path.CURVE3,
        Path.CURVE3,
        Path.CLOSEPOLY,
    ]
    return Path(vertices, codes)


def _decorate_panel(fig, panel_key):
    panel_left, panel_bottom, panel_width, panel_height = PANEL_BOUNDS[panel_key]
    color = PANEL_COLORS[panel_key]
    title = PANEL_TITLES[panel_key]

    title_bottom = panel_bottom + panel_height - TITLE_BAR_HEIGHT
    title_path = _rounded_box_path(
        panel_left,
        title_bottom,
        panel_width,
        TITLE_BAR_HEIGHT,
        top_radius=ROUNDING,
        bottom_radius=0.0,
    )
    fig.patches.append(
        mpatches.PathPatch(
            title_path,
            facecolor=color,
            edgecolor="none",
            linewidth=0,
            transform=fig.transFigure,
            figure=fig,
            zorder=9,
        )
    )

    border_path = _rounded_box_path(
        panel_left,
        panel_bottom,
        panel_width,
        panel_height,
        top_radius=ROUNDING,
        bottom_radius=ROUNDING,
    )
    fig.patches.append(
        mpatches.PathPatch(
            border_path,
            facecolor="none",
            edgecolor=color,
            linewidth=2,
            transform=fig.transFigure,
            figure=fig,
            zorder=10,
        )
    )

    fig.text(
        panel_left + panel_width / 2,
        title_bottom + TITLE_BAR_HEIGHT / 2,
        title,
        color="white",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=11,
    )


def _add_panel_axes(fig, panel_key, *, left_padding=AXES_PADDING_LEFT):
    panel_left, panel_bottom, panel_width, panel_height = PANEL_BOUNDS[panel_key]
    return fig.add_axes(
        [
            panel_left + left_padding,
            panel_bottom + AXES_PADDING_BOTTOM,
            panel_width - left_padding - AXES_PADDING_RIGHT,
            panel_height - TITLE_BAR_HEIGHT - AXES_PADDING_TOP - AXES_PADDING_BOTTOM,
        ]
    )


def _populate_panel_a(ax, *, doy_start, incidence_vec, doy_current):
    doy_vec = doy_start + np.arange(incidence_vec.size)
    plot_mask = doy_vec <= DOY_MAX
    ax.bar(
        doy_vec[plot_mask],
        incidence_vec[plot_mask],
        color=PANEL_COLORS["A"],
        width=1.0,
        align="center",
    )

    y_max = max(int(incidence_vec.max()) * 1.18, 1)
    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylim(0, y_max)

    ax.axvline(doy_current, color="black", linestyle="--", linewidth=1.5)
    ax.text(
        doy_current - 3,
        y_max * 0.93,
        "Current day",
        ha="right",
        va="top",
        fontsize=12,
    )

    ax.set_xlabel("Date")
    ax.set_ylabel("Number of cases")
    ax.set_yticks([])
    month_start_xticks(ax, year=2017, interval_months=1)
    ax.tick_params(axis="x", labelsize=12)


def _populate_panel_b(ax, *, serial_interval_dist_vec):
    serial_interval_lag_vec = np.arange(1, serial_interval_dist_vec.size + 1)
    ax.bar(
        serial_interval_lag_vec,
        serial_interval_dist_vec,
        color=PANEL_COLORS["B"],
        width=1.0,
        align="center",
    )
    ax.set_xlim(0.5, serial_interval_lag_vec[-1] + 0.5)
    ax.set_ylim(0, serial_interval_dist_vec.max() * 1.18)
    ax.set_xlabel("Serial interval (days)", labelpad=14)
    ax.set_ylabel("Probability")
    ax.set_xticks([])
    ax.set_yticks([])


def _populate_panel_c(ax, *, outbreak_df, seasonal_profile_grid):
    doy_vec = outbreak_df["day_of_year"].to_numpy()
    rep_no_mean_vec = outbreak_df["reproduction_number_mean"].to_numpy()
    rep_no_lower_vec = outbreak_df["reproduction_number_lower"].to_numpy()
    rep_no_upper_vec = outbreak_df["reproduction_number_upper"].to_numpy()

    in_range_mask = doy_vec <= DOY_MAX
    doy_vec = doy_vec[in_range_mask]
    rep_no_mean_vec = rep_no_mean_vec[in_range_mask]
    rep_no_lower_vec = rep_no_lower_vec[in_range_mask]
    rep_no_upper_vec = rep_no_upper_vec[in_range_mask]

    doy_seasonal_vec = np.arange(DOY_MIN, DOY_MAX + 1)
    seasonal_rep_no_vec = seasonal_profile_grid[doy_seasonal_vec - 1]

    fill_handle = ax.fill_between(
        doy_vec,
        rep_no_lower_vec,
        rep_no_upper_vec,
        color=PANEL_COLORS["C"],
        alpha=0.35,
        linewidth=0,
    )
    (seasonal_handle,) = ax.plot(
        doy_seasonal_vec, seasonal_rep_no_vec, color="black", linewidth=2
    )
    (inferred_handle,) = ax.plot(
        doy_vec, rep_no_mean_vec, color=PANEL_COLORS["C"], linewidth=2
    )

    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylim(0, rep_no_upper_vec.max() * 1.05)
    ax.set_xlabel("Date")
    ax.set_ylabel("Time-dependent\nreproduction number")
    ax.set_yticks([])
    month_start_xticks(ax, year=2017, interval_months=1)
    ax.tick_params(axis="x", labelsize=12)
    ax.legend(
        [seasonal_handle, inferred_handle, fill_handle],
        ["Seasonal", "Seasonal + case data", "95% credible interval"],
        loc="upper right",
        framealpha=0.9,
        fontsize=11,
    )


def _populate_panel_d(
    ax,
    *,
    outbreak_df,
    doy_current,
    intervention_graphic_path,
    safe_graphic_path,
):
    doy_vec = outbreak_df["day_of_year"].to_numpy()
    prob_vec = outbreak_df["additional_case_prob"].to_numpy()
    x_lower = 305  # 1 November (non-leap year)
    plot_mask = (doy_vec >= x_lower) & (doy_vec <= doy_current)

    y_upper = float(np.ceil(prob_vec[plot_mask].max() * 40) / 40)

    ax.axhspan(0, ADDITIONAL_CASE_PROB_THRESHOLD, color="#9CCC65", alpha=0.45, zorder=0)
    ax.axhspan(
        ADDITIONAL_CASE_PROB_THRESHOLD,
        1.0,
        color="#FFE082",
        alpha=0.45,
        zorder=0,
    )
    ax.axhline(
        ADDITIONAL_CASE_PROB_THRESHOLD,
        color="#558B2F",
        linestyle=":",
        linewidth=1,
        zorder=1,
    )
    ax.plot(
        doy_vec[plot_mask],
        prob_vec[plot_mask],
        color=PANEL_COLORS["D"],
        linewidth=2,
    )
    ax.axvline(doy_current, color="black", linestyle="--", linewidth=1.5)

    ax.set_xlim(x_lower, DOY_MAX)
    ax.set_ylim(0, y_upper)
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability of additional\ncases on/after date")
    ax.set_yticks([])

    ax.text(
        doy_current - 0.5,
        y_upper * 0.93,
        "Current day",
        ha="right",
        va="top",
        fontsize=12,
    )
    intervention_image = mpimg.imread(intervention_graphic_path)
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(intervention_image, zoom=0.18),
            (0.7, 0.56),
            xycoords="axes fraction",
            frameon=False,
        )
    )
    safe_image = mpimg.imread(safe_graphic_path)
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(safe_image, zoom=0.75 * 0.18),
            (0.71, 0.06),
            xycoords="axes fraction",
            frameon=False,
        )
    )

    month_start_xticks(ax, year=2017, interval_months=1)
    ax.tick_params(axis="x", labelsize=12)


def _draw_inference_arrows(fig):
    panel_a_left, panel_a_bottom, panel_a_width, _ = PANEL_BOUNDS["A"]
    panel_b_left, panel_b_bottom, panel_b_width, _ = PANEL_BOUNDS["B"]
    _, panel_c_bottom, _, panel_c_height = PANEL_BOUNDS["C"]

    panel_a_centre_x = panel_a_left + panel_a_width * 0.5
    panel_b_centre_x = panel_b_left + panel_b_width * 0.5
    panel_c_top = panel_c_bottom + panel_c_height
    rail_y = (panel_a_bottom + panel_c_top) / 2

    line_kwargs: dict[str, Any] = dict(
        arrowstyle="-",
        linestyle="--",
        color="dimgray",
        linewidth=2,
        transform=fig.transFigure,
        figure=fig,
        zorder=5,
    )

    fig.patches.append(
        mpatches.FancyArrowPatch(
            (panel_a_centre_x, panel_a_bottom),
            (panel_a_centre_x, rail_y),
            **line_kwargs,
        )
    )
    fig.patches.append(
        mpatches.FancyArrowPatch(
            (panel_b_centre_x, panel_b_bottom),
            (panel_b_centre_x, rail_y),
            **line_kwargs,
        )
    )
    fig.patches.append(
        mpatches.FancyArrowPatch(
            (panel_a_centre_x, rail_y),
            (panel_b_centre_x, rail_y),
            **line_kwargs,
        )
    )
    fig.patches.append(
        mpatches.FancyArrowPatch(
            (panel_a_centre_x, rail_y),
            (panel_a_centre_x, panel_c_top),
            arrowstyle="-|>",
            linestyle="--",
            color="dimgray",
            linewidth=2,
            mutation_scale=18,
            transform=fig.transFigure,
            figure=fig,
            zorder=5,
        )
    )

    fig.text(
        panel_a_centre_x - 0.015,
        rail_y,
        "Inference (optional)",
        ha="right",
        va="center",
        fontsize=14,
        fontstyle="italic",
        color="dimgray",
    )


def _block_arrow(fig, start, end, *, connectionstyle="arc3,rad=0"):
    fig.patches.append(
        mpatches.FancyArrowPatch(
            start,
            end,
            arrowstyle="simple,head_length=0.7,head_width=0.9,tail_width=0.35",
            connectionstyle=connectionstyle,
            color="dimgray",
            linewidth=0,
            mutation_scale=30,
            transform=fig.transFigure,
            figure=fig,
            zorder=4,
        )
    )


def _draw_output_arrows(fig):
    panel_a_left, panel_a_bottom, panel_a_width, _ = PANEL_BOUNDS["A"]
    panel_b_left, panel_b_bottom, panel_b_width, _ = PANEL_BOUNDS["B"]
    panel_c_left, panel_c_bottom, panel_c_width, panel_c_height = PANEL_BOUNDS["C"]
    panel_d_left, panel_d_bottom, panel_d_width, panel_d_height = PANEL_BOUNDS["D"]

    _block_arrow(
        fig,
        start=(panel_c_left + panel_c_width, panel_c_bottom + panel_c_height * 0.55),
        end=(panel_d_left, panel_d_bottom + panel_d_height * 0.55),
    )
    _block_arrow(
        fig,
        start=(panel_b_left + panel_b_width * 0.7, panel_b_bottom),
        end=(
            panel_d_left + panel_d_width * 0.7,
            panel_d_bottom + panel_d_height,
        ),
    )
    _block_arrow(
        fig,
        start=(panel_a_left + panel_a_width, panel_a_bottom),
        end=(panel_d_left, panel_d_bottom + panel_d_height),
    )


if __name__ == "__main__":
    make_plots()
