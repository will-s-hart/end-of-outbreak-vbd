"""Conceptual schematic figure: a 2x2 panel layout (three inputs, one output) drawn from the
outbreak simulated by `scripts/schematic.py`.
"""

from typing import Any

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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

AXES_PADDING_LEFT = 0.062
AXES_PADDING_RIGHT = 0.020
AXES_PADDING_BOTTOM = 0.065
AXES_PADDING_TOP = 0.010

DOY_MIN = 91
DOY_MAX = 335
ADDITIONAL_CASE_PROB_THRESHOLD = 0.05

SAFE_BAND_COLOR = "#9CCC65"
ELEVATED_BAND_COLOR = "#FFE082"
SAFE_COLOR = "#558B2F"
INTERVENTION_COLOR = PANEL_COLORS["D"]
MOSQUITO_COLOR = "#333333"

# Both cards use equal drawing units and the same width in panel-D axes coordinates.
CARD_AXES_LEFT = 0.46
CARD_AXES_WIDTH = 0.5
CARD_WIDTH = 10.0
INTERVENTION_CARD_HEIGHT = 8.7
SAFE_LABEL_HEIGHT = 1.3
ICON_SCALE = 1.25
CARD_CORNER_RADIUS = 0.45
ICON_LINEWIDTH = 2.0


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
    )

    _draw_inference_arrows(fig)
    _draw_output_arrows(fig)

    fig.savefig(inputs["fig_path"], dpi=300)
    plt.close(fig)


def _rounded_box_path(left, bottom, width, height, *, top_radius, bottom_radius):
    # Box path in figure coordinates with separately rounded top and bottom corners.
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
    # Draw a panel's coloured title bar, title and rounded border.
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
    # Add axes inside a panel's border, below its title bar.
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
    # Simulated incidence, with the current day marked.
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
    # Serial-interval distribution.
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
    # Seasonal R_t profile and the posterior R_t after fitting to the case data.
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


def _populate_panel_d(ax, *, outbreak_df, doy_current):
    doy_vec = outbreak_df["day_of_year"].to_numpy()
    prob_vec = outbreak_df["additional_case_prob"].to_numpy()
    x_lower = 305  # 1 November (non-leap year)
    plot_mask = (doy_vec >= x_lower) & (doy_vec <= doy_current)

    y_upper = float(np.ceil(prob_vec[plot_mask].max() * 40) / 40)

    ax.axhspan(
        0, ADDITIONAL_CASE_PROB_THRESHOLD, color=SAFE_BAND_COLOR, alpha=0.45, zorder=0
    )
    ax.axhspan(
        ADDITIONAL_CASE_PROB_THRESHOLD,
        1.0,
        color=ELEVATED_BAND_COLOR,
        alpha=0.45,
        zorder=0,
    )
    ax.axhline(
        ADDITIONAL_CASE_PROB_THRESHOLD,
        color=SAFE_COLOR,
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
    ax.set_ylabel("Risk of additional\ncases on/after date")
    ax.set_yticks([])

    ax.text(
        doy_current - 0.5,
        y_upper * 0.93,
        "Current day",
        ha="right",
        va="top",
        fontsize=12,
    )
    threshold_axes_y = ADDITIONAL_CASE_PROB_THRESHOLD / y_upper
    _draw_intervention_card(ax, centre_y=(threshold_axes_y + 1) / 2)
    _draw_safe_label(ax, centre_y=threshold_axes_y / 2)

    month_start_xticks(ax, year=2017, interval_months=1)
    ax.tick_params(axis="x", labelsize=12)


def _add_card_axes(ax, *, centre_y, height, color):
    """Create a rounded card with equal x/y units, centred in the risk band."""
    fig_width, fig_height = ax.figure.get_size_inches()
    ax_bbox = ax.get_position()
    axes_height = (
        CARD_AXES_WIDTH
        * (height / CARD_WIDTH)
        * (ax_bbox.width * fig_width)
        / (ax_bbox.height * fig_height)
    )
    card_ax = ax.inset_axes(
        [CARD_AXES_LEFT, centre_y - axes_height / 2, CARD_AXES_WIDTH, axes_height]
    )
    card_ax.set_xlim(0, CARD_WIDTH)
    card_ax.set_ylim(0, height)
    card_ax.set_aspect("equal")
    card_ax.axis("off")
    card = mpatches.FancyBboxPatch(
        (0, 0),
        CARD_WIDTH,
        height,
        boxstyle=f"round,pad=0,rounding_size={CARD_CORNER_RADIUS}",
        # Both cards stay opaque white over their respective risk bands.
        facecolor="white",
        edgecolor=color,
        linewidth=1.0,
        clip_on=False,
        zorder=0,
    )
    card_ax.add_patch(card)
    return card_ax


def _draw_intervention_card(ax, *, centre_y):
    card_ax = _add_card_axes(
        ax,
        centre_y=centre_y,
        height=INTERVENTION_CARD_HEIGHT,
        color=INTERVENTION_COLOR,
    )
    centre_x = CARD_WIDTH / 2
    text_kwargs: dict[str, Any] = dict(
        color=INTERVENTION_COLOR, fontweight="bold", ha="center", va="center"
    )

    _draw_warning_sign(card_ax, centre=(centre_x, 7.05))
    card_ax.text(centre_x, 5.35, "RISK IS ELEVATED", fontsize=11.5, **text_kwargs)
    card_ax.plot(
        [0.75, CARD_WIDTH - 0.75],
        [4.65, 4.65],
        color=INTERVENTION_COLOR,
        linewidth=0.6,
        solid_capstyle="round",
    )

    icon_y = 2.95
    _draw_magnifying_glass(card_ax, centre=(1.65, icon_y), scale=ICON_SCALE)
    _draw_plus(card_ax, centre=(3.25, icon_y))
    _draw_crossed_out_mosquito(card_ax, centre=(centre_x, icon_y), scale=ICON_SCALE)
    _draw_plus(card_ax, centre=(6.75, icon_y))
    _draw_long_sleeved_shirt(card_ax, centre=(8.3, icon_y), scale=ICON_SCALE)

    card_ax.text(centre_x, 0.95, "MAINTAIN INTERVENTIONS", fontsize=10, **text_kwargs)


def _draw_warning_sign(ax, *, centre):
    # Round the triangle itself, not just the stroke joins, for smooth inner corners.
    origin = np.array(centre)
    vertices = np.array(
        [
            (-0.13, 0.99),
            (0.0, 1.22),
            (0.13, 0.99),
            (1.1, -0.69),
            (1.23, -0.92),
            (0.97, -0.92),
            (-0.97, -0.92),
            (-1.23, -0.92),
            (-1.1, -0.69),
            (-0.13, 0.99),
            (-0.13, 0.99),
        ]
    )
    ax.add_patch(
        mpatches.PathPatch(
            Path(
                origin + 1.1 * vertices,
                [Path.MOVETO]
                + [Path.CURVE3, Path.CURVE3, Path.LINETO] * 3
                + [Path.CLOSEPOLY],
            ),
            facecolor="none",
            edgecolor=INTERVENTION_COLOR,
            linewidth=2.6,
            joinstyle="round",
        )
    )
    x, y = centre
    ax.plot(
        [x, x],
        [y - 0.10, y + 0.50],
        color=INTERVENTION_COLOR,
        linewidth=3.2,
        solid_capstyle="round",
    )
    ax.add_patch(
        mpatches.Circle(
            (x, y - 0.54), 0.11, facecolor=INTERVENTION_COLOR, edgecolor="none"
        )
    )


def _draw_magnifying_glass(ax, *, centre, scale):
    x, y = centre
    lens_x, lens_y, lens_radius = x - 0.22 * scale, y + 0.27 * scale, 0.58 * scale
    ax.add_patch(
        mpatches.Circle(
            (lens_x, lens_y),
            lens_radius,
            facecolor="none",
            edgecolor=INTERVENTION_COLOR,
            linewidth=ICON_LINEWIDTH,
        )
    )
    neck_r_vec = scale * np.array([0.58, 0.88]) / np.sqrt(2)
    ax.plot(
        lens_x + neck_r_vec,
        lens_y - neck_r_vec,
        color=INTERVENTION_COLOR,
        linewidth=ICON_LINEWIDTH,
        solid_capstyle="round",
    )
    handle_r_vec = scale * np.array([0.88, 1.45]) / np.sqrt(2)
    ax.plot(
        lens_x + handle_r_vec,
        lens_y - handle_r_vec,
        color=INTERVENTION_COLOR,
        linewidth=4.0,
        solid_capstyle="round",
    )


def _draw_plus(ax, *, centre, half_length=0.22):
    x, y = centre
    line_kwargs: dict[str, Any] = dict(
        color=INTERVENTION_COLOR,
        linewidth=1.4,
        solid_capstyle="round",
    )
    ax.plot([x - half_length, x + half_length], [y, y], **line_kwargs)
    ax.plot([x, x], [y - half_length, y + half_length], **line_kwargs)


def _draw_crossed_out_mosquito(ax, *, centre, scale):
    origin = np.array(centre)
    ring_radius = 0.95 * scale

    # Bend the leg tips inward for rim clearance without shrinking the insect.
    left_legs = [
        [(-0.06, 0.28), (-0.32, 0.40), (-0.34, 0.55), (-0.34, 0.64), (-0.38, 0.68)],
        [(-0.08, 0.18), (-0.40, 0.12), (-0.48, -0.13), (-0.54, -0.28), (-0.68, -0.35)],
        [(-0.07, 0.08), (-0.23, -0.17), (-0.24, -0.40), (-0.25, -0.57), (-0.39, -0.68)],
    ]
    for side in (-1, 1):
        side_scale = scale * np.array([side, 1])
        for vertices in left_legs:
            ax.add_patch(
                mpatches.PathPatch(
                    Path(
                        origin + side_scale * np.array(vertices),
                        [Path.MOVETO] + [Path.CURVE3] * 4,
                    ),
                    facecolor="none",
                    edgecolor=MOSQUITO_COLOR,
                    linewidth=0.8,
                    capstyle="round",
                    zorder=1,
                )
            )

        ax.add_patch(
            mpatches.PathPatch(
                Path(
                    origin
                    + side_scale * np.array([(0.03, 0.46), (0.10, 0.56), (0.16, 0.62)]),
                    [Path.MOVETO, Path.CURVE3, Path.CURVE3],
                ),
                facecolor="none",
                edgecolor=MOSQUITO_COLOR,
                linewidth=0.6,
                capstyle="round",
                zorder=1,
            )
        )
        wing_vertices = np.array(
            [
                (-0.06, 0.23),
                (-0.32, 0.25),
                (-0.86, 0.08),
                (-0.73, -0.08),
                (-0.63, -0.19),
                (-0.24, 0.00),
                (-0.06, 0.12),
                (-0.06, 0.23),
            ]
        )
        ax.add_patch(
            mpatches.PathPatch(
                Path(
                    origin + side_scale * wing_vertices,
                    [Path.MOVETO] + [Path.CURVE4] * 6 + [Path.CLOSEPOLY],
                ),
                facecolor="#D9D9D9",
                edgecolor=MOSQUITO_COLOR,
                linewidth=0.65,
                zorder=2,
            )
        )
    ax.plot(
        [origin[0], origin[0]],
        origin[1] + scale * np.array([0.45, 0.79]),
        color=MOSQUITO_COLOR,
        linewidth=0.7,
        solid_capstyle="round",
        zorder=1,
    )
    for body_centre, width, height in [
        ((0.0, -0.23), 0.14, 0.78),  # abdomen
        ((0.0, 0.2), 0.21, 0.3),  # thorax
        ((0.0, 0.4), 0.16, 0.16),  # head
    ]:
        ax.add_patch(
            mpatches.Ellipse(
                origin + scale * np.array(body_centre),
                width=width * scale,
                height=height * scale,
                facecolor=MOSQUITO_COLOR,
                edgecolor="none",
                zorder=3,
            )
        )

    # Clipping keeps the thick slash inside the circular rim.
    ring = mpatches.Circle(
        centre,
        ring_radius,
        facecolor="none",
        edgecolor=INTERVENTION_COLOR,
        linewidth=2.3,
        zorder=5,
    )
    ax.add_patch(ring)
    slash = origin + ring_radius / np.sqrt(2) * np.array([(-1, 1), (1, -1)])
    (slash_line,) = ax.plot(
        slash[:, 0],
        slash[:, 1],
        color=INTERVENTION_COLOR,
        linewidth=2.6,
        solid_capstyle="butt",
        zorder=4,
    )
    slash_line.set_clip_path(ring)


def _draw_long_sleeved_shirt(ax, *, centre, scale):
    origin = np.array(centre)
    move, line, curve, close = Path.MOVETO, Path.LINETO, Path.CURVE3, Path.CLOSEPOLY
    outline_vertices = np.array(
        [
            (-0.26, 0.83),
            (-0.54, 0.75),
            (-0.66, 0.72),
            (-0.69, 0.60),
            (-1.00, -0.60),
            (-0.75, -0.67),
            (-0.55, 0.12),
            (-0.55, -0.72),
            (-0.55, -0.79),
            (-0.46, -0.80),
            (0.0, -0.87),
            (0.46, -0.80),
            (0.55, -0.79),
            (0.55, -0.72),
            (0.55, 0.12),
            (0.75, -0.67),
            (1.00, -0.60),
            (0.69, 0.60),
            (0.66, 0.72),
            (0.54, 0.75),
            (0.26, 0.83),
            (0.0, 0.53),
            (-0.26, 0.83),
            (-0.26, 0.83),
        ]
    )
    outline_codes = [
        move,
        line,
        curve,
        curve,
        line,
        line,
        line,
        line,
        curve,
        curve,
        curve,
        curve,
        curve,
        curve,
        line,
        line,
        line,
        line,
        curve,
        curve,
        line,
        curve,
        curve,
        close,
    ]
    ax.add_patch(
        mpatches.PathPatch(
            Path(origin + scale * outline_vertices, outline_codes),
            facecolor="none",
            edgecolor=INTERVENTION_COLOR,
            linewidth=1.6,
            joinstyle="round",
        )
    )

    collar_rib = [(-0.33, 0.81), (0.0, 0.36), (0.33, 0.81)]
    hem_band = [(-0.55, -0.67), (0.0, -0.76), (0.55, -0.67)]
    for vertices in (collar_rib, hem_band):
        ax.add_patch(
            mpatches.PathPatch(
                Path(origin + scale * np.array(vertices), [move, curve, curve]),
                facecolor="none",
                edgecolor=INTERVENTION_COLOR,
                linewidth=0.9,
                capstyle="round",
            )
        )
    for side in (-1, 1):
        cuff = origin + scale * np.array([side, 1]) * [(0.965, -0.47), (0.717, -0.54)]
        ax.plot(cuff[:, 0], cuff[:, 1], color=INTERVENTION_COLOR, linewidth=0.9)


def _draw_safe_label(ax, *, centre_y):
    label_ax = _add_card_axes(
        ax,
        centre_y=centre_y,
        height=SAFE_LABEL_HEIGHT,
        color=SAFE_COLOR,
    )
    label_ax.text(
        CARD_WIDTH / 2,
        SAFE_LABEL_HEIGHT / 2,
        "SAFE TO RELAX INTERVENTIONS",
        color=SAFE_COLOR,
        fontsize=9,
        fontweight="bold",
        ha="center",
        va="center",
    )


def _draw_inference_arrows(fig):
    # Dashed connector from the incidence and serial-interval panels into the R_t panel.
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
    # Grey block arrow between two points in figure coordinates.
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
    # Block arrows from each input panel into the output panel.
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
