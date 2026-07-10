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
    "D": "D. Output: probability of additional cases",
}

FIG_WIDTH = 13.0
FIG_HEIGHT = 9.0
ASPECT = FIG_WIDTH / FIG_HEIGHT

LEFT_MARGIN = 0.035
RIGHT_MARGIN = 0.035
TOP_MARGIN = 0.025
BOTTOM_MARGIN = 0.04
H_GAP = 0.06
V_GAP = 0.11
TITLE_BAR_H = 0.05
ROUNDING = 0.010

PANEL_H = (1 - TOP_MARGIN - BOTTOM_MARGIN - V_GAP) / 2
LEFT_RIGHT_RATIO = 1.0
_total_w = 1 - LEFT_MARGIN - RIGHT_MARGIN - H_GAP
PANEL_W_RIGHT = _total_w / (1 + LEFT_RIGHT_RATIO)
PANEL_W_LEFT = _total_w - PANEL_W_RIGHT

PANEL_BBOX = {
    "A": (LEFT_MARGIN, BOTTOM_MARGIN + PANEL_H + V_GAP, PANEL_W_LEFT, PANEL_H),
    "B": (
        LEFT_MARGIN + PANEL_W_LEFT + H_GAP,
        BOTTOM_MARGIN + PANEL_H + V_GAP,
        PANEL_W_RIGHT,
        PANEL_H,
    ),
    "C": (LEFT_MARGIN, BOTTOM_MARGIN, PANEL_W_LEFT, PANEL_H),
    "D": (
        LEFT_MARGIN + PANEL_W_LEFT + H_GAP,
        BOTTOM_MARGIN,
        PANEL_W_RIGHT,
        PANEL_H,
    ),
}

AX_PAD_LEFT = 0.045
AX_PAD_RIGHT = 0.020
AX_PAD_BOTTOM = 0.065
AX_PAD_TOP = 0.010

DOY_MIN = 91
DOY_MAX = 335
RISK_THRESHOLD = 0.05


def make_plots():
    set_plot_config()
    inputs = get_inputs_schematic()
    df = pd.read_csv(inputs["results_paths"]["outbreak"], index_col="day_of_outbreak")
    doy_start = int(df["day_of_year"].iloc[0])
    incidence_vec = df["cases"].to_numpy()
    final_case_doy = doy_start + int(np.nonzero(incidence_vec)[0][-1])
    current_day_doy = final_case_doy + inputs["current_day_offset"]

    fig = plt.figure(figsize=(FIG_WIDTH, FIG_HEIGHT))

    for key in ("A", "B", "C", "D"):
        _decorate_panel(fig, key)

    _populate_panel_a(
        _add_panel_axes(fig, "A"),
        doy_start=doy_start,
        incidence_vec=incidence_vec,
        current_day_doy=current_day_doy,
    )
    _populate_panel_b(
        _add_panel_axes(fig, "B"),
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
    )
    _populate_panel_c(
        _add_panel_axes(fig, "C"),
        df=df,
        seasonal_full=inputs["seasonal_full"],
    )
    _populate_panel_d(
        _add_panel_axes(fig, "D"),
        df=df,
        current_day_doy=current_day_doy,
        intervention_graphic_path=inputs["intervention_graphic_path"],
        safe_graphic_path=inputs["safe_graphic_path"],
    )

    _draw_inference_arrows(fig)
    _draw_output_arrows(fig)

    fig.savefig(inputs["fig_path"], dpi=300)
    plt.close(fig)


def _rounded_box_path(x, y, w, h, *, r_top, r_bot):
    rx_top = r_top
    ry_top = r_top * ASPECT
    rx_bot = r_bot
    ry_bot = r_bot * ASPECT
    verts = [
        (x + rx_top, y + h),
        (x + w - rx_top, y + h),
        (x + w, y + h),
        (x + w, y + h - ry_top),
        (x + w, y + ry_bot),
        (x + w, y),
        (x + w - rx_bot, y),
        (x + rx_bot, y),
        (x, y),
        (x, y + ry_bot),
        (x, y + h - ry_top),
        (x, y + h),
        (x + rx_top, y + h),
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
    return Path(verts, codes)


def _decorate_panel(fig, panel_key):
    px, py, pw, ph = PANEL_BBOX[panel_key]
    color = PANEL_COLORS[panel_key]
    title = PANEL_TITLES[panel_key]

    title_y = py + ph - TITLE_BAR_H
    title_path = _rounded_box_path(
        px, title_y, pw, TITLE_BAR_H, r_top=ROUNDING, r_bot=0.0
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

    border_path = _rounded_box_path(px, py, pw, ph, r_top=ROUNDING, r_bot=ROUNDING)
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
        px + pw / 2,
        title_y + TITLE_BAR_H / 2,
        title,
        color="white",
        fontsize=14,
        fontweight="bold",
        ha="center",
        va="center",
        zorder=11,
    )


def _add_panel_axes(fig, panel_key, *, left_pad=AX_PAD_LEFT):
    px, py, pw, ph = PANEL_BBOX[panel_key]
    return fig.add_axes(
        [
            px + left_pad,
            py + AX_PAD_BOTTOM,
            pw - left_pad - AX_PAD_RIGHT,
            ph - TITLE_BAR_H - AX_PAD_TOP - AX_PAD_BOTTOM,
        ]
    )


def _populate_panel_a(ax, *, doy_start, incidence_vec, current_day_doy):
    doy = doy_start + np.arange(incidence_vec.size)
    mask = doy <= DOY_MAX
    ax.bar(
        doy[mask],
        incidence_vec[mask],
        color=PANEL_COLORS["A"],
        width=1.0,
        align="center",
    )

    ymax = max(int(incidence_vec.max()) * 1.18, 1)
    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylim(0, ymax)

    ax.axvline(current_day_doy, color="black", linestyle="--", linewidth=1.5)
    ax.text(
        current_day_doy - 3,
        ymax * 0.93,
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
    days = np.arange(serial_interval_dist_vec.size)
    ax.bar(
        days,
        serial_interval_dist_vec,
        color=PANEL_COLORS["B"],
        width=1.0,
        align="center",
    )
    ax.set_xlim(-0.5, days[-1] + 0.5)
    ax.set_ylim(0, serial_interval_dist_vec.max() * 1.18)
    ax.set_xlabel("Serial interval (days)", labelpad=14)
    ax.set_ylabel("Probability")
    ax.set_xticks([])
    ax.set_yticks([])


def _populate_panel_c(ax, *, df, seasonal_full):
    doy_inf = df["day_of_year"].to_numpy()
    mean_inf = df["reproduction_number_mean"].to_numpy()
    lower_inf = df["reproduction_number_lower"].to_numpy()
    upper_inf = df["reproduction_number_upper"].to_numpy()

    in_range = doy_inf <= DOY_MAX
    doy_inf = doy_inf[in_range]
    mean_inf = mean_inf[in_range]
    lower_inf = lower_inf[in_range]
    upper_inf = upper_inf[in_range]

    doy_grid_seasonal = np.arange(DOY_MIN, DOY_MAX + 1)
    seasonal = seasonal_full[doy_grid_seasonal - 1]

    fill_handle = ax.fill_between(
        doy_inf,
        lower_inf,
        upper_inf,
        color=PANEL_COLORS["C"],
        alpha=0.35,
        linewidth=0,
    )
    (seasonal_handle,) = ax.plot(
        doy_grid_seasonal, seasonal, color="black", linewidth=2
    )
    (inferred_handle,) = ax.plot(
        doy_inf, mean_inf, color=PANEL_COLORS["C"], linewidth=2
    )

    ax.set_xlim(DOY_MIN, DOY_MAX)
    ax.set_ylim(0, upper_inf.max() * 1.05)
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
    df,
    current_day_doy,
    intervention_graphic_path,
    safe_graphic_path,
):
    doy_inf = df["day_of_year"].to_numpy()
    prob = df["additional_case_prob"].to_numpy()
    x_lower = 305  # 1 November (non-leap year)
    mask = (doy_inf >= x_lower) & (doy_inf <= current_day_doy)

    y_upper = float(np.ceil(prob[mask].max() * 40) / 40)

    ax.axhspan(0, RISK_THRESHOLD, color="#9CCC65", alpha=0.45, zorder=0)
    ax.axhspan(RISK_THRESHOLD, 1.0, color="#FFE082", alpha=0.45, zorder=0)
    ax.axhline(RISK_THRESHOLD, color="#558B2F", linestyle=":", linewidth=1, zorder=1)
    ax.plot(doy_inf[mask], prob[mask], color=PANEL_COLORS["D"], linewidth=2)
    ax.axvline(current_day_doy, color="black", linestyle="--", linewidth=1.5)

    ax.set_xlim(x_lower, DOY_MAX)
    ax.set_ylim(0, y_upper)
    ax.set_xlabel("Date")
    ax.set_ylabel("Probability of additional\ncases on/after date")
    ax.set_yticks([])

    ax.text(
        current_day_doy - 0.5,
        y_upper * 0.93,
        "Current day",
        ha="right",
        va="top",
        fontsize=12,
    )
    intervention_img = mpimg.imread(intervention_graphic_path)
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(intervention_img, zoom=0.18),
            (0.7, 0.56),
            xycoords="axes fraction",
            frameon=False,
        )
    )
    safe_img = mpimg.imread(safe_graphic_path)
    ax.add_artist(
        AnnotationBbox(
            OffsetImage(safe_img, zoom=0.75 * 0.18),
            (0.71, 0.06),
            xycoords="axes fraction",
            frameon=False,
        )
    )

    month_start_xticks(ax, year=2017, interval_months=1)
    ax.tick_params(axis="x", labelsize=12)


def _draw_inference_arrows(fig):
    a_x, a_y, a_w, _ = PANEL_BBOX["A"]
    b_x, b_y, b_w, _ = PANEL_BBOX["B"]
    c_x, c_y, c_w, c_h = PANEL_BBOX["C"]

    a_col_x = a_x + a_w * 0.5
    b_col_x = b_x + b_w * 0.5
    c_top_y = c_y + c_h
    rail_y = (a_y + c_top_y) / 2

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
        mpatches.FancyArrowPatch((a_col_x, a_y), (a_col_x, rail_y), **line_kwargs)
    )
    fig.patches.append(
        mpatches.FancyArrowPatch((b_col_x, b_y), (b_col_x, rail_y), **line_kwargs)
    )
    fig.patches.append(
        mpatches.FancyArrowPatch((a_col_x, rail_y), (b_col_x, rail_y), **line_kwargs)
    )
    fig.patches.append(
        mpatches.FancyArrowPatch(
            (a_col_x, rail_y),
            (a_col_x, c_top_y),
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
        a_col_x - 0.015,
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
    a_x, a_y, a_w, a_h = PANEL_BBOX["A"]
    b_x, b_y, b_w, _ = PANEL_BBOX["B"]
    c_x, c_y, c_w, c_h = PANEL_BBOX["C"]
    d_x, d_y, d_w, d_h = PANEL_BBOX["D"]

    _block_arrow(
        fig,
        start=(c_x + c_w, c_y + c_h * 0.55),
        end=(d_x, d_y + d_h * 0.55),
    )
    _block_arrow(
        fig,
        start=(b_x + b_w * 0.7, b_y),
        end=(d_x + d_w * 0.7, d_y + d_h),
    )
    _block_arrow(
        fig,
        start=(a_x + a_w, a_y),
        end=(d_x, d_y + d_h),
    )


if __name__ == "__main__":
    make_plots()
