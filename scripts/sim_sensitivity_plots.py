import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from endoutbreakvbd.utils import month_start_xticks, set_plot_config
from scripts.inputs import get_inputs_sim_sensitivity
from scripts.sim_study_plots import _make_many_outbreak_decision_plot


def make_plots():
    set_plot_config()
    inputs = get_inputs_sim_sensitivity()
    for analysis_spec in (inputs["rep_no_factor"], inputs["decay_speed"]):
        _make_rep_no_comparison_plot(
            curve_specs=analysis_spec["curve_specs"],
            y_limits=analysis_spec["curve_y_limits"],
            fig_path=analysis_spec["curves_fig_path"],
        )
        for run_spec in analysis_spec["run_specs"]:
            _make_many_outbreak_decision_plot(
                results_path=run_spec["results_path"],
                risk_threshold_pct_vals=inputs["many_outbreak_risk_threshold_pct_vals"],
                cmap_names=(run_spec["cmap_name"],),
                x_limits=(91, 366),
                y_limits=(0, 31),
                x_tick_interval_months=2,
                fig_path=run_spec["fig_path"],
            )


def _make_rep_no_comparison_plot(*, curve_specs, y_limits, fig_path=None):
    fig, ax = plt.subplots()
    doy_vec = np.arange(1, 366)
    for label, rep_no_doy_func, cmap_name in curve_specs:
        color = matplotlib.colormaps[cmap_name](0.7)
        ax.plot(doy_vec, rep_no_doy_func(doy_vec), label=label, color=color)
    month_start_xticks(ax, interval_months=2)
    ax.set_xlabel("Date")
    ax.set_ylim(*y_limits)
    ax.set_ylabel("Time-dependent reproduction number")
    ax.legend()
    if fig_path is not None:
        fig.savefig(fig_path)
    return fig, ax


if __name__ == "__main__":
    make_plots()
