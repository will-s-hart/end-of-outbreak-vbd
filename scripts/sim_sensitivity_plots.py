import matplotlib
import matplotlib.pyplot as plt
import numpy as np

from endoutbreakvbd.utils import month_start_xticks, set_plot_config
from scripts.inputs import get_inputs_sim_sensitivity
from scripts.sim_study_plots import _make_many_outbreak_decision_plot


def make_plots():
    set_plot_config()
    inputs = get_inputs_sim_sensitivity()
    for analysis in (inputs["rep_no_factor"], inputs["decay_speed"]):
        _make_rep_no_comparison_plot(
            curves=analysis["curves"],
            ylim=analysis["curves_ylim"],
            save_path=analysis["curves_fig_path"],
        )
        for run in analysis["runs"]:
            _make_many_outbreak_decision_plot(
                data_path=run["results_path"],
                perc_risk_threshold_vals=inputs[
                    "many_outbreak_perc_risk_threshold_vals"
                ],
                cmap_names=(run["cmap_name"],),
                xlim=(91, 366),
                ylim=(0, 31),
                xtick_interval_months=2,
                save_path=run["fig_path"],
            )


def _make_rep_no_comparison_plot(*, curves, ylim, save_path=None):
    fig, ax = plt.subplots()
    doy_vec = np.arange(1, 366)
    for label, rep_no_func_doy, cmap_name in curves:
        color = matplotlib.colormaps[cmap_name](0.7)
        ax.plot(doy_vec, rep_no_func_doy(doy_vec), label=label, color=color)
    month_start_xticks(ax, interval_months=2)
    ax.set_xlabel("Date")
    ax.set_ylim(*ylim)
    ax.set_ylabel("Time-dependent reproduction number")
    ax.legend()
    if save_path is not None:
        fig.savefig(save_path)
    return fig, ax


if __name__ == "__main__":
    make_plots()
