import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd.additional_case_prob import calc_decision_delay
from endoutbreakvbd.utils import ordered_legend, set_plot_config
from scripts.inputs import get_inputs_inference_test
from scripts.lazio_outbreak_plots import (
    _make_decision_plot,
    _make_prob_plot,
    _make_rep_no_plot,
    _make_scaling_factor_plot,
    _make_suitability_plot,
)


def make_plots(quasi_real_time=False):
    set_plot_config()
    inputs = get_inputs_inference_test(quasi_real_time=quasi_real_time)
    df_data = pd.read_csv(inputs["results_paths"]["outbreak_data"], index_col=0)
    doy_vec = df_data["day_of_year"].to_numpy()
    # The final table row is the projected decision day, where incidence is unknown (NaN).
    # Plotting utilities accept a day axis one longer than the observed incidence series.
    incidence_vec = df_data["cases"].dropna().to_numpy(dtype=int)
    suitability_vec = df_data["suitability"].to_numpy()
    suitability_mean_vec = df_data["suitability_mean"].to_numpy()
    rep_no_factor_vec = df_data["rep_no_factor"].to_numpy()
    rep_no_vec = df_data["reproduction_number"].to_numpy()
    prob_vec = df_data["additional_case_prob"].to_numpy()

    for plot_func, plot_kwargs, actual_vec, legend_loc, save_path in [
        (
            _make_suitability_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "suitability_mean_vec": suitability_mean_vec,
                "data_path": inputs["results_paths"]["suitability"],
            },
            suitability_vec,
            None,
            inputs["fig_paths"]["suitability"],
        ),
        (
            _make_scaling_factor_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "data_path": inputs["results_paths"]["suitability"],
            },
            rep_no_factor_vec,
            None,
            inputs["fig_paths"]["scaling_factor"],
        ),
        (
            _make_rep_no_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "model_names": ["Suitability-based", "Autoregressive"],
                "data_paths": [
                    inputs["results_paths"]["suitability"],
                    inputs["results_paths"]["autoregressive"],
                ],
            },
            rep_no_vec,
            None,
            inputs["fig_paths"]["rep_no"],
        ),
        (
            _make_prob_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "model_names": ["Suitability-based", "Autoregressive"],
                "existing_decisions": None,
                "data_paths": [
                    inputs["results_paths"]["suitability"],
                    inputs["results_paths"]["autoregressive"],
                ],
            },
            prob_vec,
            "lower center",
            inputs["fig_paths"]["additional_case_prob"],
        ),
    ]:
        fig, ax = plot_func(**plot_kwargs)
        ax.set_xlabel("Date")
        ax.plot(doy_vec, actual_vec, color="black", label="True")
        ordered_legend(ax, {"True": 0, "Seasonal prior": 1}, loc=legend_loc)
        fig.savefig(save_path)
    perc_risk_thresholds = inputs["perc_risk_threshold_grid"]
    fig, ax = _make_decision_plot(
        incidence_vec=incidence_vec,
        model_names=["Suitability-based", "Autoregressive"],
        existing_decisions=None,
        data_paths=[
            inputs["results_paths"]["suitability"],
            inputs["results_paths"]["autoregressive"],
        ],
        perc_risk_thresholds=perc_risk_thresholds,
    )
    time_final_case = np.nonzero(incidence_vec)[0][-1]
    prob_days = np.arange(time_final_case + 1, len(incidence_vec))
    prob_vals = df_data["additional_case_prob"].to_numpy()[prob_days]
    decision_delays = calc_decision_delay(
        prob_vec=prob_vals,
        days=prob_days,
        perc_risk_threshold=perc_risk_thresholds,
        time_final_case=time_final_case,
    )
    ax.plot(perc_risk_thresholds, decision_delays, color="black", label="True")
    ordered_legend(ax, {"True": 0, "Seasonal prior": 1})
    fig.savefig(inputs["fig_paths"]["decision"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-q",
        "--quasi-real-time",
        action="store_true",
        help="Perform quasi-real-time analyses",
    )
    args = parser.parse_args()
    make_plots(quasi_real_time=args.quasi_real_time)
