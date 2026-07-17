import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd.additional_case_prob import calc_decision_delay
from endoutbreakvbd.utils import ordered_legend, set_plot_config
from scripts.inputs import get_inputs_inference_test
from scripts.lazio_outbreak_plots import (
    _make_additional_case_prob_plot,
    _make_decision_delay_plot,
    _make_rep_no_factor_plot,
    _make_rep_no_plot,
    _make_suitability_plot,
)


def make_plots(quasi_real_time=False):
    set_plot_config()
    inputs = get_inputs_inference_test(quasi_real_time=quasi_real_time)
    outbreak_df = pd.read_csv(inputs["results_paths"]["outbreak_data"], index_col=0)
    doy_vec = outbreak_df["day_of_year"].to_numpy()
    # The final table row is the projected decision day, where incidence is unknown (NaN).
    # Plotting utilities accept a day axis one longer than the observed incidence series.
    incidence_vec = outbreak_df["incidence"].dropna().to_numpy(dtype=int)
    suitability_vec = outbreak_df["suitability"].to_numpy()
    suitability_mean_vec = outbreak_df["suitability_mean"].to_numpy()
    rep_no_factor_vec = outbreak_df["rep_no_factor"].to_numpy()
    rep_no_vec = outbreak_df["reproduction_number"].to_numpy()
    prob_vec = outbreak_df["additional_case_prob"].to_numpy()

    for plot_func, plot_kwargs, truth_vec, legend_loc, fig_path in [
        (
            _make_suitability_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "suitability_mean_vec": suitability_mean_vec,
                "results_path": inputs["results_paths"]["suitability"],
            },
            suitability_vec,
            None,
            inputs["fig_paths"]["suitability"],
        ),
        (
            _make_rep_no_factor_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "results_path": inputs["results_paths"]["suitability"],
            },
            rep_no_factor_vec,
            None,
            inputs["fig_paths"]["rep_no_factor"],
        ),
        (
            _make_rep_no_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "model_names": ["Suitability-based", "Autoregressive"],
                "results_paths": [
                    inputs["results_paths"]["suitability"],
                    inputs["results_paths"]["autoregressive"],
                ],
            },
            rep_no_vec,
            None,
            inputs["fig_paths"]["rep_no"],
        ),
        (
            _make_additional_case_prob_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "model_names": ["Suitability-based", "Autoregressive"],
                "existing_decisions": None,
                "results_paths": [
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
        ax.plot(doy_vec, truth_vec, color="black", label="True")
        ordered_legend(ax, {"True": 0, "Seasonal prior": 1}, loc=legend_loc)
        fig.savefig(fig_path)
    risk_threshold_pct_vec = inputs["risk_threshold_pct_grid"]
    fig, ax = _make_decision_delay_plot(
        incidence_vec=incidence_vec,
        model_names=["Suitability-based", "Autoregressive"],
        existing_decisions=None,
        results_paths=[
            inputs["results_paths"]["suitability"],
            inputs["results_paths"]["autoregressive"],
        ],
        risk_threshold_pct_vec=risk_threshold_pct_vec,
    )
    t_final_case = np.nonzero(incidence_vec)[0][-1]
    t_calc_vec = np.arange(t_final_case + 1, len(incidence_vec))
    prob_vec = outbreak_df["additional_case_prob"].to_numpy()[t_calc_vec]
    decision_delay_vec = calc_decision_delay(
        prob_vec=prob_vec,
        t_vec=t_calc_vec,
        risk_threshold_pct=risk_threshold_pct_vec,
        t_final_case=t_final_case,
    )
    ax.plot(risk_threshold_pct_vec, decision_delay_vec, color="black", label="True")
    ordered_legend(ax, {"True": 0, "Seasonal prior": 1})
    fig.savefig(inputs["fig_paths"]["decision_delay"])


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
