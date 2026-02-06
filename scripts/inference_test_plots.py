import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd.further_case_risk import calc_declaration_delay
from endoutbreakvbd.inputs import get_inputs_inference_test
from endoutbreakvbd.utils import set_plot_config
from scripts.lazio_outbreak_plots import (
    _make_declaration_plot,
    _make_rep_no_plot,
    _make_risk_plot,
    _make_scaling_factor_plot,
    _make_suitability_plot,
)


def make_plots(quasi_real_time=False):
    set_plot_config()
    inputs = get_inputs_inference_test(quasi_real_time=quasi_real_time)
    df_data = pd.read_csv(inputs["results_paths"]["outbreak_data"], index_col=0)
    doy_vec = df_data["day_of_year"].to_numpy()
    incidence_vec = df_data["cases"].to_numpy()
    suitability_vec = df_data["suitability"].to_numpy()
    suitability_mean_vec = df_data["suitability_mean"].to_numpy()
    rep_no_factor_vec = df_data["rep_no_factor"].to_numpy()
    rep_no_vec = df_data["rep_no"].to_numpy()
    risk_vec = df_data["further_case_risk"].to_numpy()

    for plot_func, plot_kwargs, actual_vec, save_path in [
        (
            _make_suitability_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "suitability_mean_vec": suitability_mean_vec,
                "data_path": inputs["results_paths"]["suitability"],
            },
            suitability_vec,
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
            inputs["fig_paths"]["scaling_factor"],
        ),
        (
            _make_rep_no_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "model_names": ["Autoregressive model", "Suitability model"],
                "data_paths": [
                    inputs["results_paths"]["autoregressive"],
                    inputs["results_paths"]["suitability"],
                ],
            },
            rep_no_vec,
            inputs["fig_paths"]["rep_no"],
        ),
        (
            _make_risk_plot,
            {
                "doy_vec": doy_vec,
                "incidence_vec": incidence_vec,
                "model_names": ["Autoregressive model", "Suitability model"],
                "existing_declarations": None,
                "data_paths": [
                    inputs["results_paths"]["autoregressive"],
                    inputs["results_paths"]["suitability"],
                ],
            },
            risk_vec,
            inputs["fig_paths"]["risk"],
        ),
    ]:
        fig, ax = plot_func(**plot_kwargs)
        ax.plot(doy_vec, actual_vec, color="black", label="True")
        ax.legend()
        fig.savefig(save_path)
    fig, ax = _make_declaration_plot(
        incidence_vec=incidence_vec,
        model_names=["Autoregressive model", "Suitability model"],
        existing_declarations=None,
        data_paths=[
            inputs["results_paths"]["autoregressive"],
            inputs["results_paths"]["suitability"],
        ],
    )
    perc_risk_thresholds = ax.get_lines()[0].get_xdata()
    time_last_case = np.nonzero(incidence_vec)[0][-1]
    risk_days = np.arange(time_last_case + 1, len(incidence_vec))
    risk_vals = df_data["further_case_risk"].to_numpy()[risk_days]
    declaration_delays = calc_declaration_delay(
        risk_vec=risk_vals,
        perc_risk_threshold=perc_risk_thresholds,
        delay_of_first_risk=1,
    )
    ax.plot(perc_risk_thresholds, declaration_delays, color="black", label="True")
    ax.legend()
    fig.savefig(inputs["fig_paths"]["declaration"])


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
