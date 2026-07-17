import argparse

from endoutbreakvbd.utils import set_plot_config
from scripts.inputs import get_inputs_lazio_frozen
from scripts.lazio_outbreak_plots import (
    _make_additional_case_prob_plot,
    _make_decision_delay_plot,
    _make_rep_no_plot,
)


def make_plots():
    set_plot_config()
    inputs = get_inputs_lazio_frozen()
    model_names = ["Suitability-based", "Frozen autoregressive"]
    results_paths = [
        inputs["results_paths"]["suitability"],
        inputs["results_paths"]["autoregressive_frozen"],
    ]
    _make_rep_no_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        model_names=model_names,
        results_paths=results_paths,
        fig_path=inputs["fig_paths"]["rep_no"],
    )
    _make_additional_case_prob_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        model_names=model_names,
        existing_decisions=inputs["existing_decisions"],
        results_paths=results_paths,
        fig_path=inputs["fig_paths"]["additional_case_prob"],
    )
    _make_decision_delay_plot(
        incidence_vec=inputs["incidence_vec"],
        model_names=model_names,
        existing_decisions=inputs["existing_decisions"],
        results_paths=results_paths,
        risk_threshold_pct_vec=inputs["risk_threshold_pct_grid"],
        fig_path=inputs["fig_paths"]["decision_delay"],
    )


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    make_plots()
