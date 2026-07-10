import argparse

from endoutbreakvbd.utils import set_plot_config
from scripts.inputs import get_inputs_lazio_epiestim
from scripts.lazio_outbreak_plots import (
    _make_decision_plot,
    _make_prob_plot,
    _make_rep_no_plot,
)


def make_plots():
    set_plot_config()
    inputs = get_inputs_lazio_epiestim()
    model_names = ["Suitability-based", "EpiEstim"]
    data_paths = [
        inputs["results_paths"]["suitability"],
        inputs["results_paths"]["epiestim"],
    ]
    _make_rep_no_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        model_names=model_names,
        data_paths=data_paths,
        save_path=inputs["fig_paths"]["rep_no"],
    )
    _make_prob_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        model_names=model_names,
        existing_decisions=inputs["existing_decisions"],
        data_paths=data_paths,
        save_path=inputs["fig_paths"]["additional_case_prob"],
    )
    _make_decision_plot(
        incidence_vec=inputs["incidence_vec"],
        model_names=model_names,
        existing_decisions=inputs["existing_decisions"],
        data_paths=data_paths,
        perc_risk_thresholds=inputs["perc_risk_threshold_grid"],
        save_path=inputs["fig_paths"]["decision"],
    )


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    make_plots()
