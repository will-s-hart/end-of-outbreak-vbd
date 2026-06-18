import argparse

from endoutbreakvbd.utils import set_plot_config
from scripts.inputs import get_inputs_lazio_epiestim
from scripts.lazio_outbreak_plots import (
    _make_declaration_plot,
    _make_rep_no_plot,
    _make_risk_plot,
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
    _make_risk_plot(
        doy_vec=inputs["doy_vec"],
        incidence_vec=inputs["incidence_vec"],
        model_names=model_names,
        existing_declarations=inputs["existing_declarations"],
        data_paths=data_paths,
        save_path=inputs["fig_paths"]["risk"],
    )
    _make_declaration_plot(
        incidence_vec=inputs["incidence_vec"],
        model_names=model_names,
        existing_declarations=inputs["existing_declarations"],
        data_paths=data_paths,
        save_path=inputs["fig_paths"]["declaration"],
    )


if __name__ == "__main__":
    argparse.ArgumentParser().parse_args()
    make_plots()
