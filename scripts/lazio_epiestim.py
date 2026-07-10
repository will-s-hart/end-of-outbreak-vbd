import argparse

import numpy as np
import pandas as pd
import rpy2.robjects as ro
import scipy.stats
from rpy2.robjects import pandas2ri
from rpy2.robjects.packages import importr
from rpy2.robjects.vectors import FloatVector, IntVector

from endoutbreakvbd import calc_additional_case_prob_analytical
from endoutbreakvbd._types import IntArray, RepNoOutput
from endoutbreakvbd.rep_no_models import Defaults
from endoutbreakvbd.utils import (
    lognormal_params_from_median_percentile_2_5,
    rep_no_from_grid,
)
from scripts.inputs import get_inputs_lazio_epiestim
from scripts.lazio_epiestim_plots import make_plots


def run_analyses():
    inputs = get_inputs_lazio_epiestim()
    incidence_vec = inputs["incidence_vec"]
    serial_interval_dist_vec = inputs["serial_interval_dist_vec"]
    rng = np.random.default_rng(2)

    utils = importr("utils")
    utils.chooseCRANmirror(ind=1)
    utils.install_packages("EpiEstim")
    epiestim = importr("EpiEstim")

    incid_epiestim = IntVector(incidence_vec)
    # EpiEstim expects the first element of the serial interval vector to give the
    # probability of a 0-day serial interval, which is (and must be) 0.
    si_distr_epiestim = FloatVector(np.append(np.zeros(1), serial_interval_dist_vec))

    rep_no_prior_lognormal_params = lognormal_params_from_median_percentile_2_5(
        median=Defaults().rep_no_prior_median,
        percentile_2_5=Defaults().rep_no_prior_percentile_2_5,
    )
    rep_no_prior_lognormal = scipy.stats.lognorm(
        s=rep_no_prior_lognormal_params["sigma"],
        scale=np.exp(rep_no_prior_lognormal_params["mu"]),
    )
    config = epiestim.make_config(
        si_distr=si_distr_epiestim,
        mean_prior=float(rep_no_prior_lognormal.mean()),
        std_prior=float(rep_no_prior_lognormal.std()),
    )
    epiestim_result = epiestim.estimate_R(
        incid=incid_epiestim, method="non_parametric_si", config=config
    )
    with (ro.default_converter + pandas2ri.converter).context():
        df_from_epiestim = ro.conversion.get_conversion().rpy2py(epiestim_result[0])

    # R_t is not estimated for the first sliding window, so prepend NaNs to align the
    # estimates with the incidence vector.
    lead = np.full(7, np.nan)
    rep_no_mean_vec = np.append(lead, df_from_epiestim["Mean(R)"].to_numpy())
    rep_no_std_vec = np.append(lead, df_from_epiestim["Std(R)"].to_numpy())

    # EpiEstim's per-window posterior for R is a gamma distribution, so the reported
    # mean and standard deviation fully determine it.
    shape = rep_no_mean_vec**2 / rep_no_std_vec**2
    scale = rep_no_std_vec**2 / rep_no_mean_vec
    rep_no_lower_vec = scipy.stats.gamma.ppf(0.025, a=shape, scale=scale)
    rep_no_upper_vec = scipy.stats.gamma.ppf(0.975, a=shape, scale=scale)

    # Sample from the gamma posterior to compute the probability of additional cases.
    rep_no_sample_mat = rng.gamma(
        shape=shape[:, None], scale=scale[:, None], size=(len(shape), 4000)
    )

    def rep_no_func(t: int | IntArray) -> RepNoOutput:
        return rep_no_from_grid(t, rep_no_grid=rep_no_sample_mat, periodic=False)

    # Report one projected day past the data (the current-day risk), matching the model fits.
    # EpiEstim gives no R_t estimate there, so pad it with NaN; the additional-case probability is
    # still defined (0 for that day — no possible future sources).
    n_out = len(incidence_vec) + 1
    prob_vec = calc_additional_case_prob_analytical(
        incidence_vec=incidence_vec,
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=np.arange(n_out),
    )

    df_out = pd.DataFrame(
        {
            "day_of_outbreak": np.arange(n_out),
            "reproduction_number_mean": np.append(rep_no_mean_vec, np.nan),
            "reproduction_number_lower": np.append(rep_no_lower_vec, np.nan),
            "reproduction_number_upper": np.append(rep_no_upper_vec, np.nan),
            "additional_case_prob": prob_vec,
        }
    ).set_index("day_of_outbreak")
    df_out.to_csv(inputs["results_paths"]["epiestim"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    args = parser.parse_args()
    run_analyses()
    if not args.results_only:
        make_plots()
