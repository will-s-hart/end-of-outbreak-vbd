"""Simulation study for the under-reporting model.

Simulates one true outbreak from a known declining reproduction number, thins it to a reported
series at a fixed reporting probability, and contrasts end-of-outbreak analyses against the
truth:

- the under-reporting offshoot with the reproduction number inferred (est. R);
- the under-reporting offshoot with the reproduction number fixed to the truth (true R) — this
  isolates the latent-case (reporting) inference from reproduction-number estimation error after
  the final case;
- a naive analysis that treats the reported cases as complete.

The reporting probability is chosen so that, in this realisation, true cases continue after the
final reported case, which the naive analysis misses.
"""

import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd import calc_additional_case_prob_analytical
from endoutbreakvbd.inference import fit_autoregressive_model, fit_known_rep_no_model
from endoutbreakvbd.model import simulate_outbreak
from scripts.inputs import get_inputs_sim_underreporting
from scripts.sim_underreporting_plots import make_plots


def run_analyses():
    inputs = get_inputs_sim_underreporting()
    serial_interval_dist_vec = inputs["serial_interval_dist_vec"]
    reporting_prob = inputs["reporting_prob"]

    # A single shared random number generator threads through the simulation and every fit.
    rng = np.random.default_rng(inputs["seed"])
    true_vec = _simulate_true_outbreak(
        rng,
        serial_interval_dist_vec,
        inputs["min_outbreak_size"],
        inputs["incidence_init"],
    )
    reported_vec = rng.binomial(true_vec, reporting_prob)
    reported_vec[0] = true_vec[0]  # fixed, fully reported index case
    # The fits report every day plus one projected day past the data (the current-day risk), so
    # the output spans 0..len(reported_vec). The simulated truth is known to be zero there, while
    # reported and inferred incidence are unobserved and reindex to NaN.
    t_calc_vec = np.arange(len(true_vec) + 1)

    # Under-reporting model with the reproduction number inferred (autoregressive).
    posterior_ds = fit_autoregressive_model(
        incidence=reported_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        reporting_prob=reporting_prob,
        rng=rng,
        compute_diagnostics=True,
        raise_on_poor_diagnostics=False,
    )
    # Under-reporting model with the reproduction number fixed to the truth.
    known_r_posterior_ds = fit_known_rep_no_model(
        incidence=reported_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_func=_true_rep_no,
        reporting_prob=reporting_prob,
        rng=rng,
        compute_diagnostics=False,
    )
    # Naive analysis: reported cases treated as complete, reproduction number inferred.
    naive_posterior_ds = fit_autoregressive_model(
        incidence=reported_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rng=rng,
        compute_diagnostics=False,
    )
    true_prob_vec = calc_additional_case_prob_analytical(
        incidence=true_vec,
        rep_no_func=_true_rep_no,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=t_calc_vec,
    )

    output_df = pd.DataFrame(
        {
            "day_of_outbreak": t_calc_vec,
            "true_incidence": np.append(true_vec, 0),
            "reported_incidence": np.append(reported_vec, np.nan),
            **{
                f"incidence_{stat}": posterior_ds[f"incidence_{stat}"]
                .reindex(data_time=t_calc_vec)
                .values
                for stat in ("mean", "lower", "upper")
            },
            "rep_no_true": _true_rep_no(t_calc_vec),
            "rep_no_mean": posterior_ds["rep_no_mean"].values,
            "rep_no_lower": posterior_ds["rep_no_lower"].values,
            "rep_no_upper": posterior_ds["rep_no_upper"].values,
            "rep_no_naive_mean": naive_posterior_ds["rep_no_mean"].values,
            "additional_case_prob_true": true_prob_vec,
            "additional_case_prob_est_r": posterior_ds["additional_case_prob"].values,
            "additional_case_prob_est_r_lower": posterior_ds[
                "additional_case_prob_lower"
            ].values,
            "additional_case_prob_est_r_upper": posterior_ds[
                "additional_case_prob_upper"
            ].values,
            "additional_case_prob_known_r": known_r_posterior_ds[
                "additional_case_prob"
            ].values,
            "additional_case_prob_known_r_lower": known_r_posterior_ds[
                "additional_case_prob_lower"
            ].values,
            "additional_case_prob_known_r_upper": known_r_posterior_ds[
                "additional_case_prob_upper"
            ].values,
            "additional_case_prob_naive": naive_posterior_ds[
                "additional_case_prob"
            ].values,
        }
    ).set_index("day_of_outbreak")
    output_df.to_csv(inputs["results_paths"]["sim"])
    pd.Series(posterior_ds.attrs["diagnostics"], name="value").rename_axis(
        "stat"
    ).to_csv(inputs["results_paths"]["diagnostics"])


def _true_rep_no(t):
    """Known declining reproduction-number profile used to generate the synthetic truth.

    A slow linear decline (crossing 1 around day 75, floored at 0.3 by day ~110) so that,
    with the long chikungunya serial interval (mean ~12.5 days), the outbreak grows over
    several generations before dying out.
    """
    t_arr = np.asarray(t, dtype=float)
    rep_no = np.clip(2.5 - 0.02 * t_arr, 0.3, 2.5)
    return float(rep_no) if np.ndim(t) == 0 else rep_no


def _simulate_true_outbreak(
    rng, serial_interval_dist_vec, min_size, incidence_init, max_attempts=20000
):
    """Simulate the study's true outbreak (declining R), padded with zeros, of at least ``min_size``."""
    true_vec = simulate_outbreak(
        rep_no_func=_true_rep_no,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rng=rng,
        min_size=min_size,
        incidence_init=incidence_init,
        t_stop=200,
        max_attempts=max_attempts,
    )
    return np.concatenate(
        [true_vec, np.zeros(len(serial_interval_dist_vec) + 1, dtype=int)]
    )


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
