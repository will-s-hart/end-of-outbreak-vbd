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
    t_calc = np.arange(len(true_vec) + 1)

    # Under-reporting model with the reproduction number inferred (autoregressive).
    ds = fit_autoregressive_model(
        incidence_vec=reported_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        reporting_prob=reporting_prob,
        rng=rng,
        compute_diagnostics=True,
        raise_on_poor_diagnostics=False,
    )
    # Under-reporting model with the reproduction number fixed to the truth.
    ds_known_r = fit_known_rep_no_model(
        incidence_vec=reported_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_func=_true_rep_no,
        reporting_prob=reporting_prob,
        rng=rng,
        compute_diagnostics=False,
    )
    # Naive analysis: reported cases treated as complete, reproduction number inferred.
    ds_naive = fit_autoregressive_model(
        incidence_vec=reported_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rng=rng,
        compute_diagnostics=False,
    )
    true_prob = calc_additional_case_prob_analytical(
        incidence_vec=true_vec,
        rep_no_func=_true_rep_no,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=t_calc,
    )

    df_out = pd.DataFrame(
        {
            "day_of_outbreak": t_calc,
            "true": np.append(true_vec, 0),
            "reported": np.append(reported_vec, np.nan),
            **{
                f"cases_{stat}": ds[f"cases_{stat}"].reindex(data_time=t_calc).values
                for stat in ("mean", "lower", "upper")
            },
            "rep_no_true": _true_rep_no(t_calc),
            "rep_no_mean": ds["rep_no_mean"].values,
            "rep_no_lower": ds["rep_no_lower"].values,
            "rep_no_upper": ds["rep_no_upper"].values,
            "rep_no_naive_mean": ds_naive["rep_no_mean"].values,
            "additional_case_prob_true": true_prob,
            "additional_case_prob_est_r": ds["additional_case_prob"].values,
            "additional_case_prob_est_r_lower": ds["additional_case_prob_lower"].values,
            "additional_case_prob_est_r_upper": ds["additional_case_prob_upper"].values,
            "additional_case_prob_known_r": ds_known_r["additional_case_prob"].values,
            "additional_case_prob_known_r_lower": ds_known_r[
                "additional_case_prob_lower"
            ].values,
            "additional_case_prob_known_r_upper": ds_known_r[
                "additional_case_prob_upper"
            ].values,
            "additional_case_prob_naive": ds_naive["additional_case_prob"].values,
        }
    ).set_index("day_of_outbreak")
    df_out.to_csv(inputs["results_paths"]["sim"])
    pd.Series(ds.attrs["diagnostics"], name="value").rename_axis("stat").to_csv(
        inputs["results_paths"]["diagnostics"]
    )


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
