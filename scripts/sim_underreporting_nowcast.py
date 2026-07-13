"""Simulation verification of the under + delayed reporting nowcast.

Takes the same true outbreak as the under-reporting simulation study and, at a single
mid/post-peak snapshot day ``D``, thins it into a *right-truncated* reported series using a reporting
ceiling and the (real Lazio) onset-to-report delay distribution — so recent onsets are only partly
reported yet (nowcasting). The under-reporting model is then refit on the data known at ``D`` and its
reconstruction of the true incidence, and the additional-case probability at the decision day
``D + 1``, are contrasted against the truth and against naive analyses.

Five additional-case probabilities are reported at ``D + 1`` (see ``make_plots`` for the panel):

- ``true`` — analytical, from the full true outbreak + true R (the oracle);
- ``naive_true_r`` — analytical, treating the reported-by-``D`` cases as complete, at true R
  (isolates the reporting/data error with R held at the truth);
- ``naive_est_r`` — the same naive (reported-only) analysis with R inferred;
- ``imperfect_true_r`` — the under-reporting offshoot (reporting ceiling + delay) at true R;
- ``imperfect_est_r`` — the full under-reporting offshoot with R inferred (also supplies the inferred
  true-case trajectory / credible band).

The snapshot day, reporting ceiling and seed are tuned (in ``get_inputs_sim_underreporting_nowcast``)
so the probabilities separate and are not all ~1: late enough that the outbreak is genuinely waning,
but with enough reported cases to fit. Inspect ``argmax(true_vec)`` after a run to retune ``D``.
"""

import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd import calc_additional_case_prob_analytical
from endoutbreakvbd.inference import fit_autoregressive_model, fit_known_rep_no_model
from scripts.inputs import get_inputs_sim_underreporting_nowcast
from scripts.sim_underreporting import _simulate_true_outbreak, _true_rep_no
from scripts.sim_underreporting_nowcast_plots import make_plots


def run_analyses():
    inputs = get_inputs_sim_underreporting_nowcast()
    serial_interval_dist_vec = inputs["serial_interval_dist_vec"]
    reporting_prob = inputs["reporting_prob"]
    delay_cdf = np.asarray(inputs["delay"]["cdf"], dtype=float)
    snapshot_day = int(inputs["snapshot_day"])
    t_calc = snapshot_day + 1  # start-of-next-day decision (matches the QRT convention)

    # A single shared random number generator threads through the simulation and every fit. The
    # true outbreak is drawn first (before any reporting draws) so it matches the under-reporting
    # simulation study exactly.
    rng = np.random.default_rng(inputs["seed"])
    true_vec = _simulate_true_outbreak(
        rng,
        serial_interval_dist_vec,
        inputs["min_outbreak_size"],
        inputs["incidence_init"],
    )
    if snapshot_day + 1 >= len(true_vec):
        raise ValueError(
            f"snapshot_day {snapshot_day} leaves no room before the end of the simulated "
            f"outbreak (length {len(true_vec)})"
        )
    reported, not_yet, never = _simulate_reporting(
        true_vec, reporting_prob, delay_cdf, snapshot_day, rng
    )
    # Right-truncated cases-by-onset known at the snapshot (onset days 0..D).
    reported_snapshot = reported[: snapshot_day + 1].astype(int)
    onset_day = np.arange(snapshot_day + 1)

    # Under-reporting offshoot with R inferred: supplies both the inferred true-case band (over onset
    # days 0..D) and the decision-day probability. The fit reports every day plus one projected day
    # (0..D+1), so the case trajectory (0..D) and the probability at the decision day D+1 both come
    # from a single fit.
    ds_imperfect_est_r = fit_autoregressive_model(
        incidence_vec=reported_snapshot,
        serial_interval_dist_vec=serial_interval_dist_vec,
        reporting_prob=reporting_prob,
        delay_cdf=delay_cdf,
        rng=rng,
        compute_diagnostics=True,
        raise_on_poor_diagnostics=False,
    )
    # Under-reporting offshoot with R fixed to the truth (isolates the reporting inference).
    ds_imperfect_true_r = fit_known_rep_no_model(
        incidence_vec=reported_snapshot,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_func=_true_rep_no,
        reporting_prob=reporting_prob,
        delay_cdf=delay_cdf,
        rng=rng,
        compute_diagnostics=False,
    )
    # Naive analysis with R inferred: reported-by-D treated as complete.
    ds_naive_est_r = fit_autoregressive_model(
        incidence_vec=reported_snapshot,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rng=rng,
        compute_diagnostics=False,
    )
    # Analytical benchmarks (exact, no free random variables to sample):
    #   true         — full true outbreak + true R (the oracle);
    #   naive_true_r — reported-by-D treated as complete + true R (reporting/data error at true R).
    prob_true = calc_additional_case_prob_analytical(
        incidence_vec=true_vec,
        rep_no_func=_true_rep_no,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=t_calc,
    )
    prob_naive_true_r = calc_additional_case_prob_analytical(
        incidence_vec=reported_snapshot,
        rep_no_func=_true_rep_no,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=t_calc,
    )

    df_traj = pd.DataFrame(
        {
            "onset_day": onset_day,
            "true": true_vec[: snapshot_day + 1],
            "reported": reported_snapshot,
            "not_yet": not_yet[: snapshot_day + 1],
            "never": never[: snapshot_day + 1],
            "cases_mean": ds_imperfect_est_r["cases_mean"]
            .sel(data_time=onset_day)
            .values,
            "cases_lower": ds_imperfect_est_r["cases_lower"]
            .sel(data_time=onset_day)
            .values,
            "cases_upper": ds_imperfect_est_r["cases_upper"]
            .sel(data_time=onset_day)
            .values,
        }
    ).set_index("onset_day")
    df_traj.to_csv(inputs["results_paths"]["trajectory"])

    prob_rows = {
        "true": (prob_true, np.nan, np.nan),
        "naive_true_r": (prob_naive_true_r, np.nan, np.nan),
        "naive_est_r": _prob_at(ds_naive_est_r, t_calc),
        "imperfect_true_r": _prob_at(ds_imperfect_true_r, t_calc),
        "imperfect_est_r": _prob_at(ds_imperfect_est_r, t_calc),
    }
    df_probs = pd.DataFrame(
        [
            {"method": method, "prob": prob, "prob_lower": lower, "prob_upper": upper}
            for method, (prob, lower, upper) in prob_rows.items()
        ]
    ).set_index("method")
    df_probs.to_csv(inputs["results_paths"]["probs"])

    pd.Series(ds_imperfect_est_r.attrs["diagnostics"], name="value").rename_axis(
        "stat"
    ).to_csv(inputs["results_paths"]["diagnostics"])


def _simulate_reporting(true_vec, reporting_prob, delay_cdf, snapshot_day, rng):
    """Thin the true outbreak into (reported-by-D, reported-later, never-reported) at snapshot ``D``.

    Matches the model's per-day reporting probability ``reporting_prob * delay_cdf[D - onset]``: each
    true case is *ever* reported with probability ``reporting_prob`` and, if so, reported by day ``D``
    with probability ``delay_cdf[D - onset]``. The three categories sum to the true incidence. The
    index case (onset day 0) is fixed and fully reported, matching the under-reporting model.
    """
    onset_days = np.arange(len(true_vec))
    ever = rng.binomial(true_vec, reporting_prob)
    never = true_vec - ever
    available_delay = np.clip(snapshot_day - onset_days, 0, len(delay_cdf) - 1)
    prob_by_snapshot = np.where(
        onset_days <= snapshot_day, delay_cdf[available_delay], 0.0
    )
    reported = rng.binomial(ever, prob_by_snapshot)
    not_yet = ever - reported
    # Fixed, fully reported index case (no hidden day-0 infections in the model).
    reported[0], not_yet[0], never[0] = true_vec[0], 0, 0
    return reported, not_yet, never


def _prob_at(ds, t_calc):
    # (mean, lower, upper) additional-case probability at the decision day.
    return (
        float(ds["additional_case_prob"].sel(time=t_calc)),
        float(ds["additional_case_prob_lower"].sel(time=t_calc)),
        float(ds["additional_case_prob_upper"].sel(time=t_calc)),
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
