"""Quasi-real-time end-of-outbreak inference for the Lazio chikungunya outbreak under
under-reporting and right-truncation (nowcasting).

At each historical snapshot date the model is refit using only the cases-by-onset known at
that date (the forward-filled reporting matrix), with a per-day reporting probability that
combines a reporting ceiling and the estimated onset-to-report delay. The additional-case
probability is reported at each snapshot for the suitability and autoregressive models at
the primary 60% reporting ceiling. A single full-output fit at the latest snapshot supplies
the inferred true-case and reproduction-number trajectories.

This is the multi-hour cluster analysis; use ``--stride`` for a quick local wiring check.
"""

import argparse
import os
from typing import Any

import numpy as np
import pandas as pd

from endoutbreakvbd.inference import fit_autoregressive_model, fit_suitability_model
from scripts.inputs import get_inputs_lazio_underreporting_qrt
from scripts.lazio_underreporting_qrt_plots import make_plots

# Bound the joblib (n_jobs=-1) worker pool to the SLURM allocation on the cluster; loky
# otherwise sees every physical core on the node. Locally (no SLURM) it falls back to cpu_count.
if os.environ.get("SLURM_CPUS_PER_TASK"):
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", os.environ["SLURM_CPUS_PER_TASK"])


def run_analyses(
    start_date="2017-09-30", end_date="2017-12-31", stride=1, parallel=True
):
    inputs = get_inputs_lazio_underreporting_qrt(
        start_date=start_date, end_date=end_date, stride=stride
    )
    serial_interval_dist_vec = inputs["serial_interval_dist_vec"]
    delay_cdf = inputs["delay"]["cdf"]
    suitability_mean_vec = inputs["suitability_mean_vec"]

    # A single shared random number generator threads through every sweep and the trajectory.
    rng = np.random.default_rng(2)

    # Real-time nowcast sweeps: the suitability reporting-ceiling sweep (single-sourced from
    # inputs) plus the autoregressive model at the primary (60%) ceiling. Each fits the
    # right-truncated sequence of snapshots through the public quasi-real-time API (a list of
    # incidence series starting on outbreak day 0, with each projected calculation time inferred
    # from that snapshot's length).
    sweeps = [
        ("suitability", name, prob) for name, prob in inputs["suitability_sweep"]
    ] + [("autoregressive", "autoregressive_p60", inputs["reporting_prob"])]
    for model, name, reporting_prob in sweeps:
        fit_kwargs: dict[str, Any] = {
            "incidence": inputs["incidence_vec_list"],
            "serial_interval_dist_vec": serial_interval_dist_vec,
            "quasi_real_time": True,
            "reporting_prob": reporting_prob,
            "delay_cdf": delay_cdf,
            "rng": rng,
            "parallel": parallel,
            "compute_diagnostics": True,
            "raise_on_poor_diagnostics": False,
        }
        if model == "suitability":
            posterior_ds = fit_suitability_model(
                suitability_mean_vec=suitability_mean_vec, **fit_kwargs
            )
        else:
            posterior_ds = fit_autoregressive_model(**fit_kwargs)
        output_df = pd.DataFrame(
            {
                "calc_time": inputs["t_calc_vec"],
                "date": inputs["decision_date_vec"],
                "additional_case_prob": posterior_ds["additional_case_prob"].values,
                "reproduction_number_mean": posterior_ds["rep_no_mean"].values,
            }
        ).set_index("calc_time")
        output_df.to_csv(inputs["results_paths"][name])
        pd.Series(posterior_ds.attrs["diagnostics"], name="value").rename_axis(
            "stat"
        ).to_csv(inputs["results_paths"][f"{name}_diagnostics"])

    _run_trajectory(inputs, rng)
    _write_delay(inputs)


def _run_trajectory(inputs, rng):
    # Full-output suitability fit at the latest snapshot for the true-case / R_t panels.
    incidence_vec = inputs["latest_incidence_vec"]
    posterior_ds = fit_suitability_model(
        incidence=incidence_vec,
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        suitability_mean_vec=inputs["suitability_mean_vec"],
        reporting_prob=inputs["reporting_prob"],
        delay_cdf=inputs["delay"]["cdf"],
        rng=rng,
        compute_diagnostics=False,
    )
    onset_day = posterior_ds["time"].values
    date_vec = inputs["outbreak_start_date"] + pd.to_timedelta(onset_day, unit="D")
    # The fit reports one projected day past the data (the current-day risk), so the trajectory is
    # one longer than the snapshot. Incidence on that projected day was not observed.
    reported_incidence_vec = np.append(incidence_vec, np.nan)
    _posterior_trajectory_frame(
        posterior_ds,
        onset_day=onset_day,
        date_vec=date_vec,
        reported_incidence_vec=reported_incidence_vec,
    ).to_csv(inputs["results_paths"]["trajectory"])


def _posterior_trajectory_frame(
    posterior_ds, *, onset_day, date_vec, reported_incidence_vec
):
    """Assemble the QRT suitability trajectory used by the case panel."""
    return pd.DataFrame(
        {
            "onset_day": onset_day,
            "date": date_vec,
            "reported_incidence": reported_incidence_vec,
            **{
                f"incidence_{stat}": posterior_ds[f"incidence_{stat}"]
                .reindex(data_time=onset_day)
                .values
                for stat in ("mean", "lower", "upper")
            },
            "reproduction_number_mean": posterior_ds["rep_no_mean"].values,
            "reproduction_number_lower": posterior_ds["rep_no_lower"].values,
            "reproduction_number_upper": posterior_ds["rep_no_upper"].values,
            "suitability_mean": posterior_ds["suitability_mean"].values,
            "suitability_lower": posterior_ds["suitability_lower"].values,
            "suitability_upper": posterior_ds["suitability_upper"].values,
            "rep_no_factor_mean": posterior_ds["rep_no_factor_mean"].values,
            "rep_no_factor_lower": posterior_ds["rep_no_factor_lower"].values,
            "rep_no_factor_upper": posterior_ds["rep_no_factor_upper"].values,
            "additional_case_prob": posterior_ds["additional_case_prob"].values,
        }
    ).set_index("onset_day")


def _write_delay(inputs):
    delay = inputs["delay"]
    pd.DataFrame(
        {
            "delay": delay["support"],
            "cdf_fitted": delay["cdf"],
            "pmf_fitted": delay["pmf_fitted"],
            "pmf_empirical": delay["pmf_empirical"],
        }
    ).set_index("delay").to_csv(inputs["results_paths"]["delay"])


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    parser.add_argument("--start-date", default="2017-09-30")
    parser.add_argument("--end-date", default="2017-12-31")
    parser.add_argument(
        "--stride",
        type=int,
        default=1,
        help="Subsample the daily snapshot grid (use a large value for a quick check)",
    )
    parser.add_argument(
        "--serial",
        action="store_true",
        help="Fit snapshots sequentially instead of in parallel (debugging / verification)",
    )
    args = parser.parse_args()
    run_analyses(
        start_date=args.start_date,
        end_date=args.end_date,
        stride=args.stride,
        parallel=not args.serial,
    )
    if not args.results_only:
        make_plots(
            start_date=args.start_date, end_date=args.end_date, stride=args.stride
        )
