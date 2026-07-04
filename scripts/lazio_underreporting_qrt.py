"""Quasi-real-time end-of-outbreak inference for the Lazio chikungunya outbreak under
under-reporting and right-truncation (nowcasting).

At each historical snapshot date the model is refit using only the cases-by-onset known at
that date (the forward-filled reporting matrix), with a per-day reporting probability that
combines a reporting ceiling and the estimated onset-to-report delay. The additional-case
probability is reported at each snapshot for the suitability model (reporting ceilings 60/80/
100%) and the autoregressive model (60%). A single full-output fit at the latest snapshot
supplies the inferred true-case and reproduction-number trajectories.

This is the multi-hour cluster analysis; use ``--stride`` for a quick local wiring check.
"""

import argparse
import os

import numpy as np
import pandas as pd

from endoutbreakvbd.inference import _fit_model_qrt, fit_suitability_model
from endoutbreakvbd.rep_no_models import build_ar_rep_no, build_suitability_rep_no
from scripts.inputs import get_inputs_lazio_underreporting_qrt
from scripts.lazio_underreporting_qrt_plots import make_plots

# Bound the joblib (n_jobs=-1) worker pool to the SLURM allocation on the cluster; loky
# otherwise sees every physical core on the node. Locally (no SLURM) it falls back to cpu_count.
if os.environ.get("SLURM_CPUS_PER_TASK"):
    os.environ.setdefault("LOKY_MAX_CPU_COUNT", os.environ["SLURM_CPUS_PER_TASK"])


def run_analyses(
    start_date="2017-10-01", end_date="2017-12-31", stride=1, parallel=True
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
    # inputs) plus the autoregressive model at the primary (60%) ceiling.
    sweeps = [
        (
            name,
            build_suitability_rep_no(suitability_mean_vec=suitability_mean_vec),
            prob,
        )
        for name, prob in inputs["suitability_sweep"]
    ] + [("autoregressive_p60", build_ar_rep_no(), inputs["reporting_prob"])]
    for name, rep_no_vec_func, reporting_prob in sweeps:
        ds = _fit_model_qrt(
            incidence_vec=inputs["incidence_vecs"],
            serial_interval_dist_vec=serial_interval_dist_vec,
            rep_no_vec_func=rep_no_vec_func,
            t_calc=inputs["calc_times"],
            reporting_prob=reporting_prob,
            delay_cdf=delay_cdf,
            rng=rng,
            parallel=parallel,
            compute_diagnostics=True,
            raise_on_poor_diagnostics=False,
        )
        df_out = pd.DataFrame(
            {
                "calc_time": inputs["calc_times"],
                "date": inputs["decision_dates"],
                "additional_case_prob": ds["additional_case_prob"].values,
                "reproduction_number_mean": ds["rep_no_mean"].values,
            }
        ).set_index("calc_time")
        df_out.to_csv(inputs["results_paths"][name])
        pd.Series(ds.attrs["diagnostics"], name="value").rename_axis("stat").to_csv(
            inputs["results_paths"][f"{name}_diagnostics"]
        )

    _run_trajectory(inputs, rng)
    _write_delay(inputs)


def _run_trajectory(inputs, rng):
    # Full-output suitability fit at the latest snapshot for the true-case / R_t panels.
    incidence_vec = inputs["latest_incidence_vec"]
    ds = fit_suitability_model(
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=inputs["serial_interval_dist_vec"],
        suitability_mean_vec=inputs["suitability_mean_vec"],
        reporting_prob=inputs["reporting_prob"],
        delay_cdf=inputs["delay"]["cdf"],
        rng=rng,
        compute_diagnostics=False,
    )
    onset_day = ds["time"].values
    date = inputs["outbreak_start_date"] + pd.to_timedelta(onset_day, unit="D")
    df_out = pd.DataFrame(
        {
            "onset_day": onset_day,
            "date": date,
            "reported": incidence_vec,
            "cases_mean": ds["cases_mean"].values,
            "cases_lower": ds["cases_lower"].values,
            "cases_upper": ds["cases_upper"].values,
            "reproduction_number_mean": ds["rep_no_mean"].values,
            "reproduction_number_lower": ds["rep_no_lower"].values,
            "reproduction_number_upper": ds["rep_no_upper"].values,
            "suitability_mean": ds["suitability_mean"].values,
            "suitability_lower": ds["suitability_lower"].values,
            "suitability_upper": ds["suitability_upper"].values,
            "rep_no_factor_mean": ds["rep_no_factor_mean"].values,
            "rep_no_factor_lower": ds["rep_no_factor_lower"].values,
            "rep_no_factor_upper": ds["rep_no_factor_upper"].values,
            "additional_case_prob": ds["additional_case_prob"].values,
        }
    ).set_index("onset_day")
    df_out.to_csv(inputs["results_paths"]["trajectory"])


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
    parser.add_argument("--start-date", default="2017-10-01")
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
