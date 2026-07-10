"""Retrospective end-of-outbreak inference for the Lazio chikungunya outbreak under
under-reporting (no delay / right-truncation).

Unlike the quasi-real-time nowcast, this fits the full reported outbreak once per model with a
constant per-day reporting probability (``delay_cdf=None``), inflating the reported cases to a
latent true-case trajectory. The suitability model is fit across a reporting-ceiling sweep
(60/80/100%) and the autoregressive model at the primary 60% ceiling. The primary (60%)
suitability fit additionally supplies the full true-case / R_t trajectory panels.

The "full outbreak knowledge" dashed benchmark on the probability panel reuses the existing
full-reporting ``lazio_outbreak`` results (see ``get_inputs_lazio_underreporting_retro``).
"""

import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd.inference import fit_autoregressive_model, fit_suitability_model
from endoutbreakvbd.utils import posterior_trajectory_frame
from scripts.inputs import get_inputs_lazio_underreporting_retro
from scripts.lazio_underreporting_retro_plots import make_plots


def run_analyses(sampler_kwargs=None):
    sampler_kwargs = sampler_kwargs or {}
    inputs = get_inputs_lazio_underreporting_retro()
    incidence_vec = inputs["incidence_vec"]
    serial_interval_dist_vec = inputs["serial_interval_dist_vec"]
    outbreak_start_date = inputs["outbreak_start_date"]
    results_paths = inputs["results_paths"]

    # A single shared random number generator threads through every fit.
    rng = np.random.default_rng(2)

    # Suitability reporting-ceiling sweep (constant reporting probability, no delay). The primary
    # (first) ceiling additionally supplies the full trajectory panels.
    primary_name = inputs["suitability_sweep"][0][0]
    for name, reporting_prob in inputs["suitability_sweep"]:
        ds = fit_suitability_model(
            incidence_vec=incidence_vec,
            serial_interval_dist_vec=serial_interval_dist_vec,
            suitability_mean_vec=inputs["suitability_mean_vec"],
            reporting_prob=reporting_prob,
            rng=rng,
            compute_diagnostics=True,
            raise_on_poor_diagnostics=False,
            **sampler_kwargs,
        )
        _write_prob(ds, results_paths[name], outbreak_start_date)
        pd.Series(ds.attrs["diagnostics"], name="value").rename_axis("stat").to_csv(
            results_paths[f"{name}_diagnostics"]
        )
        if name == primary_name:
            _write_trajectory(
                ds, results_paths["trajectory"], incidence_vec, outbreak_start_date
            )

    # Autoregressive model at the primary ceiling.
    ds = fit_autoregressive_model(
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        reporting_prob=inputs["reporting_prob"],
        rng=rng,
        compute_diagnostics=True,
        raise_on_poor_diagnostics=False,
        **sampler_kwargs,
    )
    _write_prob(ds, results_paths["autoregressive_p60"], outbreak_start_date)
    pd.Series(ds.attrs["diagnostics"], name="value").rename_axis("stat").to_csv(
        results_paths["autoregressive_p60_diagnostics"]
    )


def _dates(ds, start_date):
    onset_day = ds["time"].values
    return onset_day, start_date + pd.to_timedelta(onset_day, unit="D")


def _write_prob(ds, save_path, start_date):
    onset_day, date = _dates(ds, start_date)
    pd.DataFrame(
        {
            "day_of_outbreak": onset_day,
            "date": date,
            "additional_case_prob": ds["additional_case_prob"].values,
            "reproduction_number_mean": ds["rep_no_mean"].values,
        }
    ).set_index("day_of_outbreak").to_csv(save_path)


def _write_trajectory(ds, save_path, incidence_vec, start_date):
    onset_day, date = _dates(ds, start_date)
    # The fit reports one projected day past the data (the current-day risk), so the posterior
    # trajectory is one longer than the observed series; that projected day has no reported cases
    # (0, matching the model's zero-padded latent cases there — not NaN, which would break the
    # incidence-bar axis scaling).
    reported = np.append(incidence_vec, 0)
    posterior_trajectory_frame(
        ds, onset_day=onset_day, date=date, reported=reported
    ).to_csv(save_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-r",
        "--results-only",
        action="store_true",
        help="Only run analyses and save results (no plots)",
    )
    parser.add_argument(
        "--draws",
        type=int,
        default=None,
        help="Override sampler draws (quick wiring check)",
    )
    parser.add_argument(
        "--tune",
        type=int,
        default=None,
        help="Override sampler tune (quick wiring check)",
    )
    args = parser.parse_args()
    sampler_kwargs = {
        k: v for k, v in (("draws", args.draws), ("tune", args.tune)) if v is not None
    }
    run_analyses(sampler_kwargs=sampler_kwargs)
    if not args.results_only:
        make_plots()
