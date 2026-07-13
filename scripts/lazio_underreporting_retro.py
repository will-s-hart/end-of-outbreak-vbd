"""Retrospective end-of-outbreak inference for the Lazio chikungunya outbreak under
under-reporting (no delay / right-truncation).

Unlike the quasi-real-time nowcast, this fits the full reported outbreak once per model with a
constant per-day reporting probability (``delay_cdf=None``), inflating the reported cases to a
latent true-case trajectory. The suitability and autoregressive models are both fit at a 60%
reporting probability. Each fit writes a full posterior trajectory frame; the suitability fit
additionally carries the suitability and R_t-factor decomposition.

The "full outbreak knowledge" dashed benchmark on the probability panel reuses the existing
full-reporting ``lazio_outbreak`` results (see ``get_inputs_lazio_underreporting_retro``).
"""

import argparse

import numpy as np
import pandas as pd

from endoutbreakvbd.inference import fit_autoregressive_model, fit_suitability_model
from scripts.inputs import get_inputs_lazio_underreporting_retro
from scripts.lazio_underreporting_retro_plots import make_plots


def run_analyses(sampler_kwargs=None):
    sampler_kwargs = sampler_kwargs or {}
    inputs = get_inputs_lazio_underreporting_retro()
    incidence_vec = inputs["incidence_vec"]
    serial_interval_dist_vec = inputs["serial_interval_dist_vec"]
    outbreak_start_date = inputs["outbreak_start_date"]
    reporting_prob = inputs["reporting_prob"]
    results_paths = inputs["results_paths"]

    # A single shared random number generator threads through both fits.
    rng = np.random.default_rng(2)

    # Suitability fit — supplies the true-case, suitability and R_t trajectory panels.
    ds_suit = fit_suitability_model(
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        suitability_mean_vec=inputs["suitability_mean_vec"],
        reporting_prob=reporting_prob,
        rng=rng,
        compute_diagnostics=True,
        raise_on_poor_diagnostics=False,
        **sampler_kwargs,
    )
    _write_results(
        ds_suit,
        results_paths["suitability_p60"],
        incidence_vec,
        outbreak_start_date,
        suitability=True,
    )
    _write_diagnostics(ds_suit, results_paths["suitability_p60_diagnostics"])

    # Autoregressive fit — supplies the comparison R_t trajectory panel.
    ds_ar = fit_autoregressive_model(
        incidence_vec=incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        reporting_prob=reporting_prob,
        rng=rng,
        compute_diagnostics=True,
        raise_on_poor_diagnostics=False,
        **sampler_kwargs,
    )
    _write_results(
        ds_ar,
        results_paths["autoregressive_p60"],
        incidence_vec,
        outbreak_start_date,
        suitability=False,
    )
    _write_diagnostics(ds_ar, results_paths["autoregressive_p60_diagnostics"])


def _write_results(ds, save_path, incidence_vec, start_date, *, suitability):
    onset_day = ds["time"].values
    date = start_date + pd.to_timedelta(onset_day, unit="D")
    # The fit reports one projected day past the data (the current-day risk), so the posterior
    # trajectory is one longer than the observed series. Incidence on that projected day was not
    # observed, so both reported and inferred cases are represented as NaN for that row.
    reported = np.append(incidence_vec, np.nan)
    columns = {
        "day_of_outbreak": onset_day,
        "date": date,
        "reported": reported,
        **{
            f"cases_{stat}": ds[f"cases_{stat}"].reindex(data_time=onset_day).values
            for stat in ("mean", "lower", "upper")
        },
        "reproduction_number_mean": ds["rep_no_mean"].values,
        "reproduction_number_lower": ds["rep_no_lower"].values,
        "reproduction_number_upper": ds["rep_no_upper"].values,
    }
    if suitability:
        columns.update(
            {
                f"{var}_{stat}": ds[f"{var}_{stat}"].values
                for var in ("suitability", "rep_no_factor")
                for stat in ("mean", "lower", "upper")
            }
        )
    columns["additional_case_prob"] = ds["additional_case_prob"].values
    pd.DataFrame(columns).set_index("day_of_outbreak").to_csv(save_path)


def _write_diagnostics(ds, save_path):
    pd.Series(ds.attrs["diagnostics"], name="value").rename_axis("stat").to_csv(
        save_path
    )


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
