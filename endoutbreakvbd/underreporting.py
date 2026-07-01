"""Under-reporting offshoot of the renewal inference.

A fixed-index Poisson-thinning renewal model with a discrete latent vector of true cases by
symptom-onset date. With per-day reporting probability ``pi_t``::

    c_t ~ Poisson(pi_t * mu_t),   N_t ~ Poisson(mu_t),   mu_t = R_t * FOI_t(N)

The latent unreported cases ``U = N - c`` carry the self-referential renewal density via a single
``pm.CustomDist``; the observed cases are a clean top-level ``pm.Poisson``. The first reported
case(s) are taken as the index (no hidden infections on day 0), so only ``t >= 1`` carries latent
cases. A per-day reporting probability handles combined under-reporting and right-truncation
(nowcasting).

This module only *builds* the PyMC model (and the reporting/convolution helpers); the shared
sampling, summary, and additional-case-probability machinery lives in ``inference._fit_model``.
It is a sibling of ``rep_no_models`` and imports nothing from ``inference``.
"""

from collections.abc import Callable
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt

from endoutbreakvbd._types import FloatArray, IncidenceSeriesInput, IntArray


def build_reporting_prob_vec(
    t: int,
    analysis_time: int,
    reporting_prob_ceiling: float,
    delay_cdf: FloatArray,
) -> FloatArray:
    """
    Per-day effective reporting probability combining under-reporting and right-truncation.

    ``reporting_prob_t = reporting_prob_ceiling * P(delay <= analysis_time - onset_t)``. Recent
    onset days (small available delay) tend to zero (truncation); old onset days tend to
    ``reporting_prob_ceiling`` (pure under-reporting).

    Parameters
    ----------
    t : int
        Length of the returned vector (number of onset days).
    analysis_time : int
        Onset-day index of the analysis snapshot ("today").
    reporting_prob_ceiling : float
        Long-run reporting probability once all delay has elapsed.
    delay_cdf : FloatArray
        Non-decreasing onset-to-report delay CDF, indexed by integer delay (0, 1, ...).

    Returns
    -------
    FloatArray
        Per-day effective reporting probability, length ``t``.
    """
    delay_cdf = np.asarray(delay_cdf, dtype=float)
    avail = analysis_time - np.arange(t)
    return np.where(
        avail < 0,
        0.0,
        reporting_prob_ceiling * delay_cdf[np.clip(avail, 0, len(delay_cdf) - 1)],
    )


def _resolve_reporting_prob(
    incidence_vec: IncidenceSeriesInput,
    reporting_prob: float,
    delay_cdf: FloatArray | None,
) -> FloatArray:
    # Build the per-day effective reporting probability: a constant ceiling (pure
    # under-reporting) or, with a delay CDF, the truncation-aware vector for a snapshot
    # taken on the last data day.
    t = len(incidence_vec)
    if delay_cdf is None:
        return np.full(t, float(reporting_prob))
    return build_reporting_prob_vec(
        t=t,
        analysis_time=t - 1,
        reporting_prob_ceiling=float(reporting_prob),
        delay_cdf=delay_cdf,
    )


def _build_convolution_matrix(
    serial_interval_dist_vec: FloatArray, t: int
) -> FloatArray:
    # Constant lower-triangular banded matrix C with FOI = C @ cases, i.e.
    # foi[s] = sum_{r<s} cases[r] * w_ext[s-1-r]. Matches the renewal FOI loop in
    # inference's full-reporting model exactly, replacing the per-fit pytensor.scan.
    w = np.asarray(serial_interval_dist_vec, dtype=float)
    w_ext = np.concatenate([w, np.zeros(max(t - 1 - len(w), 0))])
    conv_mat = np.zeros((t, t))
    for s in range(1, t):
        conv_mat[s, :s] = w_ext[:s][::-1]
    return conv_mat


def reproduction_number_horizon(
    t_data_to: int, serial_interval_max: int, t_calc: IntArray
) -> int:
    """
    Horizon to which R_t must be inferred for the offshoot. The latest latent case can sit
    anywhere in the observed window, so R_t (carrying any seasonal decline) is inferred a full
    serial interval past the data; it is not the horizon for the latents or probabilities.
    """
    return max(t_data_to + serial_interval_max, int(np.max(t_calc)) + 1)


def build_underreporting_model(
    *,
    incidence_vec: IncidenceSeriesInput,
    serial_interval_dist_vec: FloatArray,
    rep_no_vec_func: Callable[[int], Any],
    reporting_prob: float,
    delay_cdf: FloatArray | None,
    t_calc: IntArray,
) -> tuple[pm.Model, Any]:
    """
    Build the fixed-index Poisson-thinning under-reporting model.

    Returns the (unentered) PyMC model and the latent ``unobserved`` random variable, so the
    caller can attach the Metropolis step that targets it.
    """
    observed_vec = np.asarray(incidence_vec, dtype=int)
    t_data_to = len(observed_vec)
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec, dtype=float)
    serial_interval_max = len(serial_interval_dist_vec)
    t_infer_rep_no_to = reproduction_number_horizon(
        t_data_to, serial_interval_max, t_calc
    )
    n_pad = t_infer_rep_no_to - t_data_to

    reporting_prob_vec = np.clip(
        _resolve_reporting_prob(observed_vec, reporting_prob, delay_cdf), 1e-6, 1.0
    )

    # Fixed index: the first reported case(s) are the index (no hidden day-0 cases), so only
    # t >= 1 carries latent unreported cases.
    index_incidence = float(max(int(observed_vec[0]), 1))
    conv_mat = _build_convolution_matrix(serial_interval_dist_vec, t_data_to)
    observed_rest = observed_vec[1:].astype(float)
    reporting_rest = reporting_prob_vec[1:]
    index_col = pt.as_tensor_variable([index_incidence])

    model = pm.Model(
        coords={
            "time": np.arange(t_infer_rep_no_to),
            "gen_time": np.arange(1, t_data_to),
        }
    )
    with model:
        rep_no_vec = rep_no_vec_func(t_infer_rep_no_to)
        rep_no_obs = rep_no_vec[:t_data_to]

        def _logp(value: Any, rep_no_obs: Any) -> Any:
            cases = pt.concatenate([index_col, observed_rest + value])
            mu = rep_no_obs * pt.dot(conv_mat, cases)
            return pm.logp(
                pm.Poisson.dist(mu=(1 - reporting_rest) * mu[1:] + 1e-12), value
            )

        def _dist(rep_no_obs: Any, size: Any) -> Any:
            return pm.Poisson.dist(
                mu=np.maximum(
                    observed_rest * (1 - reporting_rest) / reporting_rest, 0.1
                ),
                shape=t_data_to - 1,
            )

        latent_rv = pm.CustomDist(
            "unobserved",
            rep_no_obs,
            logp=_logp,
            dist=_dist,
            dtype="int64",
            dims=("gen_time",),
        )
        cases_obs = pt.concatenate([index_col, observed_rest + latent_rv])
        pm.Deterministic(
            "cases",
            pt.concatenate([cases_obs, pt.zeros((n_pad,))]) if n_pad else cases_obs,
            dims=("time",),
        )
        mu_vec = rep_no_obs * pt.dot(conv_mat, cases_obs)
        # Clean observation likelihood — a top-level pm.Poisson on the reported cases.
        pm.Poisson(
            "obs", mu=reporting_rest * mu_vec[1:] + 1e-12, observed=observed_vec[1:]
        )
    return model, latent_rv
