from collections.abc import Callable
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.typing import ArrayLike

from endoutbreakvbd._types import FloatArray, IntArray


def _renewal_convolution_matrix(
    serial_interval_dist_vec: ArrayLike, n_days: int
) -> FloatArray:
    """
    Renewal-equation force of infection expressed as a (constant) matrix operator.

    Returns the ``n_days`` by ``n_days`` lower-triangular matrix ``conv_mat`` for which
    ``conv_mat @ incidence_vec`` is the renewal force of infection
    ``foi[s] = sum_{r < s} incidence_vec[r] * serial_interval[s - 1 - r]`` — the same quantity
    ``model.run_renewal_model`` accumulates one day at a time, but vectorised for a fixed
    incidence series (as needed by the inference models). Current-day incidence never
    contributes to its own force of infection, so ``conv_mat`` is strictly lower-triangular
    (row 0 is zero).

    Parameters
    ----------
    serial_interval_dist_vec : ArrayLike
        Discretised serial interval distribution (probability mass per day). Zero-extended
        internally when shorter than ``n_days - 1``.
    n_days : int
        Number of days (the size of the square matrix).

    Returns
    -------
    FloatArray
        The ``n_days`` by ``n_days`` lower-triangular convolution matrix.
    """
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec, dtype=float)
    serial_interval_ext = np.concatenate(
        [
            serial_interval_dist_vec,
            np.zeros(max(n_days - 1 - len(serial_interval_dist_vec), 0)),
        ]
    )
    conv_mat = np.zeros((n_days, n_days))
    for s in range(1, n_days):
        conv_mat[s, :s] = serial_interval_ext[:s][::-1]
    return conv_mat


def _build_full_reporting_model(
    *,
    incidence_vec: IntArray,
    serial_interval_dist_vec: FloatArray,
    rep_no_vec_func: Callable[[int], Any],
    t_infer_rep_no_to: int,
) -> pm.Model:
    # Build the full-reporting renewal model (renewal force of infection as a precomputed
    # convolution, observed incidence via a Poisson likelihood). R_t is inferred to
    # `t_infer_rep_no_to` (see `_reproduction_number_horizon`).
    t_data_to = len(incidence_vec)

    incidence_vec_local = np.zeros(t_data_to)
    incidence_vec_local[1:] = incidence_vec[1:]

    foi_vec = (
        _renewal_convolution_matrix(serial_interval_dist_vec, t_data_to) @ incidence_vec
    )

    nonzero_foi_idx = foi_vec > 0
    if np.any(incidence_vec_local[~nonzero_foi_idx]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of infection is 0."
        )

    model = pm.Model(coords={"time": np.arange(t_infer_rep_no_to)})
    with model:
        rep_no_vec = rep_no_vec_func(t_infer_rep_no_to)
        expected_incidence_local = rep_no_vec[:t_data_to] * foi_vec
        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[nonzero_foi_idx],
            observed=incidence_vec_local[nonzero_foi_idx],
        )
    return model


def _reporting_prob_vec(
    incidence_vec: IntArray, reporting_prob: float, delay_cdf: FloatArray | None
) -> FloatArray:
    # Per-day effective reporting probability over onset days 0..(t_data_to - 1), as seen from an
    # "as-of" day equal to the last day of this incidence snapshot (t_data_to - 1) — i.e. the
    # snapshot encodes reporting known by the end of that day. The as-of day is set purely by the
    # length of `incidence_vec`. Without a delay CDF the probability is a constant `reporting_prob`
    # (pure under-reporting). With one it is `reporting_prob * P(delay <= as_of_day - onset_day)`,
    # so recent onset days (small available delay) are truncated toward zero (right-truncation /
    # nowcasting) while old onset days plateau at `reporting_prob`.
    reporting_prob = float(reporting_prob)
    if not np.isfinite(reporting_prob) or not 0 < reporting_prob <= 1:
        raise ValueError("reporting_prob must be finite and in the interval (0, 1]")

    t_data_to = len(incidence_vec)
    if delay_cdf is None:
        return np.full(t_data_to, reporting_prob)
    delay_cdf = np.asarray(delay_cdf, dtype=float)
    if (
        delay_cdf.ndim != 1
        or delay_cdf.size == 0
        or not np.all(np.isfinite(delay_cdf))
        or np.any((delay_cdf < 0) | (delay_cdf > 1))
        or np.any(np.diff(delay_cdf) < 0)
    ):
        raise ValueError(
            "delay_cdf must be a non-empty, finite, non-decreasing 1-D array "
            "with values in [0, 1]"
        )
    as_of_day = t_data_to - 1
    available_delay = as_of_day - np.arange(t_data_to)
    return reporting_prob * delay_cdf[np.clip(available_delay, 0, len(delay_cdf) - 1)]


def _build_underreporting_model(
    *,
    incidence_vec: IntArray,
    serial_interval_dist_vec: FloatArray,
    rep_no_vec_func: Callable[[int], Any],
    reporting_prob: float,
    delay_cdf: FloatArray | None,
    t_infer_rep_no_to: int,
) -> pm.Model:
    # Build the fixed-index Poisson-thinning under-reporting model. With per-day reporting
    # probability pi_t, the true cases N_t by symptom-onset date follow the renewal process and
    # are thinned into reported and unreported counts:
    #     c_t ~ Poisson(pi_t * mu_t),  N_t ~ Poisson(mu_t),  mu_t = R_t * FOI_t(N).
    # The latent unreported cases U = N - c carry the self-referential renewal density via a
    # single pm.CustomDist ("unobserved"); the reported cases are a clean top-level pm.Poisson
    # ("obs"). The first reported case(s) are the fixed index (no hidden day-0 infections), so
    # only t >= 1 carries latent cases. R_t is inferred to `t_infer_rep_no_to` (see
    # `_reproduction_number_horizon`). The discrete latent gets a Metropolis step from pm.sample
    # automatically (NUTS handles the continuous R_t block), so no step is attached by the caller.
    observed_vec = np.asarray(incidence_vec, dtype=int)
    if observed_vec.ndim != 1 or observed_vec.size == 0 or observed_vec[0] <= 0:
        raise ValueError(
            "incidence_vec must be a non-empty 1-D array starting with at least "
            "one index case"
        )
    t_data_to = len(observed_vec)
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec, dtype=float)

    # Floor the per-day reporting probability above zero so (1 - pi) / pi stays finite and pi * mu
    # is non-degenerate in the likelihoods below.
    reporting_prob_vec = np.clip(
        _reporting_prob_vec(observed_vec, reporting_prob, delay_cdf), 1e-6, 1.0
    )

    index_incidence = float(observed_vec[0])
    conv_mat = _renewal_convolution_matrix(serial_interval_dist_vec, t_data_to)
    observed_after_index = observed_vec[1:].astype(float)
    reporting_prob_after_index = reporting_prob_vec[1:]
    index_col = pt.as_tensor_variable([index_incidence])

    model = pm.Model(
        coords={
            "time": np.arange(t_infer_rep_no_to),
            "data_time": np.arange(t_data_to),
            "unobserved_time": np.arange(1, t_data_to),
        }
    )
    with model:
        rep_no_vec = rep_no_vec_func(t_infer_rep_no_to)
        rep_no_data = rep_no_vec[:t_data_to]

        def _logp(value: Any, rep_no_data: Any) -> Any:
            cases = pt.concatenate([index_col, observed_after_index + value])
            mu = rep_no_data * pt.dot(conv_mat, cases)
            # +1e-12 keeps the Poisson mu > 0 where FOI is 0 (day 0 / no sources) or pi == 1.
            return pm.logp(
                pm.Poisson.dist(mu=(1 - reporting_prob_after_index) * mu[1:] + 1e-12),
                value,
            )

        def _dist(rep_no_data: Any, size: Any) -> Any:
            # Initial values / prior-predictive only (the density is overridden by _logp), so this
            # does not change the target posterior. Moment-match the latent unreported cases to the
            # reported count, U ~ Poisson(reported * (1 - pi) / pi); the floor only keeps mu > 0 for
            # the draw (`size` is unused — the latent length is fixed via `shape`).
            return pm.Poisson.dist(
                mu=np.maximum(
                    observed_after_index
                    * (1 - reporting_prob_after_index)
                    / reporting_prob_after_index,
                    1e-3,
                ),
                shape=t_data_to - 1,
            )

        latent_rv = pm.CustomDist(
            "unobserved",
            rep_no_data,
            logp=_logp,
            dist=_dist,
            dtype="int64",
            dims=("unobserved_time",),
        )
        cases_data = pt.concatenate([index_col, observed_after_index + latent_rv])
        pm.Deterministic("cases", cases_data, dims=("data_time",))
        mu_vec = rep_no_data * pt.dot(conv_mat, cases_data)
        pm.Poisson(
            "obs",
            mu=reporting_prob_after_index * mu_vec[1:] + 1e-12,
            observed=observed_vec[1:],
        )
    return model
