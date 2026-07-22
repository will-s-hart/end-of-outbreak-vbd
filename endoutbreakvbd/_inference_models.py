from collections.abc import Callable
from typing import Any

import numpy as np
import pymc as pm
import pytensor.tensor as pt
from numpy.typing import ArrayLike

from endoutbreakvbd._types import FloatArray, IntArray

# Added to Poisson rates in the under-reporting model so `mu` stays strictly positive where it
# would otherwise be exactly zero (for example, no force of infection or one side of the thinning
# split having probability 0). Small enough to leave the likelihood otherwise untouched. Days on
# which reporting itself is impossible are excluded from the reported-case likelihood separately.
_POISSON_MU_FLOOR = 1e-12

# The reported-count-based distribution supplied to CustomDist is used only for initialization
# and prior-predictive draws; `_logp` below defines the posterior density. Bounding its multiplier
# prevents an arbitrarily small positive reporting probability from producing an unusably large
# starting value. The live delay fit reaches approximately 85.2, so 100 preserves that behavior.
_INITIAL_UNOBSERVED_TO_REPORTED_RATIO_MAX = 100.0
_INITIAL_UNOBSERVED_MU_FLOOR = 1e-3


def _build_full_reporting_model(
    *,
    incidence_vec: IntArray,
    serial_interval_dist_vec: FloatArray,
    rep_no_vec_func: Callable[[int], Any],
    t_rep_no_stop: int,
) -> pm.Model:
    # Build the full-reporting renewal model (renewal force of infection as a precomputed
    # convolution, observed incidence via a Poisson likelihood). R_t is inferred to
    # `t_rep_no_stop` (see `_get_t_rep_no_stop`).
    t_data_stop = len(incidence_vec)

    incidence_local_vec = np.zeros(t_data_stop)
    incidence_local_vec[1:] = incidence_vec[1:]

    foi_vec = (
        _renewal_convolution_matrix(serial_interval_dist_vec, t_data_stop)
        @ incidence_vec
    )

    positive_foi_mask = foi_vec > 0
    if np.any(incidence_local_vec[~positive_foi_mask]):
        raise ValueError(
            "Local incidence cannot be greater than zero when force of infection is 0."
        )

    model = pm.Model(coords={"time": np.arange(t_rep_no_stop)})
    with model:
        rep_no_vec = rep_no_vec_func(t_rep_no_stop)
        expected_incidence_local = rep_no_vec[:t_data_stop] * foi_vec
        pm.Poisson(
            "likelihood",
            mu=expected_incidence_local[positive_foi_mask],
            observed=incidence_local_vec[positive_foi_mask],
        )
    return model


def _build_underreporting_model(
    *,
    incidence_vec: IntArray,
    serial_interval_dist_vec: FloatArray,
    rep_no_vec_func: Callable[[int], Any],
    reporting_prob: float,
    delay_cdf: FloatArray | None,
    t_rep_no_stop: int,
) -> pm.Model:
    # Build the fixed-index Poisson-thinning under-reporting model. With per-day reporting
    # probability pi_t, the true cases N_t by symptom-onset date follow the renewal process and
    # are thinned into reported and unreported counts:
    #     c_t ~ Poisson(pi_t * mu_t),  N_t ~ Poisson(mu_t),  mu_t = R_t * FOI_t(N).
    # The latent unreported cases U = N - c carry the self-referential renewal density via a
    # single pm.CustomDist ("unobserved"); the reported cases are a clean top-level pm.Poisson
    # ("obs"). The first reported case(s) are the fixed index (no hidden day-0 infections), so
    # only t >= 1 carries latent incidence. R_t is inferred to `t_rep_no_stop` (see
    # `_get_t_rep_no_stop`). The discrete latent gets a Metropolis step from pm.sample
    # automatically (NUTS handles the continuous R_t block), so no step is attached by the caller.
    observed_incidence_vec = np.asarray(incidence_vec, dtype=int)
    if (
        observed_incidence_vec.ndim != 1
        or observed_incidence_vec.size == 0
        or observed_incidence_vec[0] <= 0
    ):
        raise ValueError(
            "incidence_vec must be a non-empty 1-D array starting with at least "
            "one index case"
        )
    t_data_stop = len(observed_incidence_vec)
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec, dtype=float)

    reporting_prob_vec = _reporting_prob_vec(
        observed_incidence_vec, reporting_prob, delay_cdf
    )

    index_incidence = float(observed_incidence_vec[0])
    conv_mat = _renewal_convolution_matrix(serial_interval_dist_vec, t_data_stop)
    observed_incidence_after_index_vec = observed_incidence_vec[1:].astype(float)
    reporting_prob_after_index_vec = reporting_prob_vec[1:]
    positive_reporting_prob_mask = reporting_prob_after_index_vec > 0
    if np.any(
        observed_incidence_after_index_vec[~positive_reporting_prob_mask] > 0
    ):
        raise ValueError(
            "observed incidence cannot be positive on a day when the effective "
            "reporting probability is zero"
        )
    initial_unobserved_mu_vec = _initial_unobserved_mu_vec(
        observed_incidence_after_index_vec, reporting_prob_after_index_vec
    )
    index_col = pt.as_tensor_variable([index_incidence])

    model = pm.Model(
        coords={
            "time": np.arange(t_rep_no_stop),
            "data_time": np.arange(t_data_stop),
            "unobserved_time": np.arange(1, t_data_stop),
        }
    )
    with model:
        rep_no_vec = rep_no_vec_func(t_rep_no_stop)
        rep_no_data = rep_no_vec[:t_data_stop]

        def _logp(value: Any, rep_no_data: Any) -> Any:
            incidence = pt.concatenate(
                [index_col, observed_incidence_after_index_vec + value]
            )
            mu = rep_no_data * pt.dot(conv_mat, incidence)
            return pm.logp(
                pm.Poisson.dist(
                    mu=(1 - reporting_prob_after_index_vec) * mu[1:] + _POISSON_MU_FLOOR
                ),
                value,
            )

        def _dist(rep_no_data: Any, size: Any) -> Any:
            # Initial values / prior-predictive only (the density is overridden by _logp), so this
            # does not change the target posterior. The finite, capped mean is based on the reported
            # count; `size` is unused because the latent length is fixed via `shape`.
            return pm.Poisson.dist(
                mu=initial_unobserved_mu_vec,
                shape=t_data_stop - 1,
            )

        latent_rv = pm.CustomDist(
            "unobserved",
            rep_no_data,
            logp=_logp,
            dist=_dist,
            dtype="int64",
            dims=("unobserved_time",),
        )
        incidence_data = pt.concatenate(
            [index_col, observed_incidence_after_index_vec + latent_rv]
        )
        pm.Deterministic("incidence", incidence_data, dims=("data_time",))
        mu_vec = rep_no_data * pt.dot(conv_mat, incidence_data)
        pm.Poisson(
            "obs",
            mu=(
                reporting_prob_after_index_vec[positive_reporting_prob_mask]
                * mu_vec[1:][positive_reporting_prob_mask]
                + _POISSON_MU_FLOOR
            ),
            observed=observed_incidence_after_index_vec[
                positive_reporting_prob_mask
            ],
        )
    return model


def _renewal_convolution_matrix(
    serial_interval_dist_vec: ArrayLike, n_days: int
) -> FloatArray:
    """
    Renewal-equation force of infection expressed as a (constant) matrix operator.

    Returns the ``n_days`` by ``n_days`` lower-triangular matrix ``conv_mat`` for which
    ``conv_mat @ incidence_vec`` is the renewal force of infection
    ``foi[t] = sum_{r < t} incidence_vec[r] * serial_interval[t - 1 - r]`` — the same quantity
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
    serial_interval_ext_vec = np.concatenate(
        [
            serial_interval_dist_vec,
            np.zeros(max(n_days - 1 - len(serial_interval_dist_vec), 0)),
        ]
    )
    conv_mat = np.zeros((n_days, n_days))
    for t in range(1, n_days):
        conv_mat[t, :t] = serial_interval_ext_vec[:t][::-1]
    return conv_mat


def _reporting_prob_vec(
    incidence_vec: IntArray, reporting_prob: float, delay_cdf: FloatArray | None
) -> FloatArray:
    # Per-day effective reporting probability over onset days 0..(t_data_stop - 1), as seen from an
    # "as-of" day equal to the last day of this incidence snapshot (t_data_stop - 1) — i.e. the
    # snapshot encodes reporting known by the end of that day. The as-of day is set purely by the
    # length of `incidence_vec`. Without a delay CDF the probability is a constant `reporting_prob`
    # (pure under-reporting). With one it is `reporting_prob * P(delay <= t_as_of - t_onset)`,
    # so recent onset days (small available delay) are truncated toward zero (right-truncation /
    # nowcasting) while old onset days plateau at `reporting_prob`.
    reporting_prob = float(reporting_prob)
    if not np.isfinite(reporting_prob) or not 0 < reporting_prob <= 1:
        raise ValueError("reporting_prob must be finite and in the interval (0, 1]")

    t_data_stop = len(incidence_vec)
    if delay_cdf is None:
        return np.full(t_data_stop, reporting_prob)
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
    t_as_of = t_data_stop - 1
    available_delay = t_as_of - np.arange(t_data_stop)
    return reporting_prob * delay_cdf[np.clip(available_delay, 0, len(delay_cdf) - 1)]


def _initial_unobserved_mu_vec(
    observed_incidence_vec: IntArray, reporting_prob_vec: FloatArray
) -> FloatArray:
    """Finite reported-count-based mean used only to initialize latent cases."""
    observed_incidence_vec = np.asarray(observed_incidence_vec, dtype=float)
    reporting_prob_vec = np.asarray(reporting_prob_vec, dtype=float)
    unobserved_to_reported_ratio_vec = np.divide(
        1 - reporting_prob_vec,
        reporting_prob_vec,
        out=np.zeros_like(reporting_prob_vec),
        where=reporting_prob_vec > 0,
    )
    unobserved_to_reported_ratio_vec = np.minimum(
        unobserved_to_reported_ratio_vec,
        _INITIAL_UNOBSERVED_TO_REPORTED_RATIO_MAX,
    )
    return np.maximum(
        observed_incidence_vec * unobserved_to_reported_ratio_vec,
        _INITIAL_UNOBSERVED_MU_FLOOR,
    )
