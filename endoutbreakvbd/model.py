from collections.abc import Callable
from typing import Annotated

import numpy as np
from annotated_types import Gt

from endoutbreakvbd._types import (
    IncidenceInitInput,
    IntArray,
    RepNoFunc,
    SerialIntervalInput,
)


def run_renewal_model(
    *,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    rng: np.random.Generator,
    t_stop: Annotated[int, Gt(0)] = 1000,
    incidence_init: IncidenceInitInput = None,
    _break_on_case: bool = False,
) -> IntArray:
    """
    Simulate an outbreak from a renewal-equation branching process.

    Parameters
    ----------
    rep_no_func : RepNoFunc
        Function returning the reproduction number at a given time (day).
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    rng : np.random.Generator
        Random number generator.
    t_stop : int
        Maximum number of time steps (days) to simulate.
    incidence_init : IncidenceInitInput
        Initial incidence. A scalar seeds a single day; an array seeds the first
        ``len(array)`` days. Defaults to a single initial case.
    _break_on_case : bool
        Internal flag; if True, stop as soon as a case occurs after the seeded period
        (used by simulation-based additional-case-probability calculations).

    Returns
    -------
    IntArray
        Simulated incidence time series, truncated once transmission has died out (or at
        ``t_stop``).
    """
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec)
    serial_interval_max = len(serial_interval_dist_vec)
    serial_interval_dist_vec = np.concatenate(
        [
            serial_interval_dist_vec,
            np.zeros(np.maximum(t_stop - 1 - len(serial_interval_dist_vec), 0)),
        ]
    )
    incidence_vec: IntArray = np.zeros(t_stop, dtype=int)
    if incidence_init is None:
        incidence_init = 1
    incidence_init = np.atleast_1d(incidence_init)
    t_start = int(incidence_init.size)
    incidence_vec[:t_start] = incidence_init
    for t in range(t_start, t_stop):
        rep_no = rep_no_func(t)
        foi = np.sum(incidence_vec[:t][::-1] * serial_interval_dist_vec[:t])
        incidence = int(rng.poisson(foi * rep_no))
        incidence_vec[t] = incidence
        if (
            t >= serial_interval_max
            and incidence_vec[(t - serial_interval_max) : (t + 1)].sum() == 0
        ) or (_break_on_case and incidence > 0):
            incidence_vec = incidence_vec[: (t + 1)]
            break
    return incidence_vec


def simulate_outbreak(
    *,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    rng: np.random.Generator,
    min_size: int,
    max_size: int | None = None,
    accept: Callable[[IntArray], bool] | None = None,
    incidence_init: IncidenceInitInput = None,
    t_stop: Annotated[int, Gt(0)] = 1000,
    max_attempts: int = 10000,
) -> IntArray:
    """
    Repeatedly simulate ``run_renewal_model`` until an outbreak meets a size target.

    Rejection sampling: each attempt draws a fresh outbreak and is accepted when its total
    size is at least ``min_size`` (and at most ``max_size``, if given) and the optional
    ``accept`` predicate holds. ``rep_no_func`` is fixed across attempts; only the
    branching-process randomness differs.

    Parameters
    ----------
    rep_no_func : RepNoFunc
        Function returning the reproduction number at a given time (day).
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    rng : np.random.Generator
        Random number generator, threaded through every attempt.
    min_size : int
        Minimum accepted total outbreak size (sum of the incidence series).
    max_size : int, optional
        Maximum accepted total outbreak size. If None, no upper bound is applied.
    accept : Callable[[IntArray], bool], optional
        Extra acceptance predicate on the candidate incidence series (e.g. a final-case-day
        window). Checked only for candidates already within the size bounds.
    incidence_init : IncidenceInitInput
        Initial incidence, passed to ``run_renewal_model``.
    t_stop : int
        Maximum number of days to simulate per attempt.
    max_attempts : int
        Number of attempts before giving up.

    Returns
    -------
    IntArray
        The first accepted incidence time series.

    Raises
    ------
    RuntimeError
        If no accepted outbreak is drawn within ``max_attempts``.
    """
    for _ in range(max_attempts):
        incidence_vec = run_renewal_model(
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rng=rng,
            t_stop=t_stop,
            incidence_init=incidence_init,
        )
        size = int(incidence_vec.sum())
        if size < min_size or (max_size is not None and size > max_size):
            continue
        if accept is not None and not accept(incidence_vec):
            continue
        return incidence_vec
    raise RuntimeError(
        f"could not simulate an outbreak meeting the size target in {max_attempts} attempts"
    )
