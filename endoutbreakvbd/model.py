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
    serial_interval_dist_vec = np.concatenate([
        serial_interval_dist_vec,
        np.zeros(np.maximum(t_stop - 1 - len(serial_interval_dist_vec), 0)),
    ])
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
