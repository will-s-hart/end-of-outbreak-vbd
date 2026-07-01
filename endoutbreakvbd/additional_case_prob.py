import itertools
from typing import Annotated, Literal, cast, overload

import numpy as np
from annotated_types import Ge
from joblib import Parallel, delayed
from tqdm import tqdm

from endoutbreakvbd.model import run_renewal_model
from endoutbreakvbd._types import (
    FloatArray,
    SerialIntervalInput,
    IncidenceSeriesInput,
    IntArray,
    PercRiskThresholdInput,
    RepNoFunc,
)

nonnegint = Annotated[int, Ge(0)]


# Overloaded on t_calc so a scalar yields a scalar probability and an array yields an array
# (callers such as calc_decision_delay require the FloatArray). With
# additional_dims="broadcast" a scalar t_calc actually returns a per-sample array, but only
# the array-t_calc form is used that way, so a single pair of overloads suffices.
@overload
def calc_additional_case_prob_analytical(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: nonnegint,
    additional_dims: Literal["average", "broadcast"] = ...,
) -> float: ...


@overload
def calc_additional_case_prob_analytical(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: IntArray,
    additional_dims: Literal["average", "broadcast"] = ...,
) -> FloatArray: ...


def calc_additional_case_prob_analytical(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: nonnegint | IntArray,
    additional_dims: Literal["average", "broadcast"] = "average",
) -> float | FloatArray:
    """
    Analytically calculate the probability of one or more additional cases occurring on
    or after a given time.

    Parameters
    ----------
    incidence_vec : IncidenceSeriesInput
        Observed incidence time series. May carry trailing sample dimensions (time is the
        leading axis), e.g. posterior draws of the inferred true cases, which are aligned
        with any sample dimensions returned by ``rep_no_func``.
    rep_no_func : RepNoFunc
        Function returning the reproduction number at a given time (day). May return
        additional dimensions (e.g. posterior samples).
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    t_calc : int or IntArray
        Time(s) (day) on or after which to compute the probability of additional cases.
    additional_dims : {"average", "broadcast"}
        How to treat the trailing sample dimensions of ``incidence_vec`` / ``rep_no_func``.
        ``"average"`` (the default) averages over them, returning a single probability per
        ``t_calc``. ``"broadcast"`` keeps them, returning the per-sample probabilities
        (shape ``(*sample,)`` for scalar ``t_calc`` or ``(n_t_calc, *sample)`` for an
        array), so the caller can form its own summaries (e.g. credible intervals).

    Returns
    -------
    float or FloatArray
        Probability of additional case(s) for each time in ``t_calc`` (averaged over the
        sample dimensions, or per-sample when ``additional_dims="broadcast"``).
    """
    if additional_dims not in ("average", "broadcast"):
        raise ValueError('additional_dims must be "average" or "broadcast"')
    reduce = additional_dims == "average"
    incidence_vec = np.asarray(incidence_vec)
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec)

    if np.isscalar(t_calc):
        return _calc_additional_case_prob_analytical_scalar(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=cast(int, t_calc),
            reduce=reduce,
        )

    results = [
        _calc_additional_case_prob_analytical_scalar(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=int(t_calc_curr),
            reduce=reduce,
        )
        for t_calc_curr in np.atleast_1d(t_calc)
    ]
    if reduce:
        return np.array(results)
    # Broadcast per-t_calc results (some may be scalar early returns) to a common
    # sample shape before stacking along the leading t_calc axis.
    sample_shape = next((np.shape(r) for r in results if np.ndim(r) > 0), ())
    return np.stack(
        [np.broadcast_to(np.asarray(r, dtype=float), sample_shape) for r in results],
        axis=0,
    )


def _calc_additional_case_prob_analytical_scalar(
    *,
    incidence_vec: IntArray,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: FloatArray,
    t_calc: int,
    reduce: bool = True,
) -> float | FloatArray:
    # Analytical probability of additional cases on/after a single t_calc value. The
    # incidence may carry trailing sample dimensions (e.g. posterior draws of the true
    # cases), in which case time is the leading axis and the sample dimensions are aligned
    # with any sample dimensions of rep_no_func's output. If ``reduce`` these are averaged
    # over (a single probability); otherwise the per-sample probabilities are returned.
    sample_axes = tuple(range(1, incidence_vec.ndim))
    sample_shape = incidence_vec.shape[1:]
    nonzero_time_idx = np.nonzero(np.any(incidence_vec != 0, axis=sample_axes))[0]
    if nonzero_time_idx.size == 0:
        return 0.0
    if t_calc == 0:
        return 1.0
    t_final_case = int(np.max(nonzero_time_idx))
    serial_interval_max = len(serial_interval_dist_vec)
    t_max = t_final_case + serial_interval_max
    if t_calc > t_max:
        return 0.0

    incidence_vec_theor = incidence_vec
    if incidence_vec_theor.shape[0] < t_calc:
        incidence_vec_theor = np.concatenate(
            [
                incidence_vec_theor,
                np.zeros((t_calc - incidence_vec_theor.shape[0], *sample_shape)),
            ]
        )
    incidence_vec_theor = np.concatenate(
        [
            incidence_vec_theor[:t_calc],
            np.zeros((t_max - t_calc, *sample_shape)),
        ]
    )
    serial_interval_dist_vec = np.concatenate(
        [serial_interval_dist_vec, np.zeros(t_final_case)]
    )

    rep_no_vec_future = rep_no_func(np.arange(t_calc, t_max + 1))
    if np.isscalar(rep_no_vec_future):
        rep_no_vec_future = np.full(t_max - t_calc + 1, rep_no_vec_future, dtype=float)
    foi_vec_future = np.zeros((t_max - t_calc + 1, *sample_shape))
    for t in range(t_calc, t_max + 1):
        w_col = serial_interval_dist_vec[:t].reshape((t, *(1,) * len(sample_shape)))
        foi_vec_future[t - t_calc] = (incidence_vec_theor[:t][::-1] * w_col).sum(axis=0)
    # Contract over the leading time axis; the einsum aligns incidence and rep_no sample
    # dimensions when both are present (so draw s of the cases pairs with draw s of the
    # reproduction number). The per-sample additional-case probability is then averaged
    # over the remaining sample dimensions (reduce) or returned as-is (broadcast).
    per_sample_prob = 1 - np.exp(
        -np.einsum("t...,t...->...", foi_vec_future, rep_no_vec_future)
    )
    if reduce:
        return float(np.mean(per_sample_prob))
    return np.asarray(per_sample_prob, dtype=float)


@overload
def calc_additional_case_prob_simulation(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: nonnegint,
    n_sims: int,
    rng: np.random.Generator,
    parallel: bool = True,
) -> float: ...


@overload
def calc_additional_case_prob_simulation(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: IntArray,
    n_sims: int,
    rng: np.random.Generator,
    parallel: bool = True,
) -> FloatArray: ...


def calc_additional_case_prob_simulation(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: nonnegint | IntArray,
    n_sims: int,
    rng: np.random.Generator,
    parallel: bool = True,
) -> float | FloatArray:
    """
    Estimate the probability of one or more additional cases occurring on or after a
    given time by forward simulation.

    Parameters
    ----------
    incidence_vec : IncidenceSeriesInput
        Observed incidence time series.
    rep_no_func : RepNoFunc
        Function returning the reproduction number at a given time (day).
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    t_calc : int or IntArray
        Time(s) (day) on or after which to compute the probability of additional cases.
    n_sims : int
        Number of simulations per time point.
    rng : np.random.Generator
        Random number generator.
    parallel : bool
        Whether to run the simulations in parallel using joblib.

    Returns
    -------
    float or FloatArray
        Estimated probability of additional case(s) for each time in ``t_calc``.
    """
    incidence_vec = np.asarray(incidence_vec, dtype=int)
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec)
    t_calc = np.atleast_1d(t_calc)
    n_times = int(t_calc.size)
    n_time_sim_pairs = n_times * n_sims
    t_calc_max = int(t_calc.max())

    if incidence_vec.size < t_calc_max:
        incidence_vec = np.append(
            incidence_vec, np.zeros(t_calc_max - incidence_vec.size, dtype=int)
        )

    tasks: list[
        tuple[IntArray, RepNoFunc, FloatArray, int, np.random.Generator, int, int]
    ] = []
    child_rngs = rng.spawn(n_time_sim_pairs)
    for ((t_idx, t_curr), s_idx), child_rng in zip(
        itertools.product(enumerate(t_calc), range(n_sims)), child_rngs, strict=True
    ):
        t = int(t_curr)
        tasks.append(
            (
                incidence_vec[:t],
                rep_no_func,
                serial_interval_dist_vec,
                t,
                child_rng,
                t_idx,
                s_idx,
            )
        )

    if parallel:
        results = list(
            tqdm(
                Parallel(
                    n_jobs=-1,
                    prefer="processes",
                    return_as="generator",
                    batch_size="auto",
                )(delayed(_additional_cases_one_sim)(task) for task in tasks),
                total=n_time_sim_pairs,
                desc="Simulating additional-case probability",
            )
        )
    else:
        results = list(
            tqdm(
                map(_additional_cases_one_sim, tasks),
                total=n_time_sim_pairs,
                desc="Simulating additional-case probability",
            )
        )

    additional_cases_sims = np.full((n_times, n_sims), False)
    for t_idx, s_idx, val in results:
        additional_cases_sims[t_idx, s_idx] = val
    additional_case_prob = np.mean(additional_cases_sims, axis=1)
    if np.isscalar(t_calc):
        return float(additional_case_prob.item())
    return additional_case_prob


def _additional_cases_one_sim(
    args: tuple[
        IntArray,
        RepNoFunc,
        FloatArray,
        int,
        np.random.Generator,
        int,
        int,
    ],
) -> tuple[int, int, bool]:
    # Run one forward simulation; report whether any additional cases occur on/after t_calc
    (
        incidence_init,
        rep_no_func,
        serial_interval_dist_vec,
        t_calc,
        rng,
        t_idx,
        s_idx,
    ) = args
    if len(incidence_init) != t_calc:
        raise ValueError(
            f"t_calc ({t_calc}) does not match length of incidence_vec "
            f"({len(incidence_init)})"
        )
    if t_calc == 0:
        return t_idx, s_idx, True
    incidence_vec_sim = run_renewal_model(
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rng=rng,
        t_stop=t_calc + len(serial_interval_dist_vec),
        incidence_init=incidence_init,
        _break_on_case=True,
    )
    additional_cases = bool(np.any(incidence_vec_sim[t_calc:] > 0))
    return t_idx, s_idx, additional_cases


def calc_decision_delay(
    *,
    prob_vec: FloatArray,
    perc_risk_threshold: PercRiskThresholdInput,
    delay_of_first_prob: int,
) -> int | IntArray:
    """
    Compute the delay until the probability of additional cases first drops below a risk
    threshold.

    Parameters
    ----------
    prob_vec : FloatArray
        Probability of additional cases at successive times.
    perc_risk_threshold : PercRiskThresholdInput
        Risk threshold(s), expressed as a percentage.
    delay_of_first_prob : int
        Time (day) corresponding to the first element of ``prob_vec``.

    Returns
    -------
    int or IntArray
        Delay (days) at which the risk first falls below each threshold.
    """
    perc_risk_threshold = np.atleast_1d(perc_risk_threshold)
    below_threshold = prob_vec < (perc_risk_threshold[:, None] / 100)
    never_below_threshold = ~np.any(below_threshold, axis=1)
    if np.any(never_below_threshold):
        raise ValueError(
            "Risk does not drop below one or more thresholds: "
            f"{perc_risk_threshold[never_below_threshold]}"
        )
    decision_delay: IntArray = np.argmax(below_threshold, axis=1) + delay_of_first_prob
    if decision_delay.size == 1:
        return int(decision_delay.item())
    return decision_delay
