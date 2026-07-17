import itertools
from typing import Annotated, Literal, cast, overload

import numpy as np
from annotated_types import Ge
from joblib import Parallel, delayed
from numpy.typing import ArrayLike
from tqdm import tqdm

from endoutbreakvbd._types import (
    FloatArray,
    IncidenceSeriesInput,
    IntArray,
    RepNoFunc,
    RiskThresholdPctInput,
    SerialIntervalInput,
)
from endoutbreakvbd.model import run_renewal_model

NonNegativeInt = Annotated[int, Ge(0)]


# Overloaded on t_calc and additional_dims: averaging a scalar calculation time yields a
# scalar probability, while broadcasting or passing an array of calculation times yields an
# array.
@overload
def calc_additional_case_prob_analytical(
    *,
    incidence: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: NonNegativeInt,
    additional_dims: Literal["average"] = ...,
) -> float: ...


@overload
def calc_additional_case_prob_analytical(
    *,
    incidence: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: NonNegativeInt,
    additional_dims: Literal["broadcast"],
) -> FloatArray: ...


@overload
def calc_additional_case_prob_analytical(
    *,
    incidence: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: IntArray,
    additional_dims: Literal["average", "broadcast"] = ...,
) -> FloatArray: ...


def calc_additional_case_prob_analytical(
    *,
    incidence: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: NonNegativeInt | IntArray,
    additional_dims: Literal["average", "broadcast"] = "average",
) -> float | FloatArray:
    """
    Analytically calculate the probability of one or more additional cases occurring on
    or after a given time.

    Parameters
    ----------
    incidence : IncidenceSeriesInput
        Observed incidence time series. May carry trailing sample dimensions (time is the
        leading axis), e.g. posterior draws of the inferred true incidence, which are aligned
        with any sample dimensions returned by ``rep_no_func``.
    rep_no_func : RepNoFunc
        Function returning the reproduction number at a given time (day). May return
        additional dimensions (e.g. posterior samples).
    serial_interval_dist_vec : SerialIntervalInput
        Discretised serial interval distribution (probability mass per day).
    t_calc : int or IntArray
        Time(s) (day) on or after which to compute the probability of additional cases.
    additional_dims : {"average", "broadcast"}
        How to treat the trailing sample dimensions of ``incidence`` / ``rep_no_func``.
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
    incidence = np.asarray(incidence)
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec)

    if np.isscalar(t_calc):
        prob_result = _calc_additional_case_prob_analytical_scalar(
            incidence=incidence,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=cast(int, t_calc),
            reduce=reduce,
        )
        if reduce or isinstance(prob_result, np.ndarray):
            return prob_result
        sample_shape = _additional_case_prob_sample_shape(
            incidence=incidence, rep_no_func=rep_no_func
        )
        return np.full(sample_shape, prob_result, dtype=float)

    t_calc_vec = np.atleast_1d(t_calc)
    prob_results = [
        _calc_additional_case_prob_analytical_scalar(
            incidence=incidence,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=int(t_calc_current),
            reduce=reduce,
        )
        for t_calc_current in t_calc_vec
    ]
    if reduce:
        return np.array(prob_results)
    # When every calculation short-circuits, no result has exposed the reproduction-number
    # sample dimensions. Probe them once so the constant probabilities retain the full shape.
    if all(not isinstance(prob_result, np.ndarray) for prob_result in prob_results):
        sample_shape = _additional_case_prob_sample_shape(
            incidence=incidence, rep_no_func=rep_no_func
        )
        prob_results = [
            np.full(sample_shape, prob_result, dtype=float)
            for prob_result in prob_results
        ]
    broadcast_prob_results = np.broadcast_arrays(*prob_results)
    return np.stack(broadcast_prob_results, axis=0)


@overload
def calc_additional_case_prob_simulation(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: SerialIntervalInput,
    t_calc: NonNegativeInt,
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
    t_calc: NonNegativeInt | IntArray,
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
    t_calc_is_scalar = np.isscalar(t_calc)
    incidence_vec = np.asarray(incidence_vec, dtype=int)
    serial_interval_dist_vec = np.asarray(serial_interval_dist_vec)
    t_calc_vec = np.atleast_1d(t_calc)
    n_t_calc = int(t_calc_vec.size)
    n_t_calc_sim_pairs = n_t_calc * n_sims
    t_calc_max = int(t_calc_vec.max())

    if incidence_vec.size < t_calc_max:
        incidence_vec = np.append(
            incidence_vec, np.zeros(t_calc_max - incidence_vec.size, dtype=int)
        )

    simulation_tasks: list[
        tuple[IntArray, RepNoFunc, FloatArray, int, np.random.Generator, int, int]
    ] = []
    child_rngs = rng.spawn(n_t_calc_sim_pairs)
    for ((t_idx, t_current), s_idx), child_rng in zip(
        itertools.product(enumerate(t_calc_vec), range(n_sims)),
        child_rngs,
        strict=True,
    ):
        t = int(t_current)
        simulation_tasks.append(
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
        simulation_results = list(
            tqdm(
                Parallel(
                    n_jobs=-1,
                    prefer="processes",
                    return_as="generator",
                    batch_size="auto",
                )(
                    delayed(_has_additional_case_one_sim)(task)
                    for task in simulation_tasks
                ),
                total=n_t_calc_sim_pairs,
                desc="Simulating additional-case probability",
            )
        )
    else:
        simulation_results = list(
            tqdm(
                map(_has_additional_case_one_sim, simulation_tasks),
                total=n_t_calc_sim_pairs,
                desc="Simulating additional-case probability",
            )
        )

    additional_case_indicator_mat = np.full((n_t_calc, n_sims), False)
    for t_idx, s_idx, has_additional_case in simulation_results:
        additional_case_indicator_mat[t_idx, s_idx] = has_additional_case
    additional_case_prob_vec = np.mean(additional_case_indicator_mat, axis=1)
    if t_calc_is_scalar:
        return float(additional_case_prob_vec.item())
    return additional_case_prob_vec


def calc_decision_delay(
    *,
    prob_vec: ArrayLike,
    t_vec: ArrayLike,
    risk_threshold_pct: RiskThresholdPctInput,
    t_final_case: int,
) -> FloatArray:
    """
    Days from the final case until the probability of additional cases first drops below each
    risk threshold, considering only the days after the final case.

    ``prob_vec[i]`` is the probability at outbreak time ``t_vec[i]``; the two share an ordering
    but ``t_vec`` need not be contiguous (e.g. strided real-time snapshots). Returns NaN for any
    threshold the risk never crosses over the days after the final case.

    Parameters
    ----------
    prob_vec : ArrayLike
        Probability of additional cases at each time in ``t_vec``.
    t_vec : ArrayLike
        Outbreak times corresponding to the entries in ``prob_vec``.
    risk_threshold_pct : RiskThresholdPctInput
        Risk threshold(s), expressed as a percentage.
    t_final_case : int
        Outbreak time of the final case; delays are measured from it.

    Returns
    -------
    FloatArray
        Delay (days) at which the risk first falls below each threshold (NaN if it never does).
    """
    prob_vec = np.asarray(prob_vec, dtype=float)
    t_vec = np.asarray(t_vec)
    if prob_vec.shape != t_vec.shape:
        raise ValueError("prob_vec and t_vec must have the same shape")
    risk_threshold_pct_vec = np.atleast_1d(np.asarray(risk_threshold_pct, dtype=float))
    after_final_case_mask = t_vec > t_final_case
    prob_after_final_case_vec = prob_vec[after_final_case_mask]
    t_after_final_case_vec = t_vec[after_final_case_mask]
    if t_after_final_case_vec.size == 0:
        return np.full(risk_threshold_pct_vec.shape, np.nan)
    below_threshold_mat = prob_after_final_case_vec[None, :] < (
        risk_threshold_pct_vec[:, None] / 100
    )  # (n_thresholds, n_times_after_final_case)
    ever_below_threshold_mask = below_threshold_mat.any(axis=1)
    first_below_threshold_idx_vec = below_threshold_mat.argmax(axis=1)
    return np.where(
        ever_below_threshold_mask,
        t_after_final_case_vec[first_below_threshold_idx_vec] - t_final_case,
        np.nan,
    )


def _additional_case_prob_sample_shape(
    *, incidence: IntArray, rep_no_func: RepNoFunc
) -> tuple[int, ...]:
    # Short-circuit probabilities do not evaluate R_t, so probe one valid outbreak time to
    # recover its trailing sample dimensions. Time is the leading axis of vector-valued R_t.
    rep_no_at_t0 = np.asarray(rep_no_func(np.array([0], dtype=int)))
    rep_no_sample_shape = rep_no_at_t0.shape[1:] if rep_no_at_t0.ndim else ()
    return np.broadcast_shapes(incidence.shape[1:], rep_no_sample_shape)


def _calc_additional_case_prob_analytical_scalar(
    *,
    incidence: IntArray,
    rep_no_func: RepNoFunc,
    serial_interval_dist_vec: FloatArray,
    t_calc: int,
    reduce: bool = True,
) -> float | FloatArray:
    # Analytical probability of additional cases on/after a single t_calc value. The
    # incidence may carry trailing sample dimensions (e.g. posterior draws of true
    # incidence), in which case time is the leading axis and the sample dimensions are aligned
    # with any sample dimensions of rep_no_func's output. If ``reduce`` these are averaged
    # over (a single probability); otherwise the per-sample probabilities are returned.
    sample_axes = tuple(range(1, incidence.ndim))
    sample_shape = incidence.shape[1:]
    positive_incidence_t_idx_vec = np.nonzero(np.any(incidence != 0, axis=sample_axes))[
        0
    ]
    if positive_incidence_t_idx_vec.size == 0:
        return 0.0
    if t_calc == 0:
        return 1.0
    t_final_case = int(np.max(positive_incidence_t_idx_vec))
    serial_interval_max = len(serial_interval_dist_vec)
    t_max = t_final_case + serial_interval_max
    if t_calc > t_max:
        return 0.0

    incidence_theoretical = incidence
    if incidence_theoretical.shape[0] < t_calc:
        incidence_theoretical = np.concatenate(
            [
                incidence_theoretical,
                np.zeros((t_calc - incidence_theoretical.shape[0], *sample_shape)),
            ]
        )
    incidence_theoretical = np.concatenate(
        [
            incidence_theoretical[:t_calc],
            np.zeros((t_max - t_calc, *sample_shape)),
        ]
    )
    serial_interval_dist_vec = np.concatenate(
        [serial_interval_dist_vec, np.zeros(t_final_case)]
    )

    rep_no_future = rep_no_func(np.arange(t_calc, t_max + 1))
    if np.isscalar(rep_no_future):
        rep_no_future = np.full(t_max - t_calc + 1, rep_no_future, dtype=float)
    foi_future = np.zeros((t_max - t_calc + 1, *sample_shape))
    for t in range(t_calc, t_max + 1):
        serial_interval_col = serial_interval_dist_vec[:t].reshape(
            (t, *(1,) * len(sample_shape))
        )
        foi_future[t - t_calc] = (
            incidence_theoretical[:t][::-1] * serial_interval_col
        ).sum(axis=0)
    # Contract over the leading time axis; the einsum aligns incidence and rep_no sample
    # dimensions when both are present (so draw s of the cases pairs with draw s of the
    # reproduction number). The per-sample additional-case probability is then averaged
    # over the remaining sample dimensions (reduce) or returned as-is (broadcast).
    per_sample_prob = 1 - np.exp(
        -np.einsum("t...,t...->...", foi_future, rep_no_future)
    )
    if reduce:
        return float(np.mean(per_sample_prob))
    return np.asarray(per_sample_prob, dtype=float)


def _has_additional_case_one_sim(
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
    incidence_sim_vec = run_renewal_model(
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rng=rng,
        t_stop=t_calc + len(serial_interval_dist_vec),
        incidence_init=incidence_init,
        _break_on_case=True,
    )
    has_additional_case = bool(np.any(incidence_sim_vec[t_calc:] > 0))
    return t_idx, s_idx, has_additional_case
