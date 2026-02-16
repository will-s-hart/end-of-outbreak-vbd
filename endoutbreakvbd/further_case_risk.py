import itertools
from typing import Annotated, cast, overload

import numpy as np
from annotated_types import Ge
from joblib import Parallel, delayed
from tqdm import tqdm

from endoutbreakvbd.model import run_renewal_model
from endoutbreakvbd.types import (
    FloatArray,
    GenTimeInput,
    IncidenceSeriesInput,
    IntArray,
    PercRiskThresholdInput,
    RepNoFunc,
)

nonnegint = Annotated[int, Ge(0)]


@overload
def calc_further_case_risk_analytical(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    gen_time_dist_vec: GenTimeInput,
    t_calc: nonnegint,
) -> float: ...


@overload
def calc_further_case_risk_analytical(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    gen_time_dist_vec: GenTimeInput,
    t_calc: IntArray,
) -> FloatArray: ...


def calc_further_case_risk_analytical(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    gen_time_dist_vec: GenTimeInput,
    t_calc: nonnegint | IntArray,
) -> float | FloatArray:
    # Analytical calculation of risk of further cases on/after time t_calc
    incidence_vec = np.asarray(incidence_vec)
    gen_time_dist_vec = np.asarray(gen_time_dist_vec)

    if np.isscalar(t_calc):
        t_calc_scalar = cast(int, t_calc)
        return _calc_further_case_risk_analytical_scalar(
            incidence_vec=incidence_vec,
            rep_no_func=rep_no_func,
            gen_time_dist_vec=gen_time_dist_vec,
            t_calc=t_calc_scalar,
        )

    return np.array(
        [
            _calc_further_case_risk_analytical_scalar(
                incidence_vec=incidence_vec,
                rep_no_func=rep_no_func,
                gen_time_dist_vec=gen_time_dist_vec,
                t_calc=int(t_calc_curr),
            )
            for t_calc_curr in np.atleast_1d(t_calc)
        ],
    )


def _calc_further_case_risk_analytical_scalar(
    *,
    incidence_vec: IntArray,
    rep_no_func: RepNoFunc,
    gen_time_dist_vec: FloatArray,
    t_calc: int,
) -> float:
    if t_calc == 0:
        return 1.0

    gen_time_max = len(gen_time_dist_vec)
    nonzero_incidence_idx = np.nonzero(incidence_vec)[0]
    if nonzero_incidence_idx.size == 0:
        return 0.0
    t_last_case = int(np.max(nonzero_incidence_idx))
    t_max = t_last_case + gen_time_max
    if t_calc > t_max:
        return 0.0

    incidence_vec_theor = incidence_vec
    if incidence_vec_theor.size < t_calc:
        incidence_vec_theor = np.append(
            incidence_vec_theor,
            np.zeros(t_calc - incidence_vec_theor.size),
        )
    incidence_vec_theor = np.append(
        incidence_vec_theor[:t_calc],
        np.zeros(t_max - t_calc),
    )
    gen_time_dist_vec = np.concatenate([gen_time_dist_vec, np.zeros(t_last_case)])

    rep_no_vec_future = rep_no_func(np.arange(t_calc, t_max + 1))
    foi_vec_future = np.zeros(t_max - t_calc + 1)
    for t in range(t_calc, t_max + 1):
        foi_vec_future[t - t_calc] = np.sum(
            incidence_vec_theor[:t][::-1] * gen_time_dist_vec[:t]
        )
    # Note mean() is used to average over different possible future reproduction
    # numbers, optionally indexed along dimension 1 of rep_no_vec future
    further_case_risk = 1 - np.exp(-np.dot(foi_vec_future, rep_no_vec_future)).mean()
    return float(further_case_risk)


@overload
def calc_further_case_risk_simulation(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    gen_time_dist_vec: GenTimeInput,
    t_calc: nonnegint,
    n_sims: int,
    rng: np.random.Generator,
    parallel: bool = True,
) -> float: ...


@overload
def calc_further_case_risk_simulation(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    gen_time_dist_vec: GenTimeInput,
    t_calc: IntArray,
    n_sims: int,
    rng: np.random.Generator,
    parallel: bool = True,
) -> FloatArray: ...


def calc_further_case_risk_simulation(
    *,
    incidence_vec: IncidenceSeriesInput,
    rep_no_func: RepNoFunc,
    gen_time_dist_vec: GenTimeInput,
    t_calc: nonnegint | IntArray,
    n_sims: int,
    rng: np.random.Generator,
    parallel: bool = True,
) -> float | FloatArray:
    # Simulation-based calculation of risk of further cases on/after time t_calc

    incidence_vec = np.asarray(incidence_vec)
    gen_time_dist_vec = np.asarray(gen_time_dist_vec)
    t_calc = np.atleast_1d(t_calc)
    n_times = int(t_calc.size)
    n_time_sim_pairs = n_times * n_sims
    t_calc_max = int(t_calc.max())

    if incidence_vec.size < t_calc_max:
        incidence_vec = np.append(
            incidence_vec, np.zeros(t_calc_max - incidence_vec.size)
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
                gen_time_dist_vec,
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
                )(delayed(_further_cases_one_sim)(task) for task in tasks),
                total=n_time_sim_pairs,
                desc="Simulating further case risk",
            )
        )
    else:
        results = list(
            tqdm(
                map(_further_cases_one_sim, tasks),
                total=n_time_sim_pairs,
                desc="Simulating further case risk",
            )
        )

    further_cases_sims = np.full((n_times, n_sims), False)
    for t_idx, s_idx, val in results:
        further_cases_sims[t_idx, s_idx] = val
    further_case_risk = np.mean(further_cases_sims, axis=1)
    if np.isscalar(t_calc):
        return float(further_case_risk.item())
    return further_case_risk


def _further_cases_one_sim(
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
    (incidence_init, rep_no_func, gen_time_dist_vec, t_calc, rng, t_idx, s_idx) = args
    if len(incidence_init) != t_calc:
        raise ValueError(
            f"t_calc ({t_calc}) does not match length of incidence_vec "
            f"({len(incidence_init)})"
        )
    if t_calc == 0:
        return t_idx, s_idx, True
    incidence_vec_sim = run_renewal_model(
        rep_no_func=rep_no_func,
        gen_time_dist_vec=gen_time_dist_vec,
        rng=rng,
        t_stop=t_calc + len(gen_time_dist_vec),
        incidence_init=incidence_init,
        _break_on_case=True,
    )
    further_cases = bool(np.any(incidence_vec_sim[t_calc:] > 0))
    return t_idx, s_idx, further_cases


def calc_declaration_delay(
    *,
    risk_vec: FloatArray,
    perc_risk_threshold: PercRiskThresholdInput,
    delay_of_first_risk: int,
) -> int | IntArray:
    perc_risk_threshold = np.atleast_1d(perc_risk_threshold)
    below_threshold = risk_vec < (perc_risk_threshold[:, None] / 100)
    never_below_threshold = ~np.any(below_threshold, axis=1)
    if np.any(never_below_threshold):
        raise ValueError(
            "Risk does not drop below one or more thresholds: "
            f"{perc_risk_threshold[never_below_threshold]}"
        )
    declaration_delay: IntArray = (
        np.argmax(below_threshold, axis=1) + delay_of_first_risk
    )
    if declaration_delay.size == 1:
        return int(declaration_delay.item())
    return declaration_delay
