from typing import Annotated, Callable

import numpy as np
from annotated_types import Ge
from tqdm import tqdm

from endoutbreakvbd.model import renewal_model

nonnegint = Annotated[int, Ge(0)]


def eop_analytical(
    *,
    incidence_vec: list[int] | np.ndarray[int],
    rep_no_func: Callable[[int | np.ndarray[int]], float | np.ndarray[float]],
    gen_time_dist_vec: list[float] | np.ndarray[float],
    t_calc: nonnegint | np.ndarray[nonnegint],
) -> float | np.ndarray[float]:
    if not np.isscalar(t_calc):
        return np.array(
            [
                eop_analytical(
                    incidence_vec=incidence_vec,
                    rep_no_func=rep_no_func,
                    gen_time_dist_vec=gen_time_dist_vec,
                    t_calc=t_calc_curr,
                )
                for t_calc_curr in t_calc
            ]
        )
    if t_calc == 0:
        return 0

    gen_time_max = len(gen_time_dist_vec)
    t_last_case = np.max(np.nonzero(incidence_vec)[0])
    t_max = t_last_case + gen_time_max
    if t_calc > t_max:
        return 1

    if len(incidence_vec) < t_calc:
        incidence_vec = np.append(
            incidence_vec, np.zeros(t_calc - len(incidence_vec), dtype=int)
        )
    incidence_vec_theor = np.append(
        incidence_vec[:t_calc], np.zeros(t_max - t_calc, dtype=int)
    )
    gen_time_dist_vec = np.concatenate([gen_time_dist_vec, np.zeros(t_last_case)])

    rep_no_vec_future = rep_no_func(np.arange(t_calc, t_max + 1))
    foi_vec_future = np.zeros(t_max - t_calc + 1)
    for t in range(t_calc, t_max + 1):
        foi_vec_future[t - t_calc] = np.sum(
            incidence_vec_theor[:t][::-1] * gen_time_dist_vec[:t]
        )

    eop = np.exp(-np.dot(foi_vec_future, rep_no_vec_future)).mean(
        # Handles case where rep_no_func returns equally likely values along axis 1
    )
    return eop


def eop_simulation(
    incidence_vec: list[int] | np.ndarray[int],
    rep_no_func: Callable[[int | np.ndarray[int]], float | np.ndarray[float]],
    gen_time_dist_vec: list[float] | np.ndarray[float],
    t_calc: nonnegint | np.ndarray[nonnegint],
    n_sims: int,
    rng: np.random.Generator,
) -> float | np.ndarray[float]:
    if not np.isscalar(t_calc):
        return np.array(
            [
                eop_simulation(
                    incidence_vec=incidence_vec,
                    rep_no_func=rep_no_func,
                    gen_time_dist_vec=gen_time_dist_vec,
                    t_calc=t_calc_curr,
                    n_sims=n_sims,
                    rng=rng,
                )
                for t_calc_curr in tqdm(t_calc)
            ]
        )
    if t_calc == 0:
        return 0

    if len(incidence_vec) < t_calc:
        incidence_vec = np.append(
            incidence_vec, np.zeros(t_calc - len(incidence_vec), dtype=int)
        )
    outbreak_ended_sims = np.full(n_sims, False)
    for sim in range(n_sims):
        incidence_vec_sim = renewal_model(
            rep_no_func=rep_no_func,
            gen_time_dist_vec=gen_time_dist_vec,
            rng=rng,
            t_stop=t_calc + len(gen_time_dist_vec),
            incidence_init=incidence_vec[:t_calc],
            _break_on_case=True,
        )
        outbreak_ended_sims[sim] = np.sum(incidence_vec_sim[t_calc:]) == 0
    eop = np.mean(outbreak_ended_sims)
    return eop
