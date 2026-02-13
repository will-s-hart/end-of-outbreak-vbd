from typing import Annotated, Callable

import numpy as np
from annotated_types import Gt


def run_renewal_model(
    *,
    rep_no_func: Callable[[int | np.ndarray[int]], float | np.ndarray[float]],
    gen_time_dist_vec: list[float] | np.ndarray[float],
    rng: np.random.Generator,
    t_stop: Annotated[int, Gt(0)] = 1000,
    incidence_init: int | list[int] | np.ndarray[int] | None = None,
    _break_on_case: bool = False,
) -> np.ndarray[int]:
    gen_time_max = len(gen_time_dist_vec)
    gen_time_dist_vec = np.concatenate(
        [
            gen_time_dist_vec,
            np.zeros(np.maximum(t_stop - 1 - len(gen_time_dist_vec), 0)),
        ]
    )
    incidence_vec = np.zeros(t_stop, dtype=int)
    if incidence_init is None:
        incidence_init = 1
    t_start = np.array(incidence_init).size
    incidence_vec[:t_start] = incidence_init
    for t in range(t_start, t_stop):
        rep_no = rep_no_func(t)
        foi = np.sum(incidence_vec[:t][::-1] * gen_time_dist_vec[:t])
        incidence = rng.poisson(foi * rep_no)
        incidence_vec[t] = incidence
        if (
            t >= gen_time_max and incidence_vec[(t - gen_time_max) : (t + 1)].sum() == 0
        ) or (_break_on_case and incidence > 0):
            incidence_vec = incidence_vec[: (t + 1)]
            break
    return incidence_vec
