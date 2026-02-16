from typing import Annotated, Protocol, TypeAlias, overload

import numpy as np
from annotated_types import Gt
from numpy.typing import NDArray

IntArray: TypeAlias = NDArray[np.int_]
FloatArray: TypeAlias = NDArray[np.float64]
GenTimeInput: TypeAlias = list[float] | FloatArray
IncidenceInput: TypeAlias = int | list[int] | IntArray | None


class RepNoFunc(Protocol):
    @overload
    def __call__(self, t: int, /) -> float: ...

    @overload
    def __call__(self, t: IntArray, /) -> FloatArray: ...


def run_renewal_model(
    *,
    rep_no_func: RepNoFunc,
    gen_time_dist_vec: GenTimeInput,
    rng: np.random.Generator,
    t_stop: Annotated[int, Gt(0)] = 1000,
    incidence_init: IncidenceInput = None,
    _break_on_case: bool = False,
) -> IntArray:
    gen_time_dist_vec = np.asarray(gen_time_dist_vec)
    gen_time_max = len(gen_time_dist_vec)
    gen_time_dist_vec = np.concatenate(
        [
            gen_time_dist_vec,
            np.zeros(np.maximum(t_stop - 1 - len(gen_time_dist_vec), 0)),
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
        foi = np.sum(incidence_vec[:t][::-1] * gen_time_dist_vec[:t])
        incidence = int(rng.poisson(foi * rep_no))
        incidence_vec[t] = incidence
        if (
            t >= gen_time_max and incidence_vec[(t - gen_time_max) : (t + 1)].sum() == 0
        ) or (_break_on_case and incidence > 0):
            incidence_vec = incidence_vec[: (t + 1)]
            break
    return incidence_vec
