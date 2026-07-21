import atexit
import os
import shutil
from collections.abc import Callable, Sequence
from pathlib import Path
from typing import Any, cast

import numpy as np
import pytensor
import xarray as xr
from joblib import Parallel, delayed, parallel_config
from tqdm import tqdm

from endoutbreakvbd._inference_core import (
    _combine_diagnostic_components,
    _DiagnosticComponents,
    _fit_single_model_result,
    _is_incidence_sequence,
    _summarize_and_check_diagnostics,
)
from endoutbreakvbd._types import (
    FloatArray,
    IncidenceSeriesInput,
    IntArray,
    SerialIntervalInput,
)


def _fit_model_qrt(
    *,
    incidence: IncidenceSeriesInput | Sequence[IncidenceSeriesInput],
    serial_interval_dist_vec: SerialIntervalInput,
    rep_no_vec_func: Callable[[int], Any],
    reporting_prob: float | None = None,
    delay_cdf: FloatArray | None = None,
    step_func: Callable[[], Any] | None = None,
    thin: int = 1,
    rng: np.random.Generator | int | None = None,
    parallel: bool = True,
    compute_diagnostics: bool = False,
    raise_on_poor_diagnostics: bool = False,
    **kwargs_sample: Any,
) -> xr.Dataset:
    # Quasi-real-time loop: refit using only the data available at each successive calculation
    # time and keep that snapshot's decision-day slice. Each snapshot fit reports every day of
    # its data plus one projected day (`_fit_single_model`'s fixed convention); the decision day is that
    # projected day (its last time index), so each step keeps `time=[-1]`. This matches a single
    # non-quasi-real-time fit's "one day past the data" convention exactly. Two input shapes:
    #   - a single incidence series -> for day t, fit the data up to day t-1 (`incidence[:t]`)
    #     and keep day t (full reporting), sweeping t up to and including the full series so the
    #     final snapshot is all the data (its decision day is the projected current day, day N).
    #     The first snapshot additionally keeps day 0 (the trivial start-of-outbreak point) so the
    #     assembled trajectory spans every day 0..N, matching a single fit on the whole series;
    #   - a sequence of series -> the right-truncated data known at each snapshot (the
    #     under-reporting nowcast). Each series' reporting as-of day is its own last day (see
    #     `_reporting_prob_vec`) and its decision day is one later.
    # Latent-case histories use a per-snapshot `data_time` axis and are omitted from the aggregate;
    # only variables describing the retained decision days are concatenated on `time`. When
    # requested, diagnostics are nevertheless extracted from every R_t and latent-incidence value
    # in each complete snapshot fit before that projection, then aggregated once in the parent.
    # The per-snapshot fits are independent, so with `parallel` they are run across processes via
    # joblib (mirroring calc_additional_case_prob_simulation); each fit then samples chains
    # sequentially (cores=1) so joblib, not pm.sample, owns the parallelism.
    is_incidence_snapshot_sequence = _is_incidence_sequence(incidence)
    extra_kwargs: dict[str, Any] = {"progressbar": False}
    if reporting_prob is None:
        extra_kwargs["quiet"] = True

    # Each step pairs a snapshot incidence series with the time positions of its returned dataset
    # to keep (see above): the decision day `[-1]`, plus day 0 for the first single-series snapshot.
    if is_incidence_snapshot_sequence:
        snapshot_specs: list[tuple[IntArray, list[int]]] = [
            (np.asarray(v), [-1]) for v in incidence
        ]
    else:
        incidence_vec = np.asarray(incidence)
        if len(incidence_vec) < 2:
            raise ValueError(
                "quasi_real_time inference requires at least 2 time points"
            )
        snapshot_specs = [
            (incidence_vec[:t], [0, 1] if t == 1 else [-1])
            for t in range(1, len(incidence_vec) + 1)
        ]

    # One child RNG per fit so results depend on spawn order, not execution order (mirrors
    # calc_additional_case_prob_simulation); serial and parallel then agree exactly. A bare int
    # or None seed is replicated to preserve the previous shared-seed behaviour.
    if isinstance(rng, np.random.Generator):
        child_rngs: list[np.random.Generator | int | None] = list(
            rng.spawn(len(snapshot_specs))
        )
    else:
        child_rngs = [rng] * len(snapshot_specs)

    fit_kwargs_shared: dict[str, Any] = {
        "serial_interval_dist_vec": serial_interval_dist_vec,
        "rep_no_vec_func": rep_no_vec_func,
        "reporting_prob": reporting_prob,
        "delay_cdf": delay_cdf,
        "step_func": step_func,
        "thin": thin,
        "compute_diagnostics": compute_diagnostics,
        **extra_kwargs,
        **kwargs_sample,
    }
    if parallel:
        # Chains sequential within a fit; joblib owns the cross-fit parallelism.
        fit_kwargs_shared = {"cores": 1, **fit_kwargs_shared}
    fit_tasks = [
        (
            idx,
            keep,
            {
                **fit_kwargs_shared,
                "incidence": incidence_step_vec,
                "rng": child_rng,
            },
        )
        for idx, ((incidence_step_vec, keep), child_rng) in enumerate(
            zip(snapshot_specs, child_rngs, strict=True)
        )
    ]

    desc = "Inferring R_t using data up to each time"
    if parallel:
        # inner_max_num_threads pins the numeric threadpools in each worker so a single-core fit
        # doesn't spawn its own pool on top of joblib's process parallelism (loky is required for
        # inner_max_num_threads to take effect).
        with parallel_config(backend="loky", inner_max_num_threads=1):
            fit_results = list(
                tqdm(
                    Parallel(
                        n_jobs=-1,
                        return_as="generator",
                        batch_size="auto",
                    )(
                        delayed(_fit_model_qrt_step)(task, isolate=True)
                        for task in fit_tasks
                    ),
                    total=len(fit_tasks),
                    desc=desc,
                )
            )
    else:
        fit_results = list(
            tqdm(
                (_fit_model_qrt_step(task, isolate=False) for task in fit_tasks),
                total=len(fit_tasks),
                desc=desc,
            )
        )

    posterior_datasets: list[xr.Dataset | None] = [None] * len(fit_tasks)
    rep_no_diagnostic_components: list[_DiagnosticComponents | None] = [None] * len(
        fit_tasks
    )
    incidence_diagnostic_components: list[_DiagnosticComponents | None] = [None] * len(
        fit_tasks
    )
    for (
        idx,
        posterior_subset_ds,
        rep_no_components,
        incidence_components,
    ) in fit_results:
        posterior_datasets[idx] = posterior_subset_ds
        rep_no_diagnostic_components[idx] = rep_no_components
        incidence_diagnostic_components[idx] = incidence_components
    posterior_ds = xr.concat(posterior_datasets, dim="time")
    if compute_diagnostics:
        rep_no_components = [
            component
            for component in rep_no_diagnostic_components
            if component is not None
        ]
        if len(rep_no_components) != len(fit_tasks):
            raise RuntimeError("a QRT snapshot did not return reproduction-number diagnostics")
        diagnostics = _summarize_and_check_diagnostics(
            _combine_diagnostic_components(rep_no_components),
            raise_on_problems=raise_on_poor_diagnostics,
        )
        if reporting_prob is not None:
            incidence_components = [
                component
                for component in incidence_diagnostic_components
                if component is not None
            ]
            if len(incidence_components) != len(fit_tasks):
                raise RuntimeError(
                    "an under-reporting QRT snapshot did not return incidence diagnostics"
                )
            incidence_diagnostics = _summarize_and_check_diagnostics(
                _combine_diagnostic_components(incidence_components),
                raise_on_problems=False,
            )
            diagnostics.update(
                {
                    f"incidence_{key}": incidence_diagnostics[key]
                    for key in ("rhat_max", "ess_min", "ess_median")
                }
            )
        posterior_ds.attrs["diagnostics"] = diagnostics
    return posterior_ds


def _fit_model_qrt_step(
    task: tuple[int, list[int], dict[str, Any]], *, isolate: bool
) -> tuple[
    int,
    xr.Dataset,
    _DiagnosticComponents | None,
    _DiagnosticComponents | None,
]:
    # Fit a single quasi-real-time snapshot, extract diagnostics from its complete result, then
    # keep only time-indexed variables at this snapshot's decision-day position(s) `keep` (see
    # `_fit_model_qrt`). Per-snapshot latent-case histories use `data_time` and are intentionally
    # omitted from the returned aggregate because their lengths differ by snapshot. `isolate` is
    # True in a joblib worker, where the PyTensor compile dir must be made per-process.
    idx, keep, fit_kwargs = task
    if isolate:
        _prepare_qrt_worker()
    fit_result = _fit_single_model_result(**fit_kwargs)
    posterior_current_ds = fit_result.posterior_ds
    posterior_subset_ds = posterior_current_ds[
        [
            var
            for var in posterior_current_ds.data_vars
            if "time" in posterior_current_ds[var].dims
        ]
    ].isel(time=keep)
    return (
        idx,
        cast(xr.Dataset, posterior_subset_ds),
        fit_result.rep_no_diagnostics,
        fit_result.incidence_diagnostics,
    )


# Guards the one-time per-worker environment setup below.
_WORKER_ENV_READY = False


def _prepare_qrt_worker() -> None:
    # Isolate this worker's PyTensor compile dir: the compile FileLock is taken on
    # `config.compiledir/.lock` (read dynamically per compilation), so sharing one dir across
    # workers would serialise the C-compilation phase and defeat the parallelism. Pin the numeric
    # threadpools too, belt-and-braces alongside joblib's inner_max_num_threads.
    global _WORKER_ENV_READY
    if _WORKER_ENV_READY:
        return
    for var in (
        "OMP_NUM_THREADS",
        "OPENBLAS_NUM_THREADS",
        "MKL_NUM_THREADS",
        "RAYON_NUM_THREADS",
    ):
        os.environ.setdefault(var, "1")
    worker_compiledir = (
        Path(pytensor.config.base_compiledir) / f"qrt_worker_{os.getpid()}"
    )
    worker_compiledir.mkdir(parents=True, exist_ok=True)
    pytensor.config.compiledir = worker_compiledir
    # Remove the per-worker compile dir when the worker process exits so they don't accumulate
    # under base_compiledir across runs.
    atexit.register(shutil.rmtree, worker_compiledir, ignore_errors=True)
    _WORKER_ENV_READY = True
