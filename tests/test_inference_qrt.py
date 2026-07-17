# Note that AI tools were used to generate tests

import numpy as np
import xarray as xr

import endoutbreakvbd._inference_qrt as qrt
import endoutbreakvbd.inference as inference
import endoutbreakvbd.rep_no_models as rnm


def test_fit_model_qrt_sequence_mode_forwards_per_snapshot(monkeypatch):
    # The sequence-of-series entry (under-reporting nowcast) refits at each snapshot with the
    # matching right-truncated series, reporting_prob and delay_cdf, keeps that snapshot's
    # projected decision day (one past its data), and concatenates the per-snapshot time slices.
    calls = []

    def fake_fit_model(**kwargs):
        calls.append(kwargs)
        t = len(
            kwargs["incidence"]
        )  # projected decision day = one past the snapshot data
        return xr.Dataset(
            {
                "rep_no_mean": ("time", [float(t)]),
                "additional_case_prob": ("time", [0.1 * t]),
                "incidence": ("data_time", np.arange(t, dtype=float)),
                "incidence_mean": ("data_time", np.arange(t, dtype=float)),
            },
            coords={"time": [t], "data_time": np.arange(t)},
        )

    monkeypatch.setattr(qrt, "_fit_model", fake_fit_model)
    monkeypatch.setattr(qrt, "tqdm", lambda iterable, **kwargs: iterable)

    incidence_snapshots = [np.array([2, 1]), np.array([2, 1, 0, 0])]
    delay_cdf = np.array([0.5, 1.0])
    out = qrt._fit_model_qrt(
        incidence=incidence_snapshots,
        serial_interval_dist_vec=np.array([1.0]),
        rep_no_vec_func=lambda t_stop: np.ones(t_stop),
        reporting_prob=0.6,
        delay_cdf=delay_cdf,
        parallel=False,
    )

    assert all(c["reporting_prob"] == 0.6 for c in calls)
    assert all(np.array_equal(c["delay_cdf"], delay_cdf) for c in calls)
    np.testing.assert_array_equal(calls[0]["incidence"], incidence_snapshots[0])
    np.testing.assert_array_equal(calls[1]["incidence"], incidence_snapshots[1])
    # progressbar suppressed, but the full-reporting-only `quiet` flag is not injected here.
    assert calls[0]["progressbar"] is False
    assert "quiet" not in calls[0]
    # Each snapshot keeps its projected decision day = len(snapshot); concatenated in order.
    np.testing.assert_array_equal(out.coords["time"].to_numpy(), np.array([2, 4]))
    np.testing.assert_allclose(out["additional_case_prob"].to_numpy(), [0.2, 0.4])
    assert "data_time" not in out.dims
    assert "incidence" not in out
    assert "incidence_mean" not in out


def test_fit_autoregressive_model_routes_sequence_through_qrt(monkeypatch):
    # The public entry point forwards a sequence of series into the quasi-real-time sequence path
    # (the under-reporting nowcast route), reached via _fit_model's deferred import.
    captured = {}

    def fake_fit_model_qrt(**kwargs):
        captured.update(kwargs)
        return xr.Dataset({"rep_no_mean": ("time", [1.0])}, coords={"time": [1]})

    monkeypatch.setattr(qrt, "_fit_model_qrt", fake_fit_model_qrt)

    incidence_snapshots = [np.array([2, 1]), np.array([2, 1, 0, 0])]
    inference.fit_autoregressive_model(
        incidence=incidence_snapshots,
        serial_interval_dist_vec=np.array([1.0]),
        quasi_real_time=True,
        compute_diagnostics=False,
    )

    assert len(captured["incidence"]) == 2


def test_fit_model_qrt_spawns_distinct_child_rng_per_step(monkeypatch):
    # Each snapshot fit must receive its own spawned child RNG (not the single shared generator),
    # so results depend on spawn order rather than execution order and serial == parallel. The
    # per-snapshot slices are also reassembled in calc-time order.
    seen_rngs = []

    def fake_fit_model(**kwargs):
        seen_rngs.append(kwargs["rng"])
        t = len(kwargs["incidence"])
        return xr.Dataset({"rep_no_mean": ("time", [float(t)])}, coords={"time": [t]})

    monkeypatch.setattr(qrt, "_fit_model", fake_fit_model)
    monkeypatch.setattr(qrt, "tqdm", lambda iterable, **kwargs: iterable)

    parent_rng = np.random.default_rng(0)
    out = qrt._fit_model_qrt(
        incidence=[np.array([1, 1]), np.array([1, 1, 0]), np.array([1, 1, 0, 0])],
        serial_interval_dist_vec=np.array([1.0]),
        rep_no_vec_func=lambda t_stop: np.ones(t_stop),
        reporting_prob=0.6,
        rng=parent_rng,
        parallel=False,
    )

    assert len(seen_rngs) == 3
    assert all(isinstance(r, np.random.Generator) for r in seen_rngs)
    assert len({id(r) for r in seen_rngs}) == 3
    assert all(r is not parent_rng for r in seen_rngs)
    np.testing.assert_array_equal(out.coords["time"].to_numpy(), np.array([2, 3, 4]))


def test_fit_model_qrt_serial_matches_parallel():
    # A small real under-reporting nowcast fit must give identical results whether the snapshots
    # are fitted serially or across joblib worker processes: the spawned child RNGs make the
    # output independent of execution order, and cores=1 keeps each fit single-process.
    serial_interval_dist_vec = np.array([0.4, 0.4, 0.2])
    rep_no_vec_func = rnm.build_known_rep_no(
        rep_no_func=lambda t: np.full(np.shape(t), 0.7, dtype=float)
    )
    incidence_snapshots = [
        np.array([1, 2, 1, 0, 0]),
        np.array([1, 2, 1, 1, 0, 0, 0]),
    ]

    def fit(parallel):
        return qrt._fit_model_qrt(
            incidence=incidence_snapshots,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rep_no_vec_func=rep_no_vec_func,
            reporting_prob=0.8,
            rng=np.random.default_rng(3),
            parallel=parallel,
            draws=50,
            tune=50,
            chains=2,
            cores=1,
        )

    serial_ds = fit(parallel=False)
    parallel_ds = fit(parallel=True)

    np.testing.assert_allclose(
        serial_ds["additional_case_prob"].to_numpy(),
        parallel_ds["additional_case_prob"].to_numpy(),
    )
    np.testing.assert_allclose(
        serial_ds["rep_no_mean"].to_numpy(),
        parallel_ds["rep_no_mean"].to_numpy(),
    )
