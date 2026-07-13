# Note that AI tools were used to generate tests

import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import endoutbreakvbd._inference_models as im
import endoutbreakvbd.inference as inf
from endoutbreakvbd.rep_no_models import build_ar_rep_no, build_known_rep_no
from scripts.lazio_underreporting_retro import _write_results


class _CtxModel:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTrace:
    def __init__(self, posterior):
        self.posterior = posterior


def test_convolution_matrix_matches_renewal_foi():
    # C @ cases must reproduce the renewal force of infection
    # foi[s] = sum_{r<s} cases[r] * w[s-1-r], as used by the full-reporting model.
    serial_interval_dist_vec = np.array([0.5, 0.3, 0.15, 0.05])
    t = 8
    cases = np.array([2.0, 1.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    conv_mat = im._renewal_convolution_matrix(serial_interval_dist_vec, t)

    w_ext = np.concatenate(
        [
            serial_interval_dist_vec,
            np.zeros(max(t - 1 - len(serial_interval_dist_vec), 0)),
        ]
    )
    foi_expected = np.zeros(t)
    for s in range(1, t):
        foi_expected[s] = np.sum(cases[:s][::-1] * w_ext[:s])

    np.testing.assert_allclose(conv_mat @ cases, foi_expected)
    # Lower-triangular: current-day cases never contribute to their own FOI.
    assert np.allclose(np.triu(conv_mat), 0.0)


def test_convolution_matrix_serial_interval_longer_than_series():
    # w longer than the padded band is simply truncated by the s-length slice.
    serial_interval_dist_vec = np.array([0.6, 0.3, 0.1])
    conv_mat = im._renewal_convolution_matrix(serial_interval_dist_vec, 3)
    np.testing.assert_allclose(conv_mat[2, :2], np.array([0.3, 0.6]))
    assert conv_mat[1, 0] == pytest.approx(0.6)


def test_reporting_prob_vec_constant_when_no_delay():
    vec = im._reporting_prob_vec(np.array([3, 1, 0, 0]), 0.6, delay_cdf=None)
    np.testing.assert_allclose(vec, np.full(4, 0.6))


def test_reporting_prob_vec_truncates_recent_onsets():
    # Snapshot on the last data day (index 3): available delay = 3 - onset_day. Recent onsets
    # are truncated toward zero; the earliest onset has plateaued at the ceiling.
    delay_cdf = np.array([0.0, 0.4, 0.7, 1.0])
    vec = im._reporting_prob_vec(np.zeros(4, dtype=int), 0.5, delay_cdf=delay_cdf)
    # onset 0 -> avail 3 -> cdf 1.0 (plateau); onset 3 -> avail 0 -> cdf 0.0.
    np.testing.assert_allclose(vec, 0.5 * np.array([1.0, 0.7, 0.4, 0.0]))
    # Non-decreasing from recent (last) to old (first) onset.
    assert np.all(np.diff(vec[::-1]) >= 0)


@pytest.mark.parametrize("reporting_prob", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_reporting_prob_vec_rejects_invalid_reporting_probability(reporting_prob):
    with pytest.raises(ValueError, match="reporting_prob must be finite and in"):
        im._reporting_prob_vec(np.array([1, 0]), reporting_prob, delay_cdf=None)


@pytest.mark.parametrize(
    "delay_cdf",
    [
        np.array([]),
        np.array([[0.0, 1.0]]),
        np.array([0.0, np.nan]),
        np.array([-0.1, 1.0]),
        np.array([0.0, 1.1]),
        np.array([0.0, 0.8, 0.7]),
    ],
)
def test_reporting_prob_vec_rejects_invalid_delay_cdf(delay_cdf):
    with pytest.raises(ValueError, match="delay_cdf must be a non-empty"):
        im._reporting_prob_vec(np.array([1, 0]), 0.6, delay_cdf=delay_cdf)


def test_reproduction_number_horizon_full_vs_underreporting():
    # Under-reporting projects R_t a full serial interval past *all* data (the latent final
    # case can sit anywhere); full reporting only past the last *observed* case. Same inputs,
    # different horizon.
    incidence_vec = np.array(
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    )  # t_data_to=10, last case idx 1
    t_calc = np.array([0, 5])
    assert (
        inf._reproduction_number_horizon(
            incidence_vec=incidence_vec,
            serial_interval_max=4,
            t_calc=t_calc,
            underreporting_fit=True,
        )
        == 14
    )
    assert (
        inf._reproduction_number_horizon(
            incidence_vec=incidence_vec,
            serial_interval_max=4,
            t_calc=t_calc,
            underreporting_fit=False,
        )
        == 10
    )
    # Never short of the latest calculation time.
    assert (
        inf._reproduction_number_horizon(
            incidence_vec=np.ones(3, dtype=int),
            serial_interval_max=2,
            t_calc=np.array([0, 20]),
            underreporting_fit=True,
        )
        == 21
    )


def test_build_underreporting_model_structure():
    obs = np.array([2, 1, 1, 0, 0])
    serial_interval_dist_vec = np.array([0.4, 0.3, 0.2, 0.1])
    model = inf._build_underreporting_model(
        incidence_vec=obs,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=build_ar_rep_no(),
        reporting_prob=0.6,
        delay_cdf=None,
        t_infer_rep_no_to=len(obs) + len(serial_interval_dist_vec),
    )
    # R_t uses the projection horizon, cases use the data window, and the latent unreported
    # counts exclude the fixed index-case day.
    time_coord = model.coords["time"]
    data_time_coord = model.coords["data_time"]
    unobserved_time_coord = model.coords["unobserved_time"]
    assert (
        time_coord is not None
        and data_time_coord is not None
        and unobserved_time_coord is not None
    )
    assert len(time_coord) == len(obs) + len(serial_interval_dist_vec)
    assert list(data_time_coord) == list(range(len(obs)))
    assert list(unobserved_time_coord) == list(range(1, len(obs)))
    assert model["unobserved"].name == "unobserved"
    assert model.named_vars_to_dims["cases"] == ("data_time",)
    assert model["cases"].type.shape == (len(obs),)
    assert {"cases", "obs", "unobserved"}.issubset(
        {v.name for v in model.basic_RVs + model.deterministics}
    )


def test_underreporting_model_p1_collapses_latent_to_zero():
    # With reporting_prob=1 (no delay), no cases are unreported, so the latent U is
    # forced to zero (cases == observed). Checked via the model logp being maximal at U=0.
    obs = np.array([2, 1, 1, 0, 0])
    serial_interval_dist_vec = np.array([0.4, 0.3, 0.2, 0.1])
    model = inf._build_underreporting_model(
        incidence_vec=obs,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=build_ar_rep_no(),
        reporting_prob=1.0,
        delay_cdf=None,
        t_infer_rep_no_to=len(obs) + len(serial_interval_dist_vec),
    )
    logp_fn = model.compile_logp(sum=True)
    point = model.initial_point()
    latent_key = next(k for k in point if "unobserved" in k)
    point_zero = {**point, latent_key: np.zeros_like(point[latent_key])}
    point_nonzero = {**point, latent_key: np.zeros_like(point[latent_key])}
    point_nonzero[latent_key][0] = 5
    assert float(logp_fn(point_zero)) > float(logp_fn(point_nonzero))


def test_underreporting_model_rejects_zero_index_case():
    # The incidence series must start on the index-case day; leading zero days are ambiguous.
    obs = np.array([0, 3, 1, 0])
    serial_interval_dist_vec = np.array([0.6, 0.4])
    with pytest.raises(ValueError, match="starting with at least one index case"):
        inf._build_underreporting_model(
            incidence_vec=obs,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rep_no_vec_func=build_ar_rep_no(),
            reporting_prob=0.6,
            delay_cdf=None,
            t_infer_rep_no_to=len(obs) + len(serial_interval_dist_vec),
        )


def test_build_known_rep_no_registers_fixed_reproduction_number():
    # The builder registers a "rep_no" deterministic fixed to the supplied function of time
    # (constant across draws), evaluated over the inference horizon.
    def rep_no_func(t):
        return 2.0 - 0.1 * np.asarray(t, dtype=float)

    rep_no_vec_func = build_known_rep_no(rep_no_func=rep_no_func)
    with pm.Model(coords={"time": np.arange(5)}):
        rep_no = rep_no_vec_func(5)
    np.testing.assert_allclose(rep_no.eval(), rep_no_func(np.arange(5)))


def test_fit_model_dispatches_to_underreporting_offshoot(monkeypatch):
    # A scalar reporting_prob routes _fit_model through the under-reporting builder (not the
    # full-reporting one), avoids nutpie and any explicit step (pm.sample auto-assigns the
    # latent's Metropolis step), and returns cases summaries + a per-t_calc additional-case
    # probability.
    obs = np.array([2, 1, 1, 0, 0])
    t_infer = 9
    captured: dict = {}

    def fake_build(**kwargs):
        captured["build_kwargs"] = kwargs
        return _CtxModel()

    def fake_build_full(**kwargs):
        raise AssertionError(
            "full-reporting model should not be built for the offshoot"
        )

    monkeypatch.setattr(inf, "_build_underreporting_model", fake_build)
    monkeypatch.setattr(inf, "_build_full_reporting_model", fake_build_full)

    def fake_sample(**kwargs):
        captured["sample_kwargs"] = kwargs
        cases = np.tile(np.arange(len(obs), dtype=float), (1, 3, 1))
        posterior = xr.Dataset(
            {
                "rep_no": (("chain", "draw", "time"), np.ones((1, 3, t_infer))),
                "cases": (("chain", "draw", "data_time"), cases),
            },
            coords={
                "chain": [0],
                "draw": np.arange(3),
                "time": np.arange(t_infer),
                "data_time": np.arange(len(obs)),
            },
        )
        return _FakeTrace(posterior)

    monkeypatch.setattr(inf.pm, "sample", fake_sample)

    def fake_prob(
        *, incidence_vec, rep_no_func, serial_interval_dist_vec, t_calc, additional_dims
    ):
        captured["prob_incidence"] = incidence_vec
        captured["additional_dims"] = additional_dims
        return np.full(np.atleast_1d(t_calc).size, 0.3)

    monkeypatch.setattr(inf, "calc_additional_case_prob_analytical", fake_prob)

    out = inf._fit_model(
        incidence_vec=obs,
        serial_interval_dist_vec=np.array([0.4, 0.3, 0.2, 0.1]),
        rep_no_vec_func=lambda t: np.ones(t),
        quasi_real_time=False,
        reporting_prob=0.6,
    )

    assert "build_kwargs" in captured
    # No explicit step is attached; pm.sample assigns the latent's Metropolis step itself.
    assert "step" not in captured["sample_kwargs"]
    assert "nuts_sampler" not in captured["sample_kwargs"]
    assert captured["sample_kwargs"]["draws"] == 4000
    # The probability uses only the unpadded latent-derived case array (data_time, chain, draw).
    assert np.issubdtype(captured["prob_incidence"].dtype, np.integer)
    assert captured["prob_incidence"].shape == (len(obs), 1, 3)
    # Per-sample probabilities are requested so a credible interval can be formed.
    assert captured["additional_dims"] == "broadcast"
    # Risk and R_t span every day plus one projected day; inferred cases stop at the data boundary.
    assert {"cases_mean", "cases_lower", "cases_upper"}.issubset(out.data_vars)
    assert {"additional_case_prob_lower", "additional_case_prob_upper"}.issubset(
        out.data_vars
    )
    np.testing.assert_array_equal(
        out.coords["time"].to_numpy(), np.arange(len(obs) + 1)
    )
    np.testing.assert_array_equal(
        out.coords["data_time"].to_numpy(), np.arange(len(obs))
    )
    assert out["rep_no"].dims == ("chain", "draw", "time")
    assert out["cases"].dims == ("chain", "draw", "data_time")
    assert out["cases_mean"].dims == ("data_time",)
    assert out["additional_case_prob"].dims == ("time",)


def test_underreporting_retro_csv_reindexes_cases_to_projected_time(tmp_path):
    n_data = 3
    time = np.arange(n_data + 1)
    data_time = np.arange(n_data)
    ds = xr.Dataset(
        {
            "cases_mean": ("data_time", [1.0, 2.0, 3.0]),
            "cases_lower": ("data_time", [0.5, 1.5, 2.5]),
            "cases_upper": ("data_time", [1.5, 2.5, 3.5]),
            "rep_no_mean": ("time", [0.9, 0.8, 0.7, 0.6]),
            "rep_no_lower": ("time", [0.7, 0.6, 0.5, 0.4]),
            "rep_no_upper": ("time", [1.1, 1.0, 0.9, 0.8]),
            "additional_case_prob": ("time", [1.0, 0.7, 0.3, 0.1]),
        },
        coords={"time": time, "data_time": data_time},
    )
    save_path = tmp_path / "underreporting.csv"

    _write_results(
        ds,
        save_path,
        np.array([1, 2, 0]),
        pd.Timestamp("2017-01-01"),
        suitability=False,
    )

    result = pd.read_csv(save_path)
    assert len(result) == n_data + 1
    assert result["reported"].iloc[:-1].notna().all()
    assert np.isnan(result["reported"].iloc[-1])
    assert (
        result[["cases_mean", "cases_lower", "cases_upper"]]
        .iloc[:-1]
        .notna()
        .all()
        .all()
    )
    assert result[["cases_mean", "cases_lower", "cases_upper"]].iloc[-1].isna().all()
    assert result["additional_case_prob"].iloc[-1] == pytest.approx(0.1)


@pytest.mark.parametrize("quasi_real_time", [False, True])
def test_fit_model_rejects_step_func_with_underreporting(quasi_real_time):
    with pytest.raises(
        ValueError, match="step_func is not supported with reporting_prob"
    ):
        inf._fit_model(
            incidence_vec=np.array([1, 0]),
            serial_interval_dist_vec=np.array([1.0]),
            rep_no_vec_func=lambda t: np.ones(t),
            quasi_real_time=quasi_real_time,
            reporting_prob=0.6,
            step_func=lambda: object(),
        )
