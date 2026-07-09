# Note that AI tools were used to generate tests

import numpy as np
import pymc as pm
import pytest
import xarray as xr

import endoutbreakvbd.inference as inf
from endoutbreakvbd.rep_no_models import build_ar_rep_no, build_known_rep_no
from endoutbreakvbd.utils import renewal_convolution_matrix


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
    conv_mat = renewal_convolution_matrix(serial_interval_dist_vec, t)

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
    conv_mat = renewal_convolution_matrix(serial_interval_dist_vec, 3)
    np.testing.assert_allclose(conv_mat[2, :2], np.array([0.3, 0.6]))
    assert conv_mat[1, 0] == pytest.approx(0.6)


def test_reporting_prob_vec_constant_when_no_delay():
    vec = inf._reporting_prob_vec(np.array([3, 1, 0, 0]), 0.6, delay_cdf=None)
    np.testing.assert_allclose(vec, np.full(4, 0.6))


def test_reporting_prob_vec_truncates_recent_onsets():
    # Snapshot on the last data day (index 3): available delay = 3 - onset_day. Recent onsets
    # are truncated toward zero; the earliest onset has plateaued at the ceiling.
    delay_cdf = np.array([0.0, 0.4, 0.7, 1.0])
    vec = inf._reporting_prob_vec(np.zeros(4, dtype=int), 0.5, delay_cdf=delay_cdf)
    # onset 0 -> avail 3 -> cdf 1.0 (plateau); onset 3 -> avail 0 -> cdf 0.0.
    np.testing.assert_allclose(vec, 0.5 * np.array([1.0, 0.7, 0.4, 0.0]))
    # Non-decreasing from recent (last) to old (first) onset.
    assert np.all(np.diff(vec[::-1]) >= 0)


@pytest.mark.parametrize("reporting_prob", [0.0, -0.1, 1.1, np.nan, np.inf])
def test_reporting_prob_vec_rejects_invalid_reporting_probability(reporting_prob):
    with pytest.raises(ValueError, match="reporting_prob must be finite and in"):
        inf._reporting_prob_vec(
            np.array([1, 0]), reporting_prob, delay_cdf=None
        )


@pytest.mark.parametrize(
    ("delay_cdf", "message"),
    [
        (np.array([]), "non-empty 1-D"),
        (np.array([[0.0, 1.0]]), "non-empty 1-D"),
        (np.array([0.0, np.nan]), "only finite"),
        (np.array([-0.1, 1.0]), "interval \\[0, 1\\]"),
        (np.array([0.0, 1.1]), "interval \\[0, 1\\]"),
        (np.array([0.0, 0.8, 0.7]), "non-decreasing"),
    ],
)
def test_reporting_prob_vec_rejects_invalid_delay_cdf(delay_cdf, message):
    with pytest.raises(ValueError, match=message):
        inf._reporting_prob_vec(
            np.array([1, 0]), 0.6, delay_cdf=delay_cdf
        )


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
    # R_t horizon = t_data_to + serial_interval_max; latents live on gen_time = 1..D-1.
    time_coord = model.coords["time"]
    gen_time_coord = model.coords["gen_time"]
    assert time_coord is not None and gen_time_coord is not None
    assert len(time_coord) == len(obs) + len(serial_interval_dist_vec)
    assert list(gen_time_coord) == list(range(1, len(obs)))
    assert model["unobserved"].name == "unobserved"
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
    with pytest.raises(ValueError, match="must start with at least one index case"):
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
        cases = np.tile(np.arange(t_infer, dtype=float), (1, 3, 1))
        posterior = xr.Dataset(
            {
                "rep_no": (("chain", "draw", "time"), np.ones((1, 3, t_infer))),
                "cases": (("chain", "draw", "time"), cases),
            },
            coords={"chain": [0], "draw": np.arange(3), "time": np.arange(t_infer)},
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
    assert captured["sample_kwargs"]["draws"] == 2000
    # The probability is computed from the integer latent-derived case array (time, chain, draw).
    assert np.issubdtype(captured["prob_incidence"].dtype, np.integer)
    assert captured["prob_incidence"].shape == (t_infer, 1, 3)
    # Per-sample probabilities are requested so a credible interval can be formed.
    assert captured["additional_dims"] == "broadcast"
    # Offshoot returns latent-case summaries and slices output to the default t_calc.
    assert {"cases_mean", "cases_lower", "cases_upper"}.issubset(out.data_vars)
    assert {"additional_case_prob_lower", "additional_case_prob_upper"}.issubset(
        out.data_vars
    )
    np.testing.assert_array_equal(out.coords["time"].to_numpy(), np.arange(len(obs)))
    assert out["additional_case_prob"].dims == ("time",)
