# Note that AI tools were used to generate tests

import warnings
from dataclasses import FrozenInstanceError
from typing import Any

import numpy as np
import pytest
import xarray as xr

import endoutbreakvbd.inference as inf


class _DummyModel:
    def __init__(self, state, coords=None):
        self._state = state
        self._coords = coords

    def __enter__(self):
        self._state["coords"] = self._coords
        return self

    def __exit__(self, exc_type, exc, tb):
        return False


def _setup_minimal_pm_for_fit_model(monkeypatch):
    state = {}

    monkeypatch.setattr(
        inf.pm, "Model", lambda coords=None: _DummyModel(state, coords=coords)
    )

    poisson_calls = []

    def fake_poisson(name, mu, observed):
        poisson_calls.append(
            {"name": name, "mu": np.array(mu), "observed": np.array(observed)}
        )

    monkeypatch.setattr(inf.pm, "Poisson", fake_poisson)

    class _FakeTrace:
        def __init__(self, posterior):
            self.posterior = posterior

        def isel(self, **kwargs):
            self.posterior = self.posterior.isel(**kwargs)
            return self

        def assign_coords(self, coords=None, **kwargs):
            if coords is None:
                coords = {}
            coords = {**coords, **kwargs}
            self.posterior = self.posterior.assign_coords(coords)
            return self

    def fake_sample(**kwargs):
        state["sample_kwargs"] = kwargs
        t_stop = len(state["coords"]["time"])
        posterior = xr.Dataset(
            {
                "rep_no": (
                    ("chain", "draw", "time"),
                    np.ones((1, 6, t_stop), dtype=float),
                )
            },
            coords={"chain": [0], "draw": np.arange(6), "time": np.arange(t_stop)},
        )
        return _FakeTrace(posterior=posterior)

    monkeypatch.setattr(inf.pm, "sample", fake_sample)

    return state, poisson_calls


def _setup_pm_rep_no_depends_on_observed_sum(monkeypatch):
    state: dict[str, Any] = {"observed_sum": 0.0}

    monkeypatch.setattr(
        inf.pm, "Model", lambda coords=None: _DummyModel(state, coords=coords)
    )

    def fake_poisson(name, mu, observed):
        del name, mu
        state["observed_sum"] = float(np.sum(observed))

    monkeypatch.setattr(inf.pm, "Poisson", fake_poisson)

    class _FakeTrace:
        def __init__(self, posterior):
            self.posterior = posterior

        def isel(self, **kwargs):
            self.posterior = self.posterior.isel(**kwargs)
            return self

        def assign_coords(self, coords=None, **kwargs):
            if coords is None:
                coords = {}
            coords = {**coords, **kwargs}
            self.posterior = self.posterior.assign_coords(coords)
            return self

    def fake_sample(**kwargs):
        del kwargs
        t_stop = len(state["coords"]["time"])
        rep_no_value = state["observed_sum"]
        posterior = xr.Dataset(
            {
                "rep_no": (
                    ("chain", "draw", "time"),
                    np.full((1, 2, t_stop), rep_no_value, dtype=float),
                )
            },
            coords={"chain": [0], "draw": np.arange(2), "time": np.arange(t_stop)},
        )
        return _FakeTrace(posterior=posterior)

    monkeypatch.setattr(inf.pm, "sample", fake_sample)


def _setup_pm_with_time_varying_rep_no(monkeypatch):
    state: dict[str, Any] = {}

    monkeypatch.setattr(
        inf.pm, "Model", lambda coords=None: _DummyModel(state, coords=coords)
    )
    monkeypatch.setattr(inf.pm, "Poisson", lambda name, mu, observed: None)

    class _FakeTrace:
        def __init__(self, posterior):
            self.posterior = posterior

        def isel(self, **kwargs):
            self.posterior = self.posterior.isel(**kwargs)
            return self

        def assign_coords(self, coords=None, **kwargs):
            if coords is None:
                coords = {}
            coords = {**coords, **kwargs}
            self.posterior = self.posterior.assign_coords(coords)
            return self

    def fake_sample(**kwargs):
        del kwargs
        t_stop = len(state["coords"]["time"])
        n_draw = 3
        # rep_no[chain, draw, t] = t + 0.1 * draw, so it varies over both time and draw
        time_idx = np.arange(t_stop, dtype=float)
        draw_offset = np.arange(n_draw, dtype=float)[:, None] * 0.1
        values = (time_idx[None, :] + draw_offset)[None, :, :]
        posterior = xr.Dataset(
            {"rep_no": (("chain", "draw", "time"), values)},
            coords={"chain": [0], "draw": np.arange(n_draw), "time": np.arange(t_stop)},
        )
        return _FakeTrace(posterior=posterior)

    monkeypatch.setattr(inf.pm, "sample", fake_sample)
    return state


def _fake_suitability_posterior(t_stop):
    return xr.Dataset(
        {
            "rep_no": (
                ("chain", "draw", "time"),
                np.ones((1, 2, t_stop), dtype=float),
            ),
            "suitability": (
                ("chain", "draw", "time"),
                np.full((1, 2, t_stop), 0.5, dtype=float),
            ),
            "rep_no_factor": (
                ("chain", "draw", "time"),
                np.full((1, 2, t_stop), 2.0, dtype=float),
            ),
        },
        coords={"chain": [0], "draw": np.arange(2), "time": np.arange(t_stop)},
    )


def test_defaults_dataclass_is_frozen():
    defaults = inf.Defaults()
    assert defaults.rep_no_prior_median == 1.0
    assert defaults.log_rep_no_rho == 0.975
    with pytest.raises(FrozenInstanceError):
        setattr(defaults, "log_rep_no_rho", 0.9)


def test_fit_model_raises_when_local_incidence_positive_with_zero_foi():
    with pytest.raises(ValueError, match="force of infection is 0"):
        inf._fit_model(
            incidence_vec=np.array([0, 1]),
            serial_interval_dist_vec=np.array([0.0]),
            rep_no_vec_func=lambda t_stop: np.ones(t_stop),
            quasi_real_time=False,
        )


def test_fit_model_sets_random_seed_uses_step_func_and_thins(monkeypatch):
    state, poisson_calls = _setup_minimal_pm_for_fit_model(monkeypatch)

    rng = np.random.default_rng(0)
    out = inf._fit_model(
        incidence_vec=np.array([1, 0, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        rep_no_vec_func=lambda t_stop: np.ones(t_stop),
        quasi_real_time=False,
        step_func=lambda: "FAKE_STEP",
        thin=2,
        rng=rng,
        draws=6,
        chains=1,
        tune=0,
        progressbar=False,
    )

    assert state["sample_kwargs"]["random_seed"] is rng
    assert state["sample_kwargs"]["step"] == "FAKE_STEP"
    assert out.sizes["draw"] == 3
    np.testing.assert_array_equal(out.coords["draw"].to_numpy(), np.array([0, 1, 2]))

    assert len(poisson_calls) == 1
    assert poisson_calls[0]["name"] == "likelihood"
    np.testing.assert_array_equal(poisson_calls[0]["observed"], np.array([0.0]))


def test_fit_model_quasi_real_time_concatenates_one_time_slice_per_day(monkeypatch):
    _setup_minimal_pm_for_fit_model(monkeypatch)
    monkeypatch.setattr(inf, "tqdm", lambda iterable, **kwargs: iterable)

    out = inf._fit_model(
        incidence_vec=np.array([1, 0, 0, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        rep_no_vec_func=lambda t_stop: np.ones(t_stop),
        quasi_real_time=True,
        draws=6,
        chains=1,
        tune=0,
        progressbar=False,
    )

    assert out.sizes["time"] == 4
    np.testing.assert_array_equal(out.coords["time"].to_numpy(), np.array([0, 1, 2, 3]))


def test_fit_model_quasi_real_time_raises_for_single_time_point():
    with pytest.raises(ValueError, match="at least 2 time points"):
        inf._fit_model(
            incidence_vec=np.array([1]),
            serial_interval_dist_vec=np.array([1.0]),
            rep_no_vec_func=lambda t_stop: np.ones(t_stop),
            quasi_real_time=True,
        )


def test_fit_model_quasi_real_time_excludes_current_day_incidence(monkeypatch):
    _setup_pm_rep_no_depends_on_observed_sum(monkeypatch)
    monkeypatch.setattr(inf, "tqdm", lambda iterable, **kwargs: iterable)

    incidence_base = np.array([1, 1, 0, 0, 0, 0])
    incidence_changed = np.array([1, 1, 0, 2, 0, 0])  # changed at day 3 only
    serial_interval_dist_vec = np.array([0.6, 0.4])

    out_base = inf._fit_model(
        incidence_vec=incidence_base,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=lambda t_stop: np.ones(t_stop),
        quasi_real_time=True,
    )
    out_changed = inf._fit_model(
        incidence_vec=incidence_changed,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=lambda t_stop: np.ones(t_stop),
        quasi_real_time=True,
    )

    # A change at day t should not affect QRT inference at day t or earlier.
    np.testing.assert_allclose(
        out_base["rep_no_mean"].sel(time=np.arange(0, 4)).to_numpy(),
        out_changed["rep_no_mean"].sel(time=np.arange(0, 4)).to_numpy(),
    )
    assert float(out_base["rep_no_mean"].sel(time=4).item()) != float(
        out_changed["rep_no_mean"].sel(time=4).item()
    )


def test_fit_model_freeze_from_final_case_freezes_tail(monkeypatch):
    _setup_pm_with_time_varying_rep_no(monkeypatch)

    out = inf._fit_model(
        incidence_vec=np.array([1, 1, 0, 0, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        rep_no_vec_func=lambda t_stop: np.ones(t_stop),
        quasi_real_time=False,
        freeze_from_final_case=True,
    )

    # Final case is at time index 1; R_t after it is frozen at the time=1 samples.
    final_case_samples = out["rep_no"].isel(time=1).to_numpy()
    for t in (2, 3, 4):
        np.testing.assert_array_equal(
            out["rep_no"].isel(time=t).to_numpy(), final_case_samples
        )
    # Times up to and including the final case remain time-varying (unchanged).
    np.testing.assert_allclose(
        out["rep_no"].isel(time=0).to_numpy().ravel(), np.array([0.0, 0.1, 0.2])
    )
    np.testing.assert_allclose(final_case_samples.ravel(), np.array([1.0, 1.1, 1.2]))
    # The summary mean reflects the frozen tail.
    assert float(out["rep_no_mean"].sel(time=4).item()) == float(
        out["rep_no_mean"].sel(time=1).item()
    )


def test_fit_model_freeze_from_final_case_default_off(monkeypatch):
    _setup_pm_with_time_varying_rep_no(monkeypatch)

    out = inf._fit_model(
        incidence_vec=np.array([1, 1, 0, 0, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        rep_no_vec_func=lambda t_stop: np.ones(t_stop),
        quasi_real_time=False,
    )

    # Without freezing, R_t remains time-varying after the final case.
    assert not np.array_equal(
        out["rep_no"].isel(time=4).to_numpy(), out["rep_no"].isel(time=1).to_numpy()
    )


def test_fit_model_freeze_raises_with_quasi_real_time():
    with pytest.raises(NotImplementedError, match="freeze_from_final_case"):
        inf._fit_model(
            incidence_vec=np.array([1, 0]),
            serial_interval_dist_vec=np.array([1.0]),
            rep_no_vec_func=lambda t_stop: np.ones(t_stop),
            quasi_real_time=True,
            freeze_from_final_case=True,
        )


def test_fit_autoregressive_model_forwards_freeze_from_final_case(monkeypatch):
    monkeypatch.setattr(
        inf,
        "lognormal_params_from_median_percentile_2_5",
        lambda *, median, percentile_2_5: {"mu": 0.0, "sigma": 1.0},
    )
    monkeypatch.setattr(inf, "_fit_model", lambda **kwargs: kwargs)

    out_default = inf.fit_autoregressive_model(
        incidence_vec=np.array([1, 0]),
        serial_interval_dist_vec=np.array([1.0]),
    )
    assert out_default["freeze_from_final_case"] is False

    out_set = inf.fit_autoregressive_model(
        incidence_vec=np.array([1, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        freeze_from_final_case=True,
    )
    assert out_set["freeze_from_final_case"] is True


def test_fit_autoregressive_model_uses_defaults(monkeypatch):
    captured = {}

    def fake_lognormal(*, median, percentile_2_5):
        captured["median"] = median
        captured["percentile_2_5"] = percentile_2_5
        return {"mu": 0.0, "sigma": 1.0}

    def fake_fit_model(**kwargs):
        captured["fit_model_kwargs"] = kwargs
        return "posterior"

    monkeypatch.setattr(
        inf, "lognormal_params_from_median_percentile_2_5", fake_lognormal
    )
    monkeypatch.setattr(inf, "_fit_model", fake_fit_model)

    out = inf.fit_autoregressive_model(
        incidence_vec=np.array([1, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        quasi_real_time=True,
    )

    assert out == "posterior"
    assert captured["median"] == inf.DEFAULTS.rep_no_prior_median
    assert captured["percentile_2_5"] == inf.DEFAULTS.rep_no_prior_percentile_2_5
    assert captured["fit_model_kwargs"]["quasi_real_time"] is True


def test_fit_autoregressive_model_respects_overrides(monkeypatch):
    captured = {}

    def fake_lognormal(*, median, percentile_2_5):
        captured["median"] = median
        captured["percentile_2_5"] = percentile_2_5
        return {"mu": 0.0, "sigma": 1.0}

    monkeypatch.setattr(
        inf, "lognormal_params_from_median_percentile_2_5", fake_lognormal
    )
    monkeypatch.setattr(inf, "_fit_model", lambda **kwargs: kwargs)

    out = inf.fit_autoregressive_model(
        incidence_vec=np.array([1, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        prior_median=2.5,
        prior_percentile_2_5=0.7,
        rho=0.5,
    )

    assert captured["median"] == 2.5
    assert captured["percentile_2_5"] == 0.7
    assert out["quasi_real_time"] is False


def test_fit_suitability_model_uses_defaults(monkeypatch):
    captured = {}

    def fake_lognormal(*, median, percentile_2_5):
        captured["median"] = median
        captured["percentile_2_5"] = percentile_2_5
        return {"mu": 0.0, "sigma": 1.0}

    def fake_fit_model(**kwargs):
        captured["fit_model_kwargs"] = kwargs
        return _fake_suitability_posterior(len(kwargs["incidence_vec"]))

    monkeypatch.setattr(
        inf, "lognormal_params_from_median_percentile_2_5", fake_lognormal
    )
    monkeypatch.setattr(inf, "_fit_model", fake_fit_model)

    out = inf.fit_suitability_model(
        incidence_vec=np.array([1, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        suitability_mean_vec=np.array([0.2, 0.3]),
        quasi_real_time=True,
    )

    assert isinstance(out, xr.Dataset)
    assert captured["median"] == inf.DEFAULTS.rep_no_factor_prior_median
    assert captured["percentile_2_5"] == inf.DEFAULTS.rep_no_factor_prior_percentile_2_5
    assert captured["fit_model_kwargs"]["quasi_real_time"] is True


def test_fit_suitability_model_respects_overrides(monkeypatch):
    captured = {}

    def fake_lognormal(*, median, percentile_2_5):
        captured["median"] = median
        captured["percentile_2_5"] = percentile_2_5
        return {"mu": 0.0, "sigma": 1.0}

    monkeypatch.setattr(
        inf, "lognormal_params_from_median_percentile_2_5", fake_lognormal
    )

    def fake_fit_model(**kwargs):
        captured["fit_model_kwargs"] = kwargs
        return _fake_suitability_posterior(len(kwargs["incidence_vec"]))

    monkeypatch.setattr(inf, "_fit_model", fake_fit_model)

    inf.fit_suitability_model(
        incidence_vec=np.array([1, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        suitability_mean_vec=np.array([0.2, 0.3]),
        suitability_std=0.2,
        suitability_rho=0.3,
        rep_no_factor_prior_median=2.0,
        rep_no_factor_prior_percentile_2_5=0.9,
        log_rep_no_factor_rho=0.1,
    )

    assert captured["median"] == 2.0
    assert captured["percentile_2_5"] == 0.9
    assert captured["fit_model_kwargs"]["quasi_real_time"] is False


def test_fit_suitability_model_rep_no_func_uses_softclip(monkeypatch):
    softclip_calls = []

    def fake_softclip(x, *, lower, upper, tau=0.001):
        softclip_calls.append((np.array(x), lower, upper))
        return np.clip(x, lower, upper)

    monkeypatch.setattr(inf, "_softclip", fake_softclip)
    monkeypatch.setattr(inf.pm, "AR", lambda *args, **kwargs: np.zeros(4))
    monkeypatch.setattr(
        inf.pm, "Deterministic", lambda name, value, dims=None: np.array(value)
    )

    class _FakeMath:
        @staticmethod
        def exp(x):
            return np.exp(x)

    monkeypatch.setattr(inf.pm, "math", _FakeMath)

    class _FakeNormal:
        @staticmethod
        def dist(mu, sigma):
            return 0.0

    monkeypatch.setattr(inf.pm, "Normal", _FakeNormal)

    captured = {}

    def fake_fit_model(**kwargs):
        rep_no_vec = kwargs["rep_no_vec_func"](4)
        captured["rep_no_vec"] = rep_no_vec
        return _fake_suitability_posterior(len(kwargs["incidence_vec"]))

    monkeypatch.setattr(inf, "_fit_model", fake_fit_model)

    out = inf.fit_suitability_model(
        incidence_vec=np.array([1, 0, 0, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        suitability_mean_vec=np.array([0.1, 0.2, 0.3, 0.4]),
    )

    assert isinstance(out, xr.Dataset)
    assert len(softclip_calls) == 1
    softclip_arr, lower, upper = softclip_calls[0]
    np.testing.assert_allclose(softclip_arr, np.array([0.1, 0.2, 0.3, 0.4]))
    assert lower == 1e-8
    assert upper == 1.0
    assert captured["rep_no_vec"].shape == (4,)


def _diag_posterior():
    # Stand-in posterior; rhat/ess are monkeypatched so its contents are irrelevant.
    return xr.Dataset(
        {"rep_no": (("chain", "draw", "time"), np.ones((2, 4, 2)))},
        coords={"chain": [0, 1], "draw": np.arange(4), "time": np.arange(2)},
    )


def _sample_stats(diverging):
    return xr.Dataset(
        {"diverging": (("chain", "draw"), np.asarray(diverging, dtype=bool))}
    )


def _patch_stats(monkeypatch, *, rhat_values, ess_values):
    monkeypatch.setattr(
        inf,
        "rhat",
        lambda posterior, var_names: xr.Dataset(
            {var_names: ("time", np.asarray(rhat_values, dtype=float))}
        ),
    )
    monkeypatch.setattr(
        inf,
        "ess",
        lambda posterior, var_names: xr.Dataset(
            {var_names: ("time", np.asarray(ess_values, dtype=float))}
        ),
    )


def test_compute_check_diagnostics_good_returns_dict_no_warning(monkeypatch):
    _patch_stats(monkeypatch, rhat_values=[1.001, 1.002], ess_values=[2000.0, 3000.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics = inf._compute_check_diagnostics(
            _diag_posterior(), _sample_stats([[False, False], [False, False]])
        )
    assert not [w for w in caught if "Poor sampling" in str(w.message)]
    assert set(diagnostics) == {
        "rhat_mean",
        "rhat_median",
        "rhat_max",
        "ess_mean",
        "ess_median",
        "ess_min",
        "n_diverging",
    }
    assert diagnostics["ess_min"] == 2000.0
    assert diagnostics["rhat_max"] == 1.002
    assert diagnostics["n_diverging"] == 0.0


def test_compute_check_diagnostics_warns_when_not_raising(monkeypatch):
    _patch_stats(monkeypatch, rhat_values=[1.0, 1.0], ess_values=[500.0, 600.0])
    with pytest.warns(UserWarning, match="min ESS 500.0 < 1000"):
        diagnostics = inf._compute_check_diagnostics(
            _diag_posterior(), _sample_stats([[False, False], [False, False]])
        )
    assert diagnostics["ess_min"] == 500.0


def test_compute_check_diagnostics_raises_when_requested(monkeypatch):
    _patch_stats(monkeypatch, rhat_values=[1.02, 1.005], ess_values=[500.0, 2000.0])
    with pytest.raises(RuntimeError, match="min ESS .* max R-hat"):
        inf._compute_check_diagnostics(
            _diag_posterior(),
            _sample_stats([[True, False], [False, False]]),
            raise_on_problems=True,
        )


def test_compute_check_diagnostics_reports_divergences(monkeypatch):
    _patch_stats(monkeypatch, rhat_values=[1.0, 1.0], ess_values=[2000.0, 3000.0])
    with pytest.warns(UserWarning, match="1 divergence"):
        inf._compute_check_diagnostics(
            _diag_posterior(), _sample_stats([[True, False], [False, False]])
        )


def test_compute_check_diagnostics_divergences_na_when_no_sample_stats(monkeypatch):
    _patch_stats(monkeypatch, rhat_values=[1.0, 1.0], ess_values=[2000.0, 3000.0])
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        diagnostics = inf._compute_check_diagnostics(_diag_posterior(), None)
    assert not [w for w in caught if "Poor sampling" in str(w.message)]
    assert np.isnan(diagnostics["n_diverging"])
