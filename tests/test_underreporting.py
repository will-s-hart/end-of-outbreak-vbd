# Note that AI tools were used to generate tests

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
import pytest
import xarray as xr

import endoutbreakvbd._inference_core as core
import endoutbreakvbd._inference_models as im
import endoutbreakvbd.inference as inference
import scripts.lazio_underreporting_retro as lazio_underreporting_retro
from endoutbreakvbd.rep_no_models import build_ar_rep_no, build_known_rep_no
from scripts.lazio_underreporting_retro import _write_results
from scripts.lazio_underreporting_retro_plots import (
    _make_incidence_plot,
    _make_parameter_estimate_plots,
)


class _CtxModel:
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


class _FakeTrace:
    def __init__(self, posterior):
        self.posterior = posterior


def test_lazio_underreporting_retro_raises_on_poor_diagnostics(monkeypatch):
    calls = []
    inputs = {
        "incidence_vec": np.array([1, 0]),
        "serial_interval_dist_vec": np.array([1.0]),
        "outbreak_start_date": pd.Timestamp("2017-01-01"),
        "reporting_prob": 0.6,
        "suitability_mean_vec": np.ones(3),
        "results_paths": {
            "suitability_p60": "suitability.csv",
            "suitability_p60_diagnostics": "suitability_diagnostics.csv",
            "autoregressive_p60": "autoregressive.csv",
            "autoregressive_p60_diagnostics": "autoregressive_diagnostics.csv",
        },
    }

    def fake_fit(**kwargs):
        calls.append(kwargs)
        return object()

    monkeypatch.setattr(
        lazio_underreporting_retro,
        "get_inputs_lazio_underreporting_retro",
        lambda: inputs,
    )
    monkeypatch.setattr(lazio_underreporting_retro, "fit_suitability_model", fake_fit)
    monkeypatch.setattr(lazio_underreporting_retro, "fit_autoregressive_model", fake_fit)
    monkeypatch.setattr(lazio_underreporting_retro, "_write_results", lambda *a, **k: None)
    monkeypatch.setattr(
        lazio_underreporting_retro, "_write_diagnostics", lambda *a, **k: None
    )

    lazio_underreporting_retro.run_analyses()

    assert len(calls) == 2
    assert all(call["compute_diagnostics"] is True for call in calls)
    assert all(call["raise_on_poor_diagnostics"] is True for call in calls)


def test_convolution_matrix_matches_renewal_foi():
    # C @ incidence must reproduce the renewal force of infection
    # foi[t] = sum_{r<t} incidence[r] * serial_interval[t-1-r], as used by the model.
    serial_interval_dist_vec = np.array([0.5, 0.3, 0.15, 0.05])
    n_days = 8
    incidence_vec = np.array([2.0, 1.0, 3.0, 0.0, 1.0, 0.0, 0.0, 0.0])
    conv_mat = im._renewal_convolution_matrix(serial_interval_dist_vec, n_days)

    serial_interval_ext_vec = np.concatenate(
        [
            serial_interval_dist_vec,
            np.zeros(max(n_days - 1 - len(serial_interval_dist_vec), 0)),
        ]
    )
    foi_expected_vec = np.zeros(n_days)
    for t in range(1, n_days):
        foi_expected_vec[t] = np.sum(
            incidence_vec[:t][::-1] * serial_interval_ext_vec[:t]
        )

    np.testing.assert_allclose(conv_mat @ incidence_vec, foi_expected_vec)
    # Lower-triangular: current-day cases never contribute to their own FOI.
    assert np.allclose(np.triu(conv_mat), 0.0)


def test_convolution_matrix_serial_interval_longer_than_series():
    # A serial-interval distribution longer than the padded band is simply truncated by the
    # time-length slice.
    serial_interval_dist_vec = np.array([0.6, 0.3, 0.1])
    conv_mat = im._renewal_convolution_matrix(serial_interval_dist_vec, 3)
    np.testing.assert_allclose(conv_mat[2, :2], np.array([0.3, 0.6]))
    assert conv_mat[1, 0] == pytest.approx(0.6)


def test_reporting_prob_vec_constant_when_no_delay():
    vec = im._reporting_prob_vec(np.array([3, 1, 0, 0]), 0.6, delay_cdf=None)
    np.testing.assert_allclose(vec, np.full(4, 0.6))


def test_reporting_prob_vec_truncates_recent_onsets():
    # Snapshot at the last data time (index 3): available delay = 3 - t_onset. Recent onsets
    # are truncated toward zero; the earliest onset has plateaued at the ceiling.
    delay_cdf = np.array([0.0, 0.4, 0.7, 1.0])
    vec = im._reporting_prob_vec(np.zeros(4, dtype=int), 0.5, delay_cdf=delay_cdf)
    # onset 0 -> avail 3 -> cdf 1.0 (plateau); onset 3 -> avail 0 -> cdf 0.0.
    np.testing.assert_allclose(vec, 0.5 * np.array([1.0, 0.7, 0.4, 0.0]))
    # Non-decreasing from recent (last) to old (first) onset.
    assert np.all(np.diff(vec[::-1]) >= 0)


def test_initial_unobserved_mu_is_finite_and_caps_only_initialization():
    initial_mu_vec = im._initial_unobserved_mu_vec(
        observed_incidence_vec=np.array([0, 2, 4]),
        reporting_prob_vec=np.array([0.0, 1e-9, 0.5]),
    )

    np.testing.assert_allclose(initial_mu_vec, np.array([1e-3, 200.0, 4.0]))
    assert np.all(np.isfinite(initial_mu_vec))


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


def test_get_t_rep_no_stop_full_vs_underreporting():
    # Under-reporting projects R_t a full serial interval past *all* data (the latent final
    # case can sit anywhere); full reporting only past the last *observed* case. Same inputs,
    # different horizon.
    incidence_vec = np.array(
        [1, 1, 0, 0, 0, 0, 0, 0, 0, 0]
    )  # t_data_stop=10, last case idx 1
    t_calc_vec = np.array([0, 5])
    assert (
        core._get_t_rep_no_stop(
            incidence_vec=incidence_vec,
            serial_interval_max=4,
            t_calc_vec=t_calc_vec,
            underreporting_fit=True,
        )
        == 14
    )
    assert (
        core._get_t_rep_no_stop(
            incidence_vec=incidence_vec,
            serial_interval_max=4,
            t_calc_vec=t_calc_vec,
            underreporting_fit=False,
        )
        == 10
    )
    # Never short of the latest calculation time.
    assert (
        core._get_t_rep_no_stop(
            incidence_vec=np.ones(3, dtype=int),
            serial_interval_max=2,
            t_calc_vec=np.array([0, 20]),
            underreporting_fit=True,
        )
        == 21
    )


def test_build_underreporting_model_structure():
    observed_incidence_vec = np.array([2, 1, 1, 0, 0])
    serial_interval_dist_vec = np.array([0.4, 0.3, 0.2, 0.1])
    model = core._build_underreporting_model(
        incidence_vec=observed_incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=build_ar_rep_no(),
        reporting_prob=0.6,
        delay_cdf=None,
        t_rep_no_stop=len(observed_incidence_vec) + len(serial_interval_dist_vec),
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
    assert len(time_coord) == len(observed_incidence_vec) + len(
        serial_interval_dist_vec
    )
    assert list(data_time_coord) == list(range(len(observed_incidence_vec)))
    assert list(unobserved_time_coord) == list(range(1, len(observed_incidence_vec)))
    assert model["unobserved"].name == "unobserved"
    assert model.named_vars_to_dims["incidence"] == ("data_time",)
    assert model["incidence"].type.shape == (len(observed_incidence_vec),)
    assert {"incidence", "obs", "unobserved"}.issubset(
        {v.name for v in model.basic_RVs + model.deterministics}
    )


def test_underreporting_model_masks_zero_reporting_probability_observation():
    observed_incidence_vec = np.array([1, 1, 0, 0])
    model = core._build_underreporting_model(
        incidence_vec=observed_incidence_vec,
        serial_interval_dist_vec=np.array([0.6, 0.4]),
        rep_no_vec_func=build_ar_rep_no(),
        reporting_prob=0.6,
        delay_cdf=np.array([0.0, 0.5, 1.0]),
        t_rep_no_stop=6,
    )

    obs_rv = model["obs"]
    observed_values = model.rvs_to_values[obs_rv].eval()
    # The most recent onset day has F(0)=0 and is omitted; earlier positive-probability
    # observations remain in the likelihood.
    np.testing.assert_array_equal(observed_values, np.array([1.0, 0.0]))


def test_underreporting_model_rejects_case_when_reporting_probability_is_zero():
    with pytest.raises(ValueError, match="effective reporting probability is zero"):
        core._build_underreporting_model(
            incidence_vec=np.array([1, 1, 0, 1]),
            serial_interval_dist_vec=np.array([0.6, 0.4]),
            rep_no_vec_func=build_ar_rep_no(),
            reporting_prob=0.6,
            delay_cdf=np.array([0.0, 0.5, 1.0]),
            t_rep_no_stop=6,
        )


def test_underreporting_model_all_days_masked_still_builds_finite_logp():
    # Corner of the delayed nowcast: a short snapshot whose only non-index day has F(0)=0, so
    # every reported-case day is masked out and `obs` carries no observations. The model must
    # still build and yield a finite logp (the latent renewal density alone), rather than fail
    # on an empty likelihood — future refactors could regress this via empty-array indexing.
    model = core._build_underreporting_model(
        incidence_vec=np.array([2, 0]),
        serial_interval_dist_vec=np.array([1.0]),
        rep_no_vec_func=build_ar_rep_no(),
        reporting_prob=0.6,
        delay_cdf=np.array(
            [0.0, 1.0]
        ),  # F(0) = 0 -> the single post-index day has pi = 0
        t_rep_no_stop=4,
    )
    obs_observed = model.rvs_to_values[model["obs"]].eval()
    assert obs_observed.shape == (0,)
    assert np.isfinite(model.compile_logp(sum=True)(model.initial_point()))


def test_underreporting_model_p1_collapses_latent_to_zero():
    # With reporting_prob=1 (no delay), no cases are unreported, so the latent U is
    # forced to zero (cases == observed). Checked via the model logp being maximal at U=0.
    observed_incidence_vec = np.array([2, 1, 1, 0, 0])
    serial_interval_dist_vec = np.array([0.4, 0.3, 0.2, 0.1])
    model = core._build_underreporting_model(
        incidence_vec=observed_incidence_vec,
        serial_interval_dist_vec=serial_interval_dist_vec,
        rep_no_vec_func=build_ar_rep_no(),
        reporting_prob=1.0,
        delay_cdf=None,
        t_rep_no_stop=len(observed_incidence_vec) + len(serial_interval_dist_vec),
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
    observed_incidence_vec = np.array([0, 3, 1, 0])
    serial_interval_dist_vec = np.array([0.6, 0.4])
    with pytest.raises(ValueError, match="starting with at least one index case"):
        core._build_underreporting_model(
            incidence_vec=observed_incidence_vec,
            serial_interval_dist_vec=serial_interval_dist_vec,
            rep_no_vec_func=build_ar_rep_no(),
            reporting_prob=0.6,
            delay_cdf=None,
            t_rep_no_stop=len(observed_incidence_vec) + len(serial_interval_dist_vec),
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
    observed_incidence_vec = np.array([2, 1, 1, 0, 0])
    t_rep_no_stop = 9
    captured: dict = {}

    def fake_build(**kwargs):
        captured["build_kwargs"] = kwargs
        return _CtxModel()

    def fake_build_full(**kwargs):
        raise AssertionError(
            "full-reporting model should not be built for the offshoot"
        )

    monkeypatch.setattr(core, "_build_underreporting_model", fake_build)
    monkeypatch.setattr(core, "_build_full_reporting_model", fake_build_full)

    def fake_sample(**kwargs):
        captured["sample_kwargs"] = kwargs
        incidence = np.tile(
            np.arange(len(observed_incidence_vec), dtype=float), (1, 3, 1)
        )
        posterior = xr.Dataset(
            {
                "rep_no": (
                    ("chain", "draw", "time"),
                    np.ones((1, 3, t_rep_no_stop)),
                ),
                "incidence": (("chain", "draw", "data_time"), incidence),
            },
            coords={
                "chain": [0],
                "draw": np.arange(3),
                "time": np.arange(t_rep_no_stop),
                "data_time": np.arange(len(observed_incidence_vec)),
            },
        )
        return _FakeTrace(posterior)

    monkeypatch.setattr(core.pm, "sample", fake_sample)

    def fake_prob(
        *, incidence, rep_no_func, serial_interval_dist_vec, t_calc, additional_dims
    ):
        captured["prob_incidence"] = incidence
        captured["additional_dims"] = additional_dims
        return np.full(np.atleast_1d(t_calc).size, 0.3)

    monkeypatch.setattr(core, "calc_additional_case_prob_analytical", fake_prob)

    out = inference._fit_model(
        incidence=observed_incidence_vec,
        serial_interval_dist_vec=np.array([0.4, 0.3, 0.2, 0.1]),
        rep_no_vec_func=lambda t: np.ones(t),
        quasi_real_time=False,
        reporting_prob=0.6,
    )

    assert "build_kwargs" in captured
    # No explicit step is attached; pm.sample assigns the latent's Metropolis step itself.
    assert "step" not in captured["sample_kwargs"]
    assert "nuts_sampler" not in captured["sample_kwargs"]
    assert captured["sample_kwargs"]["draws"] == 8000
    assert captured["sample_kwargs"]["tune"] == 2000
    # The probability uses only the unpadded latent-derived case array (data_time, chain, draw).
    assert np.issubdtype(captured["prob_incidence"].dtype, np.integer)
    assert captured["prob_incidence"].shape == (len(observed_incidence_vec), 1, 3)
    # Per-sample probabilities are requested so a credible interval can be formed.
    assert captured["additional_dims"] == "broadcast"
    # Risk and R_t span every day plus one projected day; inferred cases stop at the data boundary.
    assert {"incidence_mean", "incidence_lower", "incidence_upper"}.issubset(
        out.data_vars
    )
    assert {"additional_case_prob_lower", "additional_case_prob_upper"}.issubset(
        out.data_vars
    )
    np.testing.assert_array_equal(
        out.coords["time"].to_numpy(), np.arange(len(observed_incidence_vec) + 1)
    )
    np.testing.assert_array_equal(
        out.coords["data_time"].to_numpy(), np.arange(len(observed_incidence_vec))
    )
    assert out["rep_no"].dims == ("chain", "draw", "time")
    assert out["incidence"].dims == ("chain", "draw", "data_time")
    assert out["incidence_mean"].dims == ("data_time",)
    assert out["additional_case_prob"].dims == ("time",)


def test_underreporting_retro_csv_reindexes_incidence_to_projected_time(tmp_path):
    n_data_times = 3
    time = np.arange(n_data_times + 1)
    data_time = np.arange(n_data_times)
    posterior_ds = xr.Dataset(
        {
            "incidence_mean": ("data_time", [1.0, 2.0, 3.0]),
            "incidence_lower": ("data_time", [0.5, 1.5, 2.5]),
            "incidence_upper": ("data_time", [1.5, 2.5, 3.5]),
            "rep_no_mean": ("time", [0.9, 0.8, 0.7, 0.6]),
            "rep_no_lower": ("time", [0.7, 0.6, 0.5, 0.4]),
            "rep_no_upper": ("time", [1.1, 1.0, 0.9, 0.8]),
            "additional_case_prob": ("time", [1.0, 0.7, 0.3, 0.1]),
        },
        coords={"time": time, "data_time": data_time},
    )
    results_path = tmp_path / "underreporting.csv"

    _write_results(
        posterior_ds,
        results_path,
        np.array([1, 2, 0]),
        pd.Timestamp("2017-01-01"),
        suitability=False,
    )

    results_df = pd.read_csv(results_path)
    assert len(results_df) == n_data_times + 1
    assert results_df["reported_incidence"].iloc[:-1].notna().all()
    assert np.isnan(results_df["reported_incidence"].iloc[-1])
    assert (
        results_df[["incidence_mean", "incidence_lower", "incidence_upper"]]
        .iloc[:-1]
        .notna()
        .all()
        .all()
    )
    assert (
        results_df[["incidence_mean", "incidence_lower", "incidence_upper"]]
        .iloc[-1]
        .isna()
        .all()
    )
    assert results_df["additional_case_prob"].iloc[-1] == pytest.approx(0.1)


def test_underreporting_retro_date_plots_stop_at_end_of_2017(tmp_path):
    dates = pd.date_range("2017-12-30", periods=4)
    interval_columns = {
        f"{name}_{stat}": np.full(4, value)
        for name, value in [
            ("suitability", 0.5),
            ("rep_no_factor", 1.0),
            ("reproduction_number", 0.8),
        ]
        for stat in ("mean", "lower", "upper")
    }
    suitability_df = pd.DataFrame(
        {
            "date": dates,
            "reported_incidence": [1, 0, 0, 0],
            "incidence_mean": [1.0, 0.5, 0.25, 0.1],
            "incidence_lower": [1.0, 0.0, 0.0, 0.0],
            "incidence_upper": [1.0, 1.0, 1.0, 1.0],
            **interval_columns,
        }
    )
    autoregressive_df = suitability_df[
        [
            "date",
            "reported_incidence",
            "reproduction_number_mean",
            "reproduction_number_lower",
            "reproduction_number_upper",
        ]
    ]
    full_suitability_df = suitability_df[
        [
            "suitability_mean",
            "suitability_lower",
            "suitability_upper",
            "rep_no_factor_mean",
            "rep_no_factor_lower",
            "rep_no_factor_upper",
            "reproduction_number_mean",
            "reproduction_number_lower",
            "reproduction_number_upper",
        ]
    ].assign(day_of_outbreak=np.arange(4))
    full_autoregressive_df = autoregressive_df[
        [
            "reproduction_number_mean",
            "reproduction_number_lower",
            "reproduction_number_upper",
        ]
    ].assign(day_of_outbreak=np.arange(4))

    results_paths = {
        "suitability_p60": tmp_path / "suitability.csv",
        "autoregressive_p60": tmp_path / "autoregressive.csv",
    }
    full_reporting_paths = {
        "suitability": tmp_path / "full_suitability.csv",
        "autoregressive": tmp_path / "full_autoregressive.csv",
    }
    suitability_df.to_csv(results_paths["suitability_p60"], index=False)
    autoregressive_df.to_csv(results_paths["autoregressive_p60"], index=False)
    full_suitability_df.to_csv(full_reporting_paths["suitability"], index=False)
    full_autoregressive_df.to_csv(full_reporting_paths["autoregressive"], index=False)
    fig_paths = {
        name: tmp_path / f"{name}.svg"
        for name in ("incidence", "suitability", "rep_no_factor", "rep_no", "rep_no_ar")
    }
    inputs = {
        "results_paths": results_paths,
        "full_reporting_paths": full_reporting_paths,
        "full_reporting_outbreak_start_date": dates[0],
        "calendar_day_index_max": 365,
        "suitability_mean_vec": np.full(4, 0.5),
        "fig_paths": fig_paths,
    }

    incidence_fig, incidence_ax = _make_incidence_plot(inputs, ["tab:blue"])
    figure_ids_before_parameters = set(plt.get_fignums())
    _make_parameter_estimate_plots(inputs, ["tab:blue", "tab:orange"])
    parameter_figures = [
        plt.figure(fig_id)
        for fig_id in set(plt.get_fignums()) - figure_ids_before_parameters
    ]

    assert incidence_ax.get_xlim()[1] == 365
    assert np.max(incidence_ax.lines[0].get_xdata()) == 365
    assert len(parameter_figures) == 4
    assert all(fig.axes[0].get_xlim()[1] == 365 for fig in parameter_figures)
    plt.close(incidence_fig)
    for fig in parameter_figures:
        plt.close(fig)


@pytest.mark.parametrize("quasi_real_time", [False, True])
def test_fit_model_rejects_step_func_with_underreporting(quasi_real_time):
    with pytest.raises(
        ValueError, match="step_func is not supported with reporting_prob"
    ):
        inference._fit_model(
            incidence=np.array([1, 0]),
            serial_interval_dist_vec=np.array([1.0]),
            rep_no_vec_func=lambda t: np.ones(t),
            quasi_real_time=quasi_real_time,
            reporting_prob=0.6,
            step_func=lambda: object(),
        )


@pytest.mark.parametrize("quasi_real_time", [False, True])
def test_fit_model_rejects_delay_cdf_without_reporting_prob(quasi_real_time):
    with pytest.raises(ValueError, match="delay_cdf requires reporting_prob"):
        inference._fit_model(
            incidence=np.array([1, 0]),
            serial_interval_dist_vec=np.array([1.0]),
            rep_no_vec_func=lambda t: np.ones(t),
            quasi_real_time=quasi_real_time,
            delay_cdf=np.array([0.0, 1.0]),
        )


def test_fit_model_rejects_delayed_qrt_from_single_final_series():
    with pytest.raises(ValueError, match="sequence of historical incidence snapshots"):
        inference._fit_model(
            incidence=np.array([1, 0, 0]),
            serial_interval_dist_vec=np.array([1.0]),
            rep_no_vec_func=lambda t: np.ones(t),
            quasi_real_time=True,
            reporting_prob=0.6,
            delay_cdf=np.array([0.0, 1.0]),
        )


def test_fit_model_accepts_delayed_qrt_snapshot_sequence(monkeypatch):
    captured = {}

    def fake_fit_model_qrt(**kwargs):
        captured.update(kwargs)
        return xr.Dataset()

    monkeypatch.setattr(inference, "_fit_model_qrt", fake_fit_model_qrt)
    snapshots = [np.array([1, 0]), np.array([1, 0, 0])]
    delay_cdf = np.array([0.0, 1.0])

    inference._fit_model(
        incidence=snapshots,
        serial_interval_dist_vec=np.array([1.0]),
        rep_no_vec_func=lambda t: np.ones(t),
        quasi_real_time=True,
        reporting_prob=0.6,
        delay_cdf=delay_cdf,
    )

    assert captured["incidence"] is snapshots
    np.testing.assert_array_equal(captured["delay_cdf"], delay_cdf)
