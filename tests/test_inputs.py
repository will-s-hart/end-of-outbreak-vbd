# Note that AI tools were used to generate tests

import numpy as np
import pandas as pd

import scripts.inference_test as inference_test
import scripts.inputs as inputs


def test_get_serial_interval_dist_vec_properties():
    serial_interval_dist_vec = inputs._get_serial_interval_dist_vec()
    assert len(serial_interval_dist_vec) == 40
    assert np.all(serial_interval_dist_vec >= 0)
    assert np.isclose(np.sum(serial_interval_dist_vec), 1.0, atol=1e-8)


def test_get_lazio_outbreak_df_has_expected_columns_and_doy():
    df = inputs._get_lazio_outbreak_df()
    assert "cases" in df.columns
    assert "doy" in df.columns
    assert df.index.name == "onset_date"
    assert df["doy"].between(1, 366).all()


def test_get_2017_suitability_df_has_expected_columns_and_doy():
    df = inputs._get_2017_suitability_df()
    assert "suitability_smoothed_lagged" in df.columns
    assert "doy" in df.columns
    assert df.index.name == "date"
    assert df["doy"].between(1, 366).all()


def test_get_inputs_weather_suitability_data_structure(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)
    out = inputs.get_inputs_weather_suitability_data()
    assert set(out.keys()) == {
        "results_paths",
        "fig_paths",
        "suitability_grid_df",
        "suitability_lag_days",
    }
    assert set(out["results_paths"].keys()) == {"all", "2017"}
    assert set(out["fig_paths"].keys()) == {
        "temperature",
        "suitability_model",
        "suitability",
    }
    assert {"temperature", "suitability"}.issubset(out["suitability_grid_df"].columns)


def test_get_inputs_sim_study_structure_and_callables(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)
    out = inputs.get_inputs_sim_study()

    required_keys = {
        "serial_interval_dist_vec",
        "rep_no_factor",
        "rep_no_doy_func",
        "rep_no_from_doy_start",
        "example_outbreak_doy_start_vals",
        "example_outbreak_incidence_vec",
        "example_outbreak_risk_threshold_pct_vals",
        "many_outbreak_n_sims",
        "many_outbreak_min_size",
        "many_outbreak_risk_threshold_pct_vals",
        "results_paths",
        "fig_paths",
    }
    assert required_keys.issubset(out.keys())

    rep_no_vec = out["rep_no_doy_func"](np.array([0, 10]))
    assert rep_no_vec.shape == (2,)
    shifted_rep_no_vec = out["rep_no_from_doy_start"](np.array([0, 10]), doy_start=100)
    assert shifted_rep_no_vec.shape == (2,)


def test_get_inputs_inference_test_standard_and_quasi_real_time(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)

    out_std = inputs.get_inputs_inference_test(quasi_real_time=False)
    out_qrt = inputs.get_inputs_inference_test(quasi_real_time=True)

    assert "inference_test" in str(out_std["results_paths"]["outbreak_data"])
    assert "inference_test_qrt" in str(out_qrt["results_paths"]["outbreak_data"])
    assert out_std["doy_start"] == 152
    assert len(out_std["suitability_mean_grid"]) == 365


def test_generate_inference_test_data_includes_projected_day(monkeypatch, tmp_path):
    incidence_vec = np.array([500, 0])
    monkeypatch.setattr(
        inference_test, "run_renewal_model", lambda **kwargs: incidence_vec
    )
    monkeypatch.setattr(
        inference_test,
        "_simulate_ar_vec",
        lambda *, mean_vec, **kwargs: np.asarray(mean_vec),
    )
    captured = {}

    def fake_prob(**kwargs):
        captured["t_calc"] = kwargs["t_calc"]
        return np.linspace(1.0, 0.0, len(kwargs["t_calc"]))

    monkeypatch.setattr(
        inference_test, "calc_additional_case_prob_analytical", fake_prob
    )

    outbreak_df = inference_test._generate_outbreak_data(
        serial_interval_dist_vec=np.array([1.0]),
        doy_start=100,
        suitability_mean_grid=np.full(365, 0.5),
        suitability_model_params={"suitability_std": 0.0, "suitability_rho": 0.0},
        rng=np.random.default_rng(0),
        results_path=tmp_path / "outbreak.csv",
    )

    assert len(outbreak_df) == len(incidence_vec) + 1
    np.testing.assert_array_equal(outbreak_df["incidence"].iloc[:-1], incidence_vec)
    assert np.isnan(outbreak_df["incidence"].iloc[-1])
    assert outbreak_df.drop(columns="incidence").iloc[-1].notna().all()
    np.testing.assert_array_equal(captured["t_calc"], np.arange(len(outbreak_df)))


def test_inference_test_passes_unobserved_projection_only_to_suitability_prior(
    monkeypatch, tmp_path
):
    outbreak_df = pd.DataFrame(
        {
            "incidence": [2.0, 0.0, np.nan],
            "suitability_mean": [0.4, 0.3, 0.2],
        }
    )
    results_paths = {
        "outbreak_data": tmp_path / "outbreak.csv",
        "autoregressive": tmp_path / "autoregressive.csv",
        "autoregressive_diagnostics": tmp_path / "autoregressive_diagnostics.csv",
        "suitability": tmp_path / "suitability.csv",
        "suitability_diagnostics": tmp_path / "suitability_diagnostics.csv",
    }
    monkeypatch.setattr(
        inference_test,
        "get_inputs_inference_test",
        lambda **kwargs: {
            "serial_interval_dist_vec": np.array([1.0]),
            "doy_start": 100,
            "suitability_mean_grid": np.full(365, 0.5),
            "suitability_model_params": {},
            "results_paths": results_paths,
        },
    )
    monkeypatch.setattr(
        inference_test, "_generate_outbreak_data", lambda **kwargs: outbreak_df
    )
    calls = []
    monkeypatch.setattr(
        inference_test, "_run_analyses_for_model", lambda **kwargs: calls.append(kwargs)
    )

    inference_test.run_analyses()

    assert len(calls) == 2
    for call in calls:
        np.testing.assert_array_equal(call["incidence_vec"], np.array([2, 0]))
    suitability_call = next(call for call in calls if call["model"] == "suitability")
    np.testing.assert_array_equal(
        suitability_call["fit_model_kwargs"]["suitability_mean_vec"],
        np.array([0.4, 0.3, 0.2]),
    )


def test_get_inputs_lazio_outbreak_consistency(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)

    out = inputs.get_inputs_lazio_outbreak(quasi_real_time=False)

    # The fit reports one projected day past the data, so the day axis and suitability prior run
    # one day longer than the incidence.
    assert len(out["doy_vec"]) == len(out["incidence_vec"]) + 1
    assert len(out["suitability_mean_vec"]) == len(out["incidence_vec"]) + 1

    decision_keys = {"blood_resumed_rome", "blood_resumed_anzio", "45_day_rule"}
    assert set(out["existing_decisions"].keys()) == decision_keys

    for val in out["existing_decisions"].values():
        assert {"doy", "t", "days_after_final_case"}.issubset(val.keys())
        assert isinstance(val["days_after_final_case"], (int, np.integer))


def test_get_inputs_lazio_frozen_structure(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)

    out = inputs.get_inputs_lazio_frozen()

    assert len(out["doy_vec"]) == len(out["incidence_vec"]) + 1
    assert set(out["existing_decisions"].keys()) == {
        "blood_resumed_rome",
        "blood_resumed_anzio",
        "45_day_rule",
    }
    # Suitability results are reused from the lazio_outbreak analysis.
    assert "lazio_outbreak" in str(out["results_paths"]["suitability"])
    assert set(out["results_paths"].keys()) == {"suitability", "autoregressive_frozen"}
    assert "lazio_frozen" in str(out["results_paths"]["autoregressive_frozen"])
    assert set(out["fig_paths"].keys()) == {
        "rep_no",
        "additional_case_prob",
        "decision_delay",
    }


def test_get_inputs_lazio_underreporting_retro_structure(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)

    out = inputs.get_inputs_lazio_underreporting_retro()

    serial_interval_max = len(out["serial_interval_dist_vec"])
    # The additional-case probability is reported for every (padded) day plus one projected day
    # past the data (the current-day risk); the incidence is padded with a serial interval (+1) of
    # zero-report days.
    assert len(out["t_calc_vec"]) == len(out["incidence_vec"]) + 1
    assert out["incidence_vec"][-1] == 0
    # Suitability prior extends a further serial interval beyond the padded data (the under-
    # reporting projection horizon).
    assert (
        len(out["suitability_mean_vec"])
        == len(out["incidence_vec"]) + serial_interval_max
    )
    assert out["reporting_prob"] == 0.6
    assert {"suitability_p60", "autoregressive_p60"}.issubset(out["results_paths"])
    assert "trajectory" not in out["results_paths"]
    assert "rep_no_ar" in out["fig_paths"]
    assert {"suitability", "autoregressive"}.issubset(out["full_reporting_paths"])
    assert "risk_threshold_pct_grid" in out


def test_get_inputs_lazio_epiestim_structure(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)

    out = inputs.get_inputs_lazio_epiestim()

    assert len(out["doy_vec"]) == len(out["incidence_vec"]) + 1
    assert set(out["existing_decisions"].keys()) == {
        "blood_resumed_rome",
        "blood_resumed_anzio",
        "45_day_rule",
    }
    # Suitability results are reused from the lazio_outbreak analysis.
    assert "lazio_outbreak" in str(out["results_paths"]["suitability"])
    assert set(out["results_paths"].keys()) == {"suitability", "epiestim"}
    assert "lazio_epiestim" in str(out["results_paths"]["epiestim"])
    assert set(out["fig_paths"].keys()) == {
        "rep_no",
        "additional_case_prob",
        "decision_delay",
    }
