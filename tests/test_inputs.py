# Note that AI tools were used to generate tests

import numpy as np
import pandas as pd
import pytest

import scripts.inputs as inputs


def test_get_serial_interval_dist_properties():
    serial_interval_dist = inputs._get_serial_interval_dist()
    assert len(serial_interval_dist) == 40
    assert np.all(serial_interval_dist >= 0)
    assert np.isclose(np.sum(serial_interval_dist), 1.0, atol=1e-8)


def test_get_lazio_outbreak_data_has_expected_columns_and_doy():
    df = inputs._get_lazio_outbreak_data()
    assert "cases" in df.columns
    assert "doy" in df.columns
    assert df.index.name == "onset_date"
    assert df["doy"].between(1, 366).all()


def test_get_2017_suitability_data_has_expected_columns_and_doy():
    df = inputs._get_2017_suitability_data()
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
        "df_suitability_grid",
        "suitability_lag_days",
    }
    assert set(out["results_paths"].keys()) == {"all", "2017"}
    assert set(out["fig_paths"].keys()) == {
        "temperature",
        "suitability_model",
        "suitability",
    }
    assert {"temperature", "suitability"}.issubset(out["df_suitability_grid"].columns)


def test_get_inputs_sim_study_structure_and_callables(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)
    out = inputs.get_inputs_sim_study()

    required_keys = {
        "serial_interval_dist_vec",
        "rep_no_factor",
        "rep_no_func_doy",
        "rep_no_from_doy_start",
        "example_outbreak_doy_start_vals",
        "example_outbreak_incidence_vec",
        "example_outbreak_perc_risk_threshold_vals",
        "many_outbreak_n_sims",
        "many_outbreak_outbreak_size_threshold",
        "many_outbreak_perc_risk_threshold_vals",
        "results_paths",
        "fig_paths",
    }
    assert required_keys.issubset(out.keys())

    rep_no_vals = out["rep_no_func_doy"](np.array([0, 10]))
    assert rep_no_vals.shape == (2,)
    rep_no_vals_shift = out["rep_no_from_doy_start"](np.array([0, 10]), doy_start=100)
    assert rep_no_vals_shift.shape == (2,)


def test_get_inputs_inference_test_standard_and_quasi_real_time(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)

    out_std = inputs.get_inputs_inference_test(quasi_real_time=False)
    out_qrt = inputs.get_inputs_inference_test(quasi_real_time=True)

    assert "inference_test" in str(out_std["results_paths"]["outbreak_data"])
    assert "inference_test_qrt" in str(out_qrt["results_paths"]["outbreak_data"])
    assert out_std["doy_start"] == 152
    assert len(out_std["suitability_mean_grid"]) == 365


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
        assert {"doy", "outbreak_day", "days_from_final_case"}.issubset(val.keys())
        assert isinstance(val["days_from_final_case"], (int, np.integer))


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
        "decision",
    }


def test_get_lazio_reporting_matrix_is_daily_forward_filled():
    df = inputs._get_lazio_reporting_matrix()
    assert df.index.name == "onset_date"
    assert isinstance(df.index, pd.DatetimeIndex)
    assert isinstance(df.columns, pd.DatetimeIndex)
    # Columns are a contiguous daily report grid; no NA (onset-after-snapshot filled to 0).
    assert (df.columns.to_series().diff().dropna().dt.days == 1).all()
    assert df.isna().sum().sum() == 0
    # Cumulative-by-report-date: each onset row is non-decreasing across report columns.
    assert (df.to_numpy()[:, 1:] >= df.to_numpy()[:, :-1]).all()


def test_fit_reporting_delay_matches_expected_estimates():
    df = inputs._get_lazio_reporting_matrix()
    delay = inputs._fit_reporting_delay(df)
    assert set(delay) == {
        "support",
        "cdf",
        "pmf_fitted",
        "pmf_empirical",
        "shape",
        "scale",
        "n",
        "mean",
    }
    # Only fully observed onset rows (onset >= first report date) contribute.
    assert delay["n"] == 116
    assert delay["mean"] == pytest.approx(13.17, abs=0.01)
    assert delay["shape"] == pytest.approx(1.24, abs=0.02)
    assert delay["scale"] == pytest.approx(10.77, abs=0.05)
    # CDF is a non-decreasing distribution over the delay support, plateauing at 1.
    assert len(delay["support"]) == len(delay["cdf"])
    assert np.all(np.diff(delay["cdf"]) >= 0)
    assert np.all(delay["cdf"] <= 1.0)
    assert delay["cdf"][-1] == pytest.approx(1.0)
    assert delay["pmf_empirical"].sum() == pytest.approx(1.0)
    assert delay["pmf_fitted"].sum() == pytest.approx(1.0)


def test_get_inputs_lazio_underreporting_qrt_structure(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)

    out = inputs.get_inputs_lazio_underreporting_qrt(
        start_date="2017-11-01", end_date="2017-11-15", stride=7
    )

    # One right-truncated incidence series per calculation time.
    assert len(out["incidence_vecs"]) == len(out["calc_times"])
    assert len(out["decision_dates"]) == len(out["calc_times"])
    # Each snapshot series spans onset days 0..(t_calc - 1) inclusive (the last known onset
    # day; t_calc is the next-day decision), so its length equals t_calc.
    for incidence_vec, calc_time in zip(out["incidence_vecs"], out["calc_times"]):
        assert len(incidence_vec) == calc_time
        assert np.issubdtype(incidence_vec.dtype, np.integer)
    assert np.array_equal(out["latest_incidence_vec"], out["incidence_vecs"][-1])
    # Suitability prior is sized to the latest fit's inference horizon: the latest snapshot day
    # (t_calc - 1) plus one, extended a serial interval to project R_t forward.
    serial_interval_max = len(out["serial_interval_dist_vec"])
    assert (
        len(out["suitability_mean_vec"])
        == int(out["calc_times"].max()) + serial_interval_max
    )
    assert out["reporting_prob"] == 0.6
    # The nowcast reports a single (60%) reporting ceiling; the multi-ceiling sweep moved to the
    # retrospective analysis.
    assert out["suitability_sweep"] == (("suitability_p60", 0.6),)
    assert {"suitability_p60", "autoregressive_p60", "trajectory", "delay"}.issubset(
        out["results_paths"]
    )
    # The full-reporting lazio_outbreak fits back the dashed "full outbreak knowledge" overlay.
    assert {"suitability", "autoregressive"}.issubset(out["full_reporting_paths"])
    assert "perc_risk_threshold_grid" in out


def test_get_inputs_lazio_underreporting_qrt_rejects_out_of_range_dates(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)
    with pytest.raises(ValueError, match="precedes the first report date"):
        inputs.get_inputs_lazio_underreporting_qrt(start_date="2017-01-01")
    with pytest.raises(ValueError, match="beyond the last report date"):
        inputs.get_inputs_lazio_underreporting_qrt(end_date="2018-06-01")


def test_get_inputs_lazio_underreporting_retro_structure(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)

    out = inputs.get_inputs_lazio_underreporting_retro()

    serial_interval_max = len(out["serial_interval_dist_vec"])
    # The additional-case probability is reported for every (padded) day plus one projected day
    # past the data (the current-day risk); the incidence is padded with a serial interval (+1) of
    # zero-report days.
    assert len(out["calc_times"]) == len(out["incidence_vec"]) + 1
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
    assert "perc_risk_threshold_grid" in out


def test_get_inputs_sim_underreporting_structure(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)
    out = inputs.get_inputs_sim_underreporting()
    assert {
        "serial_interval_dist_vec",
        "seed",
        "reporting_prob",
        "min_outbreak_size",
        "incidence_init",
        "perc_risk_threshold_grid",
        "results_paths",
        "fig_paths",
    } == set(out)
    assert 0 < out["reporting_prob"] <= 1
    assert set(out["fig_paths"]) == {
        "cases",
        "rep_no",
        "additional_case_prob",
        "decision",
    }


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
        "decision",
    }
