import numpy as np
import scipy.stats

import endoutbreakvbd.inputs as inputs


def test_discretise_cori_probabilities_are_nonnegative_and_sum_to_one():
    dist = scipy.stats.gamma(a=2.0, scale=1.0)
    p_vec = inputs._discretise_cori(dist, max_val=12, allow_zero=True)
    assert np.all(p_vec >= 0)
    assert np.isclose(np.sum(p_vec), 1.0, atol=1e-8)


def test_discretise_cori_allow_zero_changes_output_length():
    dist = scipy.stats.gamma(a=2.0, scale=1.0)
    p_with_zero = inputs._discretise_cori(dist, max_val=10, allow_zero=True)
    p_without_zero = inputs._discretise_cori(dist, max_val=10, allow_zero=False)
    assert len(p_with_zero) == 11
    assert len(p_without_zero) == 10


def test_get_gen_time_dist_properties():
    gen_time_dist = inputs._get_gen_time_dist()
    assert len(gen_time_dist) == 40
    assert np.all(gen_time_dist >= 0)
    assert np.isclose(np.sum(gen_time_dist), 1.0, atol=1e-8)


def test_get_lazio_outbreak_data_has_expected_columns_and_doy():
    df = inputs._get_lazio_outbreak_data()
    assert "cases" in df.columns
    assert "doy" in df.columns
    assert df.index.name == "onset_date"
    assert df["doy"].between(1, 366).all()


def test_get_2017_suitability_data_has_expected_columns_and_doy():
    df = inputs._get_2017_suitability_data()
    assert "suitability_smoothed" in df.columns
    assert "doy" in df.columns
    assert df.index.name == "date"
    assert df["doy"].between(1, 366).all()


def test_get_inputs_weather_suitability_data_structure(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)
    out = inputs.get_inputs_weather_suitability_data()
    assert set(out.keys()) == {"results_paths", "fig_paths"}
    assert set(out["results_paths"].keys()) == {"all", "2017"}
    assert set(out["fig_paths"].keys()) == {"temperature", "suitability"}


def test_get_inputs_sim_study_structure_and_callables(monkeypatch):
    monkeypatch.setattr(inputs.pathlib.Path, "mkdir", lambda *args, **kwargs: None)
    out = inputs.get_inputs_sim_study()

    required_keys = {
        "gen_time_dist_vec",
        "rep_no_factor",
        "rep_no_func_doy",
        "rep_no_from_doy_start",
        "example_outbreak_doy_start_vals",
        "example_outbreak_incidence_vec",
        "example_outbreak_perc_risk_threshold_vals",
        "many_outbreak_n_sims",
        "many_outbreak_outbreak_size_threshold",
        "many_outbreak_perc_risk_threshold",
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

    assert len(out["doy_vec"]) == len(out["incidence_vec"])
    assert len(out["suitability_mean_vec"]) == len(out["incidence_vec"])

    declaration_keys = {"blood_resumed_rome", "blood_resumed_anzio", "45_day_rule"}
    assert set(out["existing_declarations"].keys()) == declaration_keys

    for val in out["existing_declarations"].values():
        assert {"doy", "outbreak_day", "days_from_last_case"}.issubset(val.keys())
        assert isinstance(val["days_from_last_case"], (int, np.integer))
