import numpy as np
import pytest

import endoutbreakvbd.further_case_risk as fcr


def test_calc_further_case_risk_analytical_t_calc_zero_is_one():
    risk = fcr.calc_further_case_risk_analytical(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.0,
        gen_time_dist_vec=[1.0],
        t_calc=0,
    )
    assert risk == 1


def test_calc_further_case_risk_analytical_after_t_max_is_zero():
    risk = fcr.calc_further_case_risk_analytical(
        incidence_vec=[1],
        rep_no_func=lambda t: 1.0,
        gen_time_dist_vec=[1.0],
        t_calc=2,
    )
    assert risk == 0


def test_calc_further_case_risk_analytical_vectorized_t_calc_shape():
    risk = fcr.calc_further_case_risk_analytical(
        incidence_vec=[1],
        rep_no_func=lambda t: np.zeros_like(t, dtype=float),
        gen_time_dist_vec=[1.0],
        t_calc=np.array([0, 1, 2]),
    )
    assert risk.shape == (3,)
    np.testing.assert_allclose(risk, np.array([1.0, 0.0, 0.0]))


def test_calc_further_case_risk_analytical_scalar_closed_form_case():
    risk = fcr.calc_further_case_risk_analytical(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.5,
        gen_time_dist_vec=[1.0],
        t_calc=1,
    )
    assert risk == pytest.approx(1 - np.exp(-0.5))


def test_calc_further_case_risk_analytical_handles_posterior_matrix_rep_no():
    def rep_no_func(t):
        t_arr = np.atleast_1d(t)
        return np.column_stack(
            [np.full(t_arr.size, 0.2, dtype=float), np.full(t_arr.size, 0.6, dtype=float)]
        )

    risk = fcr.calc_further_case_risk_analytical(
        incidence_vec=[1],
        rep_no_func=rep_no_func,
        gen_time_dist_vec=[1.0],
        t_calc=1,
    )
    expected = 1 - np.mean(np.exp(-np.array([0.2, 0.6])))
    assert risk == pytest.approx(expected)


def test_calc_further_case_risk_analytical_all_zero_incidence_returns_zero():
    risk = fcr.calc_further_case_risk_analytical(
        incidence_vec=[0, 0, 0],
        rep_no_func=lambda t: 1.0,
        gen_time_dist_vec=[1.0],
        t_calc=1,
    )
    assert risk == 0


def test_calc_further_case_risk_simulation_parallel_false_deterministic_zero_risk(rng):
    risk = fcr.calc_further_case_risk_simulation(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.0 if np.isscalar(t) else np.zeros_like(t, dtype=float),
        gen_time_dist_vec=[1.0],
        t_calc=np.array([1, 2]),
        n_sims=20,
        rng=rng,
        parallel=False,
    )
    assert risk.shape == (2,)
    np.testing.assert_allclose(risk, np.array([0.0, 0.0]))


def test_further_cases_one_sim_uses_run_renewal_model(monkeypatch):
    called = {}

    def fake_run_renewal_model(**kwargs):
        called.update(kwargs)
        return np.array([1, 0, 1])

    monkeypatch.setattr(fcr, "run_renewal_model", fake_run_renewal_model)

    out = fcr._further_cases_one_sim(
        (
            np.array([1]),
            lambda t: 1.0,
            np.array([1.0, 0.0]),
            1,
            np.random.default_rng(0),
            0,
            0,
        )
    )

    assert called["t_stop"] == 3
    assert called["_break_on_case"] is True
    assert out == (0, 0, True)


def test_further_cases_one_sim_raises_when_t_calc_does_not_match_incidence_length():
    with pytest.raises(ValueError, match="does not match length of incidence_vec"):
        fcr._further_cases_one_sim(
            (
                np.array([1, 0]),
                lambda t: 1.0,
                np.array([1.0, 0.0]),
                1,
                np.random.default_rng(0),
                0,
                0,
            )
        )


def test_calc_declaration_delay_scalar_threshold():
    delay = fcr.calc_declaration_delay(
        risk_vec=np.array([0.2, 0.04, 0.01]),
        perc_risk_threshold=5,
        delay_of_first_risk=1,
    )
    assert delay == 2


def test_calc_declaration_delay_vector_thresholds():
    delay = fcr.calc_declaration_delay(
        risk_vec=np.array([0.2, 0.04, 0.01]),
        perc_risk_threshold=np.array([5, 2]),
        delay_of_first_risk=1,
    )
    np.testing.assert_array_equal(delay, np.array([2, 3]))


def test_calc_declaration_delay_raises_when_risk_never_below_threshold():
    with pytest.raises(ValueError, match="does not drop below"):
        fcr.calc_declaration_delay(
            risk_vec=np.array([0.2, 0.15, 0.1]),
            perc_risk_threshold=5,
            delay_of_first_risk=1,
        )


def test_simulation_risk_t_calc_zero_should_not_crash(rng):
    risk = fcr.calc_further_case_risk_simulation(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.0 if np.isscalar(t) else np.zeros_like(t, dtype=float),
        gen_time_dist_vec=[1.0],
        t_calc=np.array([0]),
        n_sims=5,
        rng=rng,
        parallel=False,
    )
    np.testing.assert_allclose(risk, np.array([1.0]))
