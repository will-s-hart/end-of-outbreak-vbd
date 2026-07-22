# Note that AI tools were used to generate tests

import numpy as np
import pytest

from endoutbreakvbd.model import run_renewal_model, simulate_outbreak


def test_run_renewal_model_extinction_with_zero_reproduction_number(rng):
    out = run_renewal_model(
        rep_no_func=lambda t: 0.0,
        serial_interval_dist_vec=[1.0],
        rng=rng,
        t_stop=10,
        incidence_init=1,
    )
    np.testing.assert_array_equal(out, np.array([1, 0, 0]))
    assert out.dtype == int


def test_run_renewal_model_break_on_case_stops_on_first_new_case():
    out = run_renewal_model(
        rep_no_func=lambda t: 2.0,
        serial_interval_dist_vec=[1.0],
        rng=np.random.default_rng(0),
        t_stop=10,
        incidence_init=1,
        _break_on_case=True,
    )
    assert len(out) == 2
    assert out[0] == 1
    assert out[1] > 0


def test_run_renewal_model_accepts_vector_initial_incidence(rng):
    out = run_renewal_model(
        rep_no_func=lambda t: 0.0,
        serial_interval_dist_vec=[1.0],
        rng=rng,
        t_stop=10,
        incidence_init=[2, 1],
    )
    np.testing.assert_array_equal(out[:2], np.array([2, 1]))
    assert out.dtype == int
    assert len(out) <= 10


def test_run_renewal_model_default_initial_incidence_is_one(rng):
    out = run_renewal_model(
        rep_no_func=lambda t: 0.0,
        serial_interval_dist_vec=[1.0],
        rng=rng,
        t_stop=10,
    )
    assert out[0] == 1


def test_run_renewal_model_short_horizon_does_not_raise(rng):
    out = run_renewal_model(
        rep_no_func=lambda t: 1.0,
        serial_interval_dist_vec=[0.5, 0.5, 0.0],
        rng=rng,
        t_stop=1,
        incidence_init=1,
    )
    np.testing.assert_array_equal(out, np.array([1]))


def test_simulate_outbreak_returns_outbreak_meeting_size_target(rng):
    # R declines below 1 so the outbreak is finite (grows a few generations, then dies out).
    out = simulate_outbreak(
        rep_no_func=lambda t: 2.0 if t < 4 else 0.2,
        serial_interval_dist_vec=[1.0],
        rng=rng,
        min_size=5,
        incidence_init=1,
    )
    assert out.sum() >= 5


def test_simulate_outbreak_honours_max_size_and_accept_predicate(rng):
    out = simulate_outbreak(
        rep_no_func=lambda t: 1.8 if t < 4 else 0.2,
        serial_interval_dist_vec=[1.0],
        rng=rng,
        min_size=3,
        max_size=50,
        accept=lambda incidence_vec: incidence_vec[0] == 1,
        incidence_init=1,
    )
    assert 3 <= out.sum() <= 50
    assert out[0] == 1


def test_simulate_outbreak_raises_when_target_unreachable(rng):
    # An always-extinct process (R = 0) can never reach a large minimum size.
    with pytest.raises(RuntimeError, match="could not simulate"):
        simulate_outbreak(
            rep_no_func=lambda t: 0.0,
            serial_interval_dist_vec=[1.0],
            rng=rng,
            min_size=1000,
            max_attempts=5,
        )
