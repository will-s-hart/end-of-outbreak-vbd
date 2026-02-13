import numpy as np
import pytest

from endoutbreakvbd.model import run_renewal_model


def test_run_renewal_model_extinction_with_zero_reproduction_number(rng):
    out = run_renewal_model(
        rep_no_func=lambda t: 0.0,
        gen_time_dist_vec=[1.0],
        rng=rng,
        t_stop=10,
        incidence_init=1,
    )
    np.testing.assert_array_equal(out, np.array([1, 0, 0]))
    assert out.dtype == int


def test_run_renewal_model_break_on_case_stops_on_first_new_case():
    out = run_renewal_model(
        rep_no_func=lambda t: 2.0,
        gen_time_dist_vec=[1.0],
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
        gen_time_dist_vec=[1.0],
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
        gen_time_dist_vec=[1.0],
        rng=rng,
        t_stop=10,
    )
    assert out[0] == 1


def test_run_renewal_model_raises_for_short_horizon_relative_to_generation_time(rng):
    with pytest.raises(ValueError, match="negative dimensions"):
        run_renewal_model(
            rep_no_func=lambda t: 1.0,
            gen_time_dist_vec=[0.5, 0.5, 0.0],
            rng=rng,
            t_stop=1,
            incidence_init=1,
        )
