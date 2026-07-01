# Note that AI tools were used to generate tests

from typing import Any, cast

import numpy as np
import pytest

import endoutbreakvbd.additional_case_prob as acp


def test_calc_additional_case_prob_analytical_t_calc_zero_is_one():
    prob = acp.calc_additional_case_prob_analytical(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.0,
        serial_interval_dist_vec=[1.0],
        t_calc=0,
    )
    assert prob == 1


def test_calc_additional_case_prob_analytical_after_t_max_is_zero():
    prob = acp.calc_additional_case_prob_analytical(
        incidence_vec=[1],
        rep_no_func=lambda t: 1.0,
        serial_interval_dist_vec=[1.0],
        t_calc=2,
    )
    assert prob == 0


def test_calc_additional_case_prob_analytical_vectorized_t_calc_shape():
    prob = acp.calc_additional_case_prob_analytical(
        incidence_vec=[1],
        rep_no_func=lambda t: np.zeros_like(t, dtype=float),
        serial_interval_dist_vec=[1.0],
        t_calc=np.array([0, 1, 2], dtype=int),
    )
    assert isinstance(prob, np.ndarray)
    assert prob.shape == (3,)
    np.testing.assert_allclose(prob, np.array([1.0, 0.0, 0.0]))


def test_calc_additional_case_prob_analytical_scalar_closed_form_case():
    prob = acp.calc_additional_case_prob_analytical(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.5,
        serial_interval_dist_vec=[1.0],
        t_calc=1,
    )
    assert prob == pytest.approx(1 - np.exp(-0.5))


def test_calc_additional_case_prob_analytical_handles_posterior_matrix_rep_no():
    def rep_no_func(t):
        t_arr = np.atleast_1d(t)
        return np.column_stack(
            [
                np.full(t_arr.size, 0.2, dtype=float),
                np.full(t_arr.size, 0.6, dtype=float),
            ]
        )

    prob = acp.calc_additional_case_prob_analytical(
        incidence_vec=[1],
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=[1.0],
        t_calc=1,
    )
    expected = 1 - np.mean(np.exp(-np.array([0.2, 0.6])))
    assert prob == pytest.approx(expected)


def test_calc_additional_case_prob_analytical_matrix_incidence_equals_per_sample_mean():
    # A (time, sample) incidence matrix with a scalar rep_no must equal the mean over the
    # per-sample 1-D probabilities.
    rng = np.random.default_rng(0)
    incidence_mat = rng.integers(0, 3, size=(5, 4))
    incidence_mat[0] = 1  # keep an index case in every sample
    serial_interval_dist_vec = [0.5, 0.3, 0.2]

    def rep_no_func(t):
        return 0.5

    for t_calc in (1, 2, 3):
        prob_mat = acp.calc_additional_case_prob_analytical(
            incidence_vec=incidence_mat,
            rep_no_func=rep_no_func,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=t_calc,
        )
        per_sample = [
            acp.calc_additional_case_prob_analytical(
                incidence_vec=incidence_mat[:, s],
                rep_no_func=rep_no_func,
                serial_interval_dist_vec=serial_interval_dist_vec,
                t_calc=t_calc,
            )
            for s in range(incidence_mat.shape[1])
        ]
        assert prob_mat == pytest.approx(float(np.mean(per_sample)))


def test_calc_additional_case_prob_analytical_matrix_incidence_and_rep_align():
    # When both incidence and rep_no carry a sample dimension, draw s of the cases must be
    # paired with draw s of the reproduction number (aligned, not an outer product).
    rng = np.random.default_rng(1)
    incidence_mat = rng.integers(0, 3, size=(5, 4))
    incidence_mat[0] = 1
    rep_no_mat = rng.uniform(0.1, 0.9, size=(20, 4))
    serial_interval_dist_vec = [0.5, 0.3, 0.2]

    def rep_no_func_mat(t):
        return rep_no_mat[np.asarray(t)]

    for t_calc in (1, 2, 3):
        prob_mat = acp.calc_additional_case_prob_analytical(
            incidence_vec=incidence_mat,
            rep_no_func=rep_no_func_mat,
            serial_interval_dist_vec=serial_interval_dist_vec,
            t_calc=t_calc,
        )
        per_sample = [
            acp.calc_additional_case_prob_analytical(
                incidence_vec=incidence_mat[:, s],
                rep_no_func=lambda t, s=s: rep_no_mat[np.asarray(t), s],
                serial_interval_dist_vec=serial_interval_dist_vec,
                t_calc=t_calc,
            )
            for s in range(incidence_mat.shape[1])
        ]
        assert prob_mat == pytest.approx(float(np.mean(per_sample)))


def test_calc_additional_case_prob_analytical_singleton_sample_matches_1d():
    # A (time, 1) incidence column must give the same probability as the 1-D series.
    incidence_1d = np.array([1, 2, 0, 1, 0])
    serial_interval_dist_vec = [0.5, 0.3, 0.2]
    prob_1d = acp.calc_additional_case_prob_analytical(
        incidence_vec=incidence_1d,
        rep_no_func=lambda t: 0.4,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=2,
    )
    prob_col = acp.calc_additional_case_prob_analytical(
        incidence_vec=incidence_1d.reshape(-1, 1),
        rep_no_func=lambda t: 0.4,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=2,
    )
    assert prob_col == pytest.approx(prob_1d)


def test_calc_additional_case_prob_analytical_broadcast_keeps_samples():
    # additional_dims="broadcast" returns per-sample probabilities; averaging them must
    # reproduce the default "average" result.
    rng = np.random.default_rng(2)
    incidence_mat = rng.integers(0, 3, size=(5, 4))
    incidence_mat[0] = 1
    rep_no_mat = rng.uniform(0.1, 0.9, size=(20, 4))
    serial_interval_dist_vec = [0.5, 0.3, 0.2]

    def rep_no_func(t):
        return rep_no_mat[np.asarray(t)]

    t_calc = np.array([1, 2, 3], dtype=int)
    per_sample = acp.calc_additional_case_prob_analytical(
        incidence_vec=incidence_mat,
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=t_calc,
        additional_dims="broadcast",
    )
    averaged = acp.calc_additional_case_prob_analytical(
        incidence_vec=incidence_mat,
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=serial_interval_dist_vec,
        t_calc=t_calc,
    )
    assert per_sample.shape == (3, 4)
    np.testing.assert_allclose(per_sample.mean(axis=1), averaged)


def test_calc_additional_case_prob_analytical_broadcast_broadcasts_early_returns():
    # An early-return t_calc (t_calc=0 -> prob 1) is broadcast to the sample shape so it
    # stacks with the per-sample results.
    incidence_mat = np.ones((4, 3), dtype=int)

    def rep_no_func(t):
        t_arr = np.atleast_1d(t)
        return np.full((t_arr.size, 3), 0.5)

    per_sample = acp.calc_additional_case_prob_analytical(
        incidence_vec=incidence_mat,
        rep_no_func=rep_no_func,
        serial_interval_dist_vec=[1.0],
        t_calc=np.array([0, 1], dtype=int),
        additional_dims="broadcast",
    )
    assert per_sample.shape == (2, 3)
    np.testing.assert_allclose(per_sample[0], 1.0)


def test_calc_additional_case_prob_analytical_invalid_additional_dims():
    with pytest.raises(ValueError, match="additional_dims"):
        acp.calc_additional_case_prob_analytical(
            incidence_vec=[1],
            rep_no_func=lambda t: 0.5,
            serial_interval_dist_vec=[1.0],
            t_calc=1,
            additional_dims=cast(Any, "nope"),
        )


def test_calc_additional_case_prob_analytical_all_zero_incidence_returns_zero():
    prob = acp.calc_additional_case_prob_analytical(
        incidence_vec=[0, 0, 0],
        rep_no_func=lambda t: 1.0,
        serial_interval_dist_vec=[1.0],
        t_calc=1,
    )
    assert prob == 0


def test_calc_additional_case_prob_simulation_parallel_false_deterministic_zero_prob(
    rng,
):
    prob = acp.calc_additional_case_prob_simulation(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.0 if np.isscalar(t) else np.zeros_like(t, dtype=float),
        serial_interval_dist_vec=[1.0],
        t_calc=np.array([1, 2], dtype=int),
        n_sims=20,
        rng=rng,
        parallel=False,
    )
    assert isinstance(prob, np.ndarray)
    assert prob.shape == (2,)
    np.testing.assert_allclose(prob, np.array([0.0, 0.0]))


def test_additional_cases_one_sim_uses_run_renewal_model(monkeypatch):
    called = {}

    def fake_run_renewal_model(**kwargs):
        called.update(kwargs)
        return np.array([1, 0, 1])

    monkeypatch.setattr(acp, "run_renewal_model", fake_run_renewal_model)

    out = acp._additional_cases_one_sim(
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


def test_additional_cases_one_sim_raises_when_t_calc_does_not_match_incidence_length():
    with pytest.raises(ValueError, match="does not match length of incidence_vec"):
        acp._additional_cases_one_sim(
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


def test_calc_decision_delay_scalar_threshold():
    delay = acp.calc_decision_delay(
        prob_vec=np.array([0.2, 0.04, 0.01]),
        perc_risk_threshold=5,
        delay_of_first_prob=1,
    )
    assert delay == 2


def test_calc_decision_delay_vector_thresholds():
    delay = acp.calc_decision_delay(
        prob_vec=np.array([0.2, 0.04, 0.01]),
        perc_risk_threshold=np.array([5, 2]),
        delay_of_first_prob=1,
    )
    np.testing.assert_array_equal(delay, np.array([2, 3]))


def test_calc_decision_delay_raises_when_prob_never_below_threshold():
    with pytest.raises(ValueError, match="does not drop below"):
        acp.calc_decision_delay(
            prob_vec=np.array([0.2, 0.15, 0.1]),
            perc_risk_threshold=5,
            delay_of_first_prob=1,
        )


def test_simulation_prob_t_calc_zero_should_not_crash(rng):
    prob = acp.calc_additional_case_prob_simulation(
        incidence_vec=[1],
        rep_no_func=lambda t: 0.0 if np.isscalar(t) else np.zeros_like(t, dtype=float),
        serial_interval_dist_vec=[1.0],
        t_calc=np.array([0], dtype=int),
        n_sims=5,
        rng=rng,
        parallel=False,
    )
    assert isinstance(prob, np.ndarray)
    np.testing.assert_allclose(prob, np.array([1.0]))
